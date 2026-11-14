"""One calibration run: a few fp16 training steps under DeepSpeed, measuring
peak GPU memory on every rank.

calibrate/sweep.py launches it with the DeepSpeed launcher, one process per GPU:

    deepspeed --num_gpus 2 --master_port 29500 calibrate/run_one.py \\
        --model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 \\
        --zero 2 --seq-len 1024 --out-dir /kaggle/working/runs/0.5B-z2-s1024-mb1-sdpa-g2

Each rank writes rank<N>.json to --out-dir, also when it runs out of memory, so
one failing rank can't stop the others from reporting. sweep.py merges the files
into one line of results.jsonl.

The setup is the one fitcheck estimates: DeepSpeed casts every parameter to fp16,
the model computes the loss from labels, the tokens are random with no padding,
and the outputs stay alive until backward. The optimizer is AdamW from the
DeepSpeed config, so FusedAdam on the GPU, or DeepSpeedCPUAdam with optimizer
offload (get_optimizer_configuration in deepspeed/runtime/engine.py).
"""

import argparse
import dataclasses
import datetime
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import traceback

import deepspeed
import torch
from transformers import AutoModelForCausalLM

from fitcheck import recommend
from fitcheck.estimate import Setup
from fitcheck.model import LORA_TARGETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--zero", type=int, default=0)
    p.add_argument("--offload-optimizer", action="store_true")
    p.add_argument("--offload-param", action="store_true")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--checkpointing", action="store_true")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--trace",
        action="store_true",
        help="also write each rank's allocator history to trace<N>.pickle",
    )
    p.add_argument("--local_rank", type=int, default=0)  # added by the launcher
    args = p.parse_args()

    # The launcher sets these (deepspeed/launcher/launch.py).
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", str(args.local_rank))))
    setup = Setup(
        num_gpus=int(os.environ.get("WORLD_SIZE", "1")),
        zero_stage=args.zero,
        offload_optimizer=args.offload_optimizer,
        offload_param=args.offload_param,
        lora_rank=args.lora_rank,
        dtype="fp16",  # the T4 has no native bf16
        micro_batch=args.micro_batch,
        seq_len=args.seq_len,
        attn=args.attn,
        checkpointing=args.checkpointing,
        grad_accum=args.grad_accum,
    )
    free, total = torch.cuda.mem_get_info()
    record = {
        "rank": rank,
        "model": args.model,
        "revision": args.revision,
        "setup": dataclasses.asdict(setup),
        "steps": args.steps,
        "env": environment(),
        "total_memory": total,
        # Before loading anything: the CUDA context, plus anything a previous
        # run left behind.
        "device_used_start": total - free,
    }
    if args.trace:
        # Every allocation and free from here on, with Python stacks, so the
        # blocks that stay cached after deepspeed.initialize can be named. View
        # the pickle at https://pytorch.org/memory_viz ("Understanding CUDA
        # Memory Usage" in the PyTorch docs).
        torch.cuda.memory._record_memory_history(
            enabled="all", context="alloc", stacks="python", max_entries=200_000
        )
    try:
        train(args, setup, record)
        record["status"] = "ok"
    except Exception as e:  # noqa: BLE001  every failure must reach the rank file
        # Some DeepSpeed paths re-raise CUDA OOM as a plain RuntimeError.
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e)
        record["status"] = "oom" if oom else "error"
        record["error"] = traceback.format_exc()[-4000:]
    # This rank's peak resident host memory, for comparison with fitcheck's CPU
    # estimate. ru_maxrss is in KiB on Linux (getrusage(2)).
    record["peak_host_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    os.makedirs(args.out_dir, exist_ok=True)
    if args.trace:
        trace = os.path.join(args.out_dir, f"trace{rank}.pickle")
        torch.cuda.memory._dump_snapshot(trace)
    path = os.path.join(args.out_dir, f"rank{rank}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(record, f)
    os.replace(path + ".tmp", path)
    if record["status"] != "ok":
        # A non-zero exit makes the launcher stop the other ranks, which may be
        # waiting for this one in a collective.
        sys.exit(1)


def train(args, setup, record):
    record["phase"] = "load"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.float16,
        attn_implementation=args.attn,
    )
    # from_pretrained returns the model in eval mode (modeling_utils.py), and
    # GradientCheckpointingLayer.__call__ only checkpoints when self.training
    # (transformers/modeling_layers.py). DeepSpeed's engine doesn't call train().
    model.train()
    vocab = model.config.vocab_size
    if args.checkpointing:
        # Defaults to use_reentrant=False (gradient_checkpointing_enable in
        # transformers/modeling_utils.py), as docs/memory-model.md assumes.
        model.gradient_checkpointing_enable()
    if args.lora_rank:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_rank, target_modules=list(LORA_TARGETS), lora_dropout=0.0
        )
        model = get_peft_model(model, lora)

    config = ds_config(setup)
    record["ds_config"] = config
    record["phase"] = "initialize"
    engine, *_ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=config,
    )
    torch.cuda.synchronize()
    record["after_init_allocated"] = torch.cuda.memory_allocated()
    record["after_init_reserved"] = torch.cuda.memory_reserved()

    generator = torch.Generator(device="cuda").manual_seed(record["rank"])
    shape = (args.micro_batch, args.seq_len)
    ids = torch.randint(0, vocab, shape, device="cuda", generator=generator)

    # Peaks per phase of each step: forward plus backward, then the optimizer
    # step. Resetting the peak between phases keeps them apart; the allocator
    # never returns cached memory, so reserved memory only grows.
    record.update(peak_fwd_bwd=[], peak_step=[], applied=[], loss=[])
    max_reserved = 0
    torch.cuda.reset_peak_memory_stats()
    for i in range(args.steps):
        record["phase"] = f"step {i} forward and backward"
        # With --grad-accum N, N micro-batches per step. engine.step() after each
        # one only counts the micro-step until the last, where it updates the
        # weights (is_gradient_accumulation_boundary in engine.py), so the last
        # call is kept apart below. The peak covers all N micro-batches, the last
        # one being where earlier gradients and the activations coexist.
        for micro in range(args.grad_accum):
            out = engine(input_ids=ids, labels=ids)
            engine.backward(out.loss)
            if micro < args.grad_accum - 1:
                del out
                engine.step()
        torch.cuda.synchronize()
        record["peak_fwd_bwd"].append(torch.cuda.max_memory_allocated())
        max_reserved = max(max_reserved, torch.cuda.max_memory_reserved())
        record["loss"].append(out.loss.item())
        del out  # the logits are freed before the optimizer step
        torch.cuda.reset_peak_memory_stats()

        record["phase"] = f"step {i} optimizer"
        skipped = engine.skipped_steps
        engine.step()
        torch.cuda.synchronize()
        record["peak_step"].append(torch.cuda.max_memory_allocated())
        max_reserved = max(max_reserved, torch.cuda.max_memory_reserved())
        # DeepSpeed counts a step skipped on fp16 overflow (engine.py); such a
        # step never makes the fp32 gradient copy, so it can't show that peak.
        record["applied"].append(engine.skipped_steps == skipped)
        torch.cuda.reset_peak_memory_stats()

    record["phase"] = "done"
    record["max_allocated"] = max(record["peak_fwd_bwd"] + record["peak_step"])
    record["max_reserved"] = max_reserved
    # Between steps: weights, optimizer states and any persistent buffers.
    record["between_steps_allocated"] = torch.cuda.memory_allocated()
    record["reserved_end"] = torch.cuda.memory_reserved()
    free, total = torch.cuda.mem_get_info()
    # Device memory outside PyTorch's allocator: the CUDA context, cuBLAS and
    # NCCL buffers. fitcheck calibrate fits CUDA_CONTEXT_BYTES from this.
    record["device_used_end"] = total - free


def ds_config(setup):
    """The config fitcheck recommend writes, with one change for calibration."""
    config = recommend.ds_config(setup)
    # Start the dynamic loss scale at 1 (2**0) instead of 2**16, so the first
    # steps don't overflow. A skipped step returns before the fp32 gradient
    # copy (FP16_Optimizer.step in runtime/fp16/fused_optimizer.py), so its
    # peak would be too low. Gradient quality doesn't matter here.
    config["fp16"]["initial_scale_power"] = 0
    return config


def environment():
    here = os.path.dirname(os.path.abspath(__file__))

    def first_line(*cmd):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, cwd=here, check=False
            ).stdout
        except OSError:
            return None
        return out.strip().splitlines()[0] if out.strip() else None

    return {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "gpu": torch.cuda.get_device_name(),
        "capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "driver": first_line(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deepspeed": deepspeed.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "python": platform.python_version(),
        "host_ram": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "fitcheck_commit": first_line("git", "rev-parse", "HEAD"),
    }


if __name__ == "__main__":
    main()
