"""Compare FitCheck's predicted allocated memory with Ratchet's measured T4 runs.

    uv run python scripts/compare_memprobe.py [results.json]

The rows come from Ratchet's train/memprobe.py (calibrate/results/ratchet-memprobe-2026-11-12.json):
one Tesla T4, plain PyTorch + PEFT LoRA r=16 on q/k/v/o, torch.optim.AdamW, fp16 weights, SDPA,
use_reentrant=False checkpointing, random ids, 3 steps, torch 2.11.0, transformers 5.22.2.
answer_only rows use logits_to_keep=65 and a user-side .float() + F.cross_entropy on 64 tokens,
which is Setup.loss_tokens=64.

How this differs from FitCheck's DeepSpeed setup:
- torch AdamW keeps its moments in the parameter dtype and has no fp32 master copy, while
  FitCheck counts 2 + 2 + 12 bytes per trainable parameter. PEFT may keep adapters in fp32
  (not checked here), which gives 4 + 4 + 8 = 16 bytes, the same. If they are fp16 it is 8
  bytes. Either way the difference is under 0.02 GiB for ~1M LoRA parameters (0.8B and 2B
  q/k/v/o) and under 0.13 GiB for the 16.8M of the all-linear row.
- There is no DeepSpeed fp32 gradient copy at the step. FitCheck's `step` term does not set
  the peak in these rows (activations are larger).

flash-linear-attention: every row has it installed except the second 0.8B seq-4096 row
(fla=false). FitCheck's Qwen3.5 linear-attention formula counts transformers' PyTorch
fallback, so that row is the matching one. Both 0.8B seq-4096 rows peak at the same
allocated bytes, so the package did not change peak allocated memory there.
"""

import dataclasses
import json
import sys
from pathlib import Path

from fitcheck import hub
from fitcheck.estimate import Setup, estimate
from fitcheck.model import from_config

GIB = 1024**3
DEFAULT = (
    Path(__file__).parent.parent / "calibrate/results/ratchet-memprobe-2026-11-12.json"
)
HAS_LOSS_TOKENS = "loss_tokens" in {f.name for f in dataclasses.fields(Setup)}
QKVO = ["q_proj", "k_proj", "v_proj", "o_proj"]


def predicted_gib(model, setup):
    est = estimate(model, setup)
    allocated = getattr(est, "allocated", None)
    if allocated is None:  # this branch has no Estimate.allocated yet
        allocated = est.weights + est.grads + est.optimizer + est.peak_extra
    return allocated / GIB


def main(path):
    rows = json.loads(Path(path).read_text())
    models = {}
    print(
        f"{'#':>2} {'model':<14} {'seq':>5} {'ckpt':<5} {'fla':<5} {'pred':>6} {'meas':>6} {'err':>7}  note"
    )
    for r in rows:
        head = (
            f"{r['index']:>2} {r['model'].split('/')[1]:<14} {r['seq_len']:>5} "
            f"{r['checkpointing']!s:<5} {r['fla']!s:<5}"
        )
        if r["status"] != "ok":
            print(
                f"{head} {'-':>6} {'-':>6} {'-':>7}  not compared: {r['status']}, no peak measured"
            )
            continue
        meas = r["peak_allocated_gib"]
        if r["targets"] != QKVO:
            print(
                f"{head} {'-':>6} {meas:6.2f} {'-':>7}  not compared: LoRA on {r['targets']}, FitCheck models q/k/v/o only"
            )
            continue
        if r["answer_only"] and not HAS_LOSS_TOKENS:
            print(f"{head} {'-':>6} {meas:6.2f} {'-':>7}  needs --loss-tokens")
            continue
        key = (r["model"], r["revision"])
        if key not in models:
            config, tensors = hub.load(r["model"], revision=r["revision"])
            models[key] = from_config(r["model"], config, tensors)
        kw = {"loss_tokens": 64} if r["answer_only"] else {}
        setup = Setup(
            num_gpus=1,
            zero_stage=0,
            lora_rank=r["lora_rank"],
            dtype="fp16",
            seq_len=r["seq_len"],
            micro_batch=r["micro_batch"],
            checkpointing=r["checkpointing"],
            **kw,
        )
        pred = predicted_gib(models[key], setup)
        print(f"{head} {pred:6.2f} {meas:6.2f} {100 * (pred - meas) / meas:+6.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
