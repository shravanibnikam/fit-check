"""Command line: fitcheck MODEL [options]."""

import argparse
import sys

from fitcheck import hub
from fitcheck.estimate import DTYPE_BYTES, Setup, estimate
from fitcheck.model import from_config

GIB = 1024**3


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fitcheck",
        description="Estimate per-GPU training memory for a Hugging Face model.",
    )
    parser.add_argument("model", help="Hub model id (org/name) or a local folder")
    parser.add_argument("--revision", default="main", help="Hub branch, tag or commit")
    parser.add_argument(
        "--gpus", type=int, default=1, help="number of GPUs (default 1)"
    )
    parser.add_argument("--gpu-mem", type=float, help="memory per GPU in GiB, e.g. 16")
    parser.add_argument(
        "--zero", type=int, default=0, choices=range(4), help="ZeRO stage"
    )
    parser.add_argument("--offload-optimizer", action="store_true")
    parser.add_argument("--offload-param", action="store_true")
    parser.add_argument(
        "--lora-rank", type=int, help="LoRA rank on q/k/v/o; omit for full"
    )
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_BYTES))
    args = parser.parse_args(argv)

    try:
        setup = Setup(
            num_gpus=args.gpus,
            zero_stage=args.zero,
            offload_optimizer=args.offload_optimizer,
            offload_param=args.offload_param,
            lora_rank=args.lora_rank,
            dtype=args.dtype,
        )
        config, tensors = hub.load(args.model, revision=args.revision)
        model = from_config(args.model, config, tensors)
    except (ValueError, OSError) as err:
        sys.exit(f"fitcheck: {err}")
    est = estimate(model, setup)

    kv = f" ({model.num_kv_heads} KV)" if model.num_kv_heads != model.num_heads else ""
    print(
        f"{model.name}: {_count(model.num_params)} params, {model.num_layers} layers, "
        f"hidden {model.hidden_size}, {model.num_heads} heads{kv} of {model.head_dim}"
    )
    task = "full fine-tune" if args.lora_rank is None else f"LoRA r={args.lora_rank}"
    offload = [
        name
        for name, on in [
            ("optimizer", args.offload_optimizer),
            ("params", args.offload_param),
        ]
        if on
    ]
    print(
        f"{task} ({_count(est.trainable_params)} trainable), {args.dtype}, "
        f"ZeRO-{args.zero} on {args.gpus} GPU(s)"
        + (f", offload {' and '.join(offload)}" if offload else "")
    )
    print()
    print("Per GPU            GiB")
    for label, value in [
        ("weights", est.weights),
        ("gradients", est.grads),
        ("optimizer", est.optimizer),
        ("overhead", est.overhead),
        ("total", est.gpu_total),
    ]:
        print(f"  {label:<12}{value / GIB:>8.2f}")
    if est.cpu:
        print(f"CPU RAM for offload, all ranks: {est.cpu / GIB:.2f} GiB")
    print()
    if args.gpu_mem is not None:
        fits = est.gpu_total <= args.gpu_mem * GIB
        print(f"Fits in {args.gpu_mem:g} GiB: {'yes' if fits else 'no'}")
    print("Activations are not counted yet, so treat this as a lower bound.")


def _count(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"
