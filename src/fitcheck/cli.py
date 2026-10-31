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
    parser.add_argument(
        "--seq-len", type=int, default=2048, help="tokens per sequence (default 2048)"
    )
    parser.add_argument(
        "--micro-batch", type=int, default=1, help="sequences per GPU per step"
    )
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument(
        "--checkpointing", action="store_true", help="activation checkpointing"
    )
    args = parser.parse_args(argv)

    try:
        setup = Setup(
            num_gpus=args.gpus,
            zero_stage=args.zero,
            offload_optimizer=args.offload_optimizer,
            offload_param=args.offload_param,
            lora_rank=args.lora_rank,
            dtype=args.dtype,
            micro_batch=args.micro_batch,
            seq_len=args.seq_len,
            attn=args.attn,
            checkpointing=args.checkpointing,
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
    ckpt = "checkpointing" if args.checkpointing else "no checkpointing"
    print(
        f"micro-batch {args.micro_batch} x {args.seq_len} tokens, "
        f"{args.attn} attention, {ckpt}"
    )
    print()
    print("Per GPU               GiB")
    for label, value in [
        ("weights", est.weights),
        ("gradients", est.grads),
        ("optimizer", est.optimizer),
        ("activations", est.activations),
        ("logits and loss", est.logits),
        ("overhead", est.overhead),
        ("total", est.gpu_total),
    ]:
        shown = "n/a" if value is None else f"{value / GIB:.2f}"
        print(f"  {label:<16}{shown:>7}")
    if est.cpu:
        print(f"CPU RAM for offload, all ranks: {est.cpu / GIB:.2f} GiB")
    print()
    for reason in model.unmodeled:
        print(f"Not modeled: {reason}.")
    if args.lora_rank is not None:
        # Group targets missing from the same layers into one line.
        gaps = {}
        for target, layers in model.lora_missing.items():
            gaps.setdefault(tuple(layers), []).append(target)
        for layers, targets in gaps.items():
            print(
                f"LoRA: {len(layers)} of {model.num_layers} layers have no "
                f"{', '.join(targets)} (layers {_ranges(layers)}); only the "
                "targets found are counted."
            )
    if est.activations is not None and est.step > est.activations + est.logits:
        print(
            "The peak is at the optimizer step: its fp32 gradient copy "
            f"({est.step / GIB:.2f} GiB) counts instead of activations."
        )
    if model.family == "qwen3_5":
        print(
            "Linear-attention layers are counted for transformers' PyTorch "
            "fallback, used when flash-linear-attention is not installed."
        )
    if args.gpu_mem is not None:
        fits = est.gpu_total <= args.gpu_mem * GIB
        # What is left out only adds memory, so "no" holds; "yes" would not.
        if not fits and model.unmodeled:
            verdict = "no, even without what is not modeled"
        elif not fits:
            verdict = "no"
        elif model.unmodeled:
            verdict = "unknown, because the estimate leaves out what is listed above"
        else:
            verdict = "yes"
        print(f"Fits in {args.gpu_mem:g} GiB: {verdict}")
    print("Not yet calibrated against measured runs.")


def _count(n: int) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def _ranges(ns) -> str:
    """[0, 1, 2, 4, 5, 7] -> "0-2, 4-5, 7"."""
    out, start = [], ns[0]
    for a, b in zip(ns, [*ns[1:], None]):
        if b != a + 1:
            out.append(f"{start}-{a}" if a > start else f"{start}")
            start = b
    return ", ".join(out)
