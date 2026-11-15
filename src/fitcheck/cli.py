"""Command line: fitcheck MODEL [options], fitcheck recommend MODEL [options],
or fitcheck calibrate RESULTS."""

import argparse
import json
import sys

from fitcheck import hub, recommend
from fitcheck.estimate import CALIBRATION, DTYPE_BYTES, Setup, estimate, unmodeled
from fitcheck.model import LORA_TARGETS, from_config

GIB = 1024**3


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["calibrate"]:
        from fitcheck import calibrate

        return calibrate.main(argv[1:])
    if argv[:1] == ["recommend"]:
        return recommend_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="fitcheck",
        description="Estimate per-GPU training memory for a Hugging Face model.",
    )
    _add_common(parser)
    parser.add_argument(
        "--zero", type=int, default=0, choices=range(4), help="ZeRO stage"
    )
    parser.add_argument("--offload-optimizer", action="store_true")
    parser.add_argument("--offload-param", action="store_true")
    parser.add_argument(
        "--checkpointing", action="store_true", help="activation checkpointing"
    )
    args = parser.parse_args(argv)
    setup, model = _load(
        args,
        zero_stage=args.zero,
        offload_optimizer=args.offload_optimizer,
        offload_param=args.offload_param,
        checkpointing=args.checkpointing,
    )
    est = estimate(model, setup)

    _print_model(model)
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
        f"{args.attn} attention, {ckpt}, gradient accumulation {args.grad_accum}"
        + (f", loss on last {args.loss_tokens} tokens" if args.loss_tokens else "")
    )
    print()
    # The breakdown of whichever moment is the peak, so the rows add up.
    if est.peak_at == "backward":
        act, logits = est.activations or 0, est.logits or 0
        held = est.backward - est.weights - est.optimizer - est.buffers
        held -= est.gathered + est.bucket + act + logits
        rows = [
            ("weights", est.weights),
            ("gradients", held),
            ("optimizer", est.optimizer),
            ("DeepSpeed buffers", est.buffers + est.gathered + est.bucket),
            ("activations", est.activations),
            ("logits and loss", est.logits),
        ]
        where = "start of backward"
    else:
        rows = [
            ("weights", est.weights),
            ("gradients", est.grads),
            ("optimizer", est.optimizer),
            ("DeepSpeed buffers", est.buffers),
            ("fp32 grad copies", est.step),
        ]
        if est.activations is None:
            rows.append(("activations", None))
        where = "optimizer step"
    print(f"Per GPU, at the {where}   GiB")
    for label, value in rows + [("overhead", est.overhead), ("total", est.gpu_total)]:
        shown = "n/a" if value is None else f"{value / GIB:.2f}"
        print(f"  {label:<18}{shown:>7}")
    if est.cpu:
        print(f"CPU RAM for offload, all ranks: {est.cpu / GIB:.2f} GiB")
    print()
    _print_caveats(model, setup, args)
    gaps = unmodeled(model, setup)
    if est.peak_at == "step" and est.activations is not None:
        print(
            "The peak is at the optimizer step, after activations are freed: "
            "the gradients and their temporary fp32 copies count instead."
        )
    if args.gpu_mem is not None:
        fits = est.gpu_total <= args.gpu_mem * GIB
        # What is left out only adds memory, so "no" holds; "yes" would not.
        if not fits and gaps:
            verdict = "no, even without what is not modeled"
        elif not fits:
            verdict = "no"
        elif gaps:
            verdict = "unknown, because the estimate leaves out what is listed above"
        else:
            verdict = "yes"
        print(f"Fits in {args.gpu_mem:g} GiB: {verdict}")
    _print_calibration()


def recommend_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="fitcheck recommend",
        description="Rank the DeepSpeed setups that fit, cheapest first, and "
        "write the best one's ds_config.json.",
    )
    _add_common(parser)
    parser.add_argument(
        "--host-mem", type=float, help="host RAM in GiB, for all GPUs together"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="AdamW learning rate (default 1e-5)"
    )
    parser.add_argument(
        "--out", default="ds_config.json", help="where to write the config"
    )
    args = parser.parse_args(argv)
    if args.gpu_mem is None or args.gpu_mem <= 0:
        parser.error("--gpu-mem is required and must be above 0")
    if args.host_mem is not None and args.host_mem < 0:
        parser.error("--host-mem must not be negative")
    base, model = _load(args)
    host = None if args.host_mem is None else args.host_mem * GIB
    opts = recommend.options(model, base, args.gpu_mem * GIB, host)
    pick = recommend.best(opts)

    _print_model(model)
    task = "full fine-tune" if args.lora_rank is None else f"LoRA r={args.lora_rank}"
    print(
        f"{task} ({_count(opts[0].estimate.trainable_params)} trainable), "
        f"{args.dtype} on {args.gpus} GPU(s) of {args.gpu_mem:g} GiB, "
        f"micro-batch {args.micro_batch} x {args.seq_len} tokens, {args.attn} "
        f"attention, gradient accumulation {args.grad_accum}"
        + (f", loss on last {args.loss_tokens} tokens" if args.loss_tokens else "")
    )
    print()
    print(f"  {'setup, cheapest first':<52}{'GPU GiB':>8}{'host GiB':>10}  fits")
    for o in opts:
        e = o.estimate
        host_gib = f"{e.cpu / GIB:.2f}" if e.cpu else "-"
        fits = f"{o.fits} ({o.reason})" if o.reason else o.fits
        if o.fits == "yes" and e.cpu:
            fits = "yes (GPU; host RAM unchecked)"
        mark = "*" if o is pick else " "
        print(
            f"{mark} {_setup_label(o.setup):<52}{e.gpu_total / GIB:>8.2f}"
            f"{host_gib:>10}  {fits}"
        )
    print()
    margin = ""
    if recommend.MARGIN:
        margin = f" plus {recommend.MARGIN:.1%}, the worst calibration error,"
    print(
        f"A setup fits when its peak allocated memory{margin} and the CUDA "
        f"context fit in {args.gpu_mem:g} GiB. The order is a rule of thumb: "
        "offload, then checkpointing, then a higher ZeRO stage."
    )
    print(
        "Host GiB is the offloaded state only. It leaves out the model as "
        "loaded on the CPU and the Python and CUDA runtimes, so it is a lower "
        "bound: a setup over --host-mem is ruled out, one under it is not "
        "checked. The measured offload runs needed about twice the estimate "
        "(docs/memory-model.md)."
    )
    _print_caveats(model, base, args)
    _print_calibration()
    print()
    if pick is None:
        if any(o.fits == "unknown" for o in opts):
            sys.exit(
                "fitcheck: can't recommend a setup, because the estimate leaves "
                "out what is listed above. No config written."
            )
        sys.exit("fitcheck: no setup fits. No config written.")

    try:
        with open(args.out, "w") as f:
            json.dump(recommend.ds_config(pick.setup, lr=args.lr), f, indent=2)
            f.write("\n")
    except OSError as err:
        sys.exit(f"fitcheck: can't write {args.out}: {err}")
    print(f"Best: {_setup_label(pick.setup)}. Wrote {args.out}.")
    if pick.estimate.cpu:
        print(
            f"It needs more than {pick.estimate.cpu / GIB:.2f} GiB of host RAM "
            "over all ranks; check that you have about twice that."
        )
    print("Also set in the training script, which the config can't hold:")
    print(f'  from_pretrained(..., attn_implementation="{args.attn}")')
    if pick.setup.checkpointing:
        # from_pretrained returns the model in eval mode, and HF only
        # checkpoints in training mode (transformers/modeling_layers.py).
        print("  model.gradient_checkpointing_enable(), then model.train()")
    if args.lora_rank is not None:
        print(
            f"  peft LoraConfig(r={args.lora_rank}, "
            f"target_modules={list(LORA_TARGETS)})"
        )
    if args.loss_tokens:
        print(f"  model(..., logits_to_keep={args.loss_tokens + 1})")
    if CALIBRATION:
        # Older torch runs SDPA's math path for grouped-query attention on GPUs
        # without flash attention (docs/memory-model.md, "Second check").
        prov = CALIBRATION["provenance"]
        versions = ", ".join(
            f"{lib} {'/'.join(prov[lib])}" for lib in ("torch", "deepspeed")
        )
        print(f"The estimate holds for the calibrated versions: {versions}.")


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options shared by the estimate and recommend commands."""
    parser.add_argument("model", help="Hub model id (org/name) or a local folder")
    parser.add_argument("--revision", default="main", help="Hub branch, tag or commit")
    parser.add_argument(
        "--gpus", type=int, default=1, help="number of GPUs (default 1)"
    )
    parser.add_argument("--gpu-mem", type=float, help="memory per GPU in GiB, e.g. 16")
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
    parser.add_argument(
        "--grad-accum", type=int, default=1, help="micro-batches per optimizer step"
    )
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument(
        "--loss-tokens",
        type=int,
        help="loss on the last N tokens only (logits_to_keep=N + 1); default all",
    )


def _load(args, **setup):
    """The Setup from the common options plus `setup`, and the model."""
    try:
        s = Setup(
            num_gpus=args.gpus,
            lora_rank=args.lora_rank,
            dtype=args.dtype,
            micro_batch=args.micro_batch,
            seq_len=args.seq_len,
            attn=args.attn,
            grad_accum=args.grad_accum,
            loss_tokens=args.loss_tokens,
            **setup,
        )
        config, tensors = hub.load(args.model, revision=args.revision)
        return s, from_config(args.model, config, tensors)
    except (ValueError, OSError) as err:
        sys.exit(f"fitcheck: {err}")


def _print_model(model) -> None:
    kv = f" ({model.num_kv_heads} KV)" if model.num_kv_heads != model.num_heads else ""
    print(
        f"{model.name}: {_count(model.num_params)} params, {model.num_layers} layers, "
        f"hidden {model.hidden_size}, {model.num_heads} heads{kv} of {model.head_dim}"
    )


def _print_caveats(model, setup, args) -> None:
    """What the estimate leaves out or assumes about this model and setup."""
    for reason in unmodeled(model, setup):
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
    if model.family == "qwen3_5":
        print(
            "Linear-attention layers are counted for transformers' PyTorch "
            "fallback, used when flash-linear-attention is not installed."
        )


def _setup_label(s: Setup) -> str:
    parts = [f"ZeRO-{s.zero_stage}"]
    if s.offload_param:
        parts.append("offload optimizer and params")
    elif s.offload_optimizer:
        parts.append("offload optimizer")
    if s.checkpointing:
        parts.append("checkpointing")
    return ", ".join(parts)


def _print_calibration() -> None:
    if CALIBRATION:
        prov = CALIBRATION["provenance"]
        err = CALIBRATION["allocated_error"]
        held = (
            f"; allocated memory within {err['max_abs_error']:.0%} on "
            f"{err['runs']} measured runs, the allocator may reserve more"
        )
        print(
            f"Calibrated on {', '.join(prov['models'])} runs on {', '.join(prov['gpu'])}"
            f"{held}. Other models and GPUs are extrapolated."
        )
    else:
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
