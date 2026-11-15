"""fitcheck calibrate RESULTS: compare predictions with measured runs, then fit
the CUDA context constant into calibration.json.

RESULTS is the results.jsonl that calibrate/sweep.py writes, one line per run.
For every run this prints the predicted peak allocated memory (what PyTorch
allocates, `Estimate.allocated`) against the measured max_memory_allocated, the
larger of the ranks. The target is within 2% of it. Reserved memory is printed
for information only: DeepSpeed's initialize leaves freed blocks in PyTorch's
cache, and PyTorch frees that cache and retries before it raises an OOM, so
reserved is not a limit.

The one fitted constant is CUDA_CONTEXT_BYTES, device memory outside the
allocator (CUDA context, cuBLAS and NCCL buffers): the median over ranks of
device memory in use minus memory reserved, at the end of each run. The fits
verdict is predicted allocated plus that constant against each GPU's memory.

A run is valid when it finished with at least MIN_APPLIED_STEPS optimizer steps
that fp16 overflow did not skip.

It refuses to fit until the CUDA saved-tensor check (scripts/saved_tensors.py
--device cuda --json) has passed on the same torch as the runs, because the
constant would otherwise sit next to a formula error. --ignore-cuda-check fits
anyway and records why in calibration.json.
"""

import argparse
import datetime
import hashlib
import json
import os
import statistics
import sys
from importlib.resources import files

from fitcheck import hub
from fitcheck.estimate import Setup, estimate
from fitcheck.model import from_config

GIB = 1024**3
MIN_APPLIED_STEPS = 4
TARGET = 0.02  # within 2% of measured allocated memory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fitcheck calibrate",
        description="Compare predictions with measured runs and fit calibration.json.",
    )
    parser.add_argument("results", help="results.jsonl from calibrate/sweep.py")
    parser.add_argument(
        "--saved-tensors",
        help="JSON from the CUDA saved-tensor check "
        "(default: saved_tensors_cuda.json next to RESULTS)",
    )
    parser.add_argument(
        "--out",
        default=str(files("fitcheck") / "calibration.json"),
        help="where to write the fitted constants (default: the package's own)",
    )
    parser.add_argument(
        "--ignore-cuda-check",
        action="store_true",
        help="fit even if the CUDA saved-tensor check is missing or failed",
    )
    args = parser.parse_args(argv)

    with open(args.results, "rb") as f:
        raw = f.read()
    runs = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    models = {}
    rows = [_row(run, models) for run in runs]
    valid = [r for r in rows if r["valid"]]
    _print_report(rows)
    _print_host_memory(rows)
    if not valid:
        sys.exit("fitcheck calibrate: no valid runs to fit")

    check_path = args.saved_tensors or os.path.join(
        os.path.dirname(args.results), "saved_tensors_cuda.json"
    )
    check, problem = _cuda_check(check_path, runs)
    if problem and not args.ignore_cuda_check:
        sys.exit(
            f"fitcheck calibrate: not fitting: {problem}. Fix or explain the "
            "formula first, or pass --ignore-cuda-check."
        )
    if problem:
        check["overridden"] = problem

    outside = [x for r in valid for x in r["outside"]]
    context = round(statistics.median(outside))
    errors = _allocated_error(valid)
    print()
    print(f"Fitted on {len(valid)} valid runs of {len(rows)}:")
    print(f"  CUDA context  {context / GIB:.2f} GiB (median over ranks)")
    print("Predicted allocated against measured allocated, absolute error:")
    for model, e in errors["per_model"].items():
        print(f"  {model}: {_summary(e)}")
    print(f"  all: {_summary(errors)}")
    _print_verdicts(rows, context)

    envs = [run["env"] for run in runs if run.get("env")]

    def values(key):
        return sorted({e[key] for e in envs if e.get(key)})

    dates = values("date")
    calibration = {
        "cuda_context_bytes": context,
        "allocated_error": errors,
        "provenance": {
            "fitted": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "results_file": os.path.basename(args.results),
            "results_sha256": hashlib.sha256(raw).hexdigest(),
            "runs": len(rows),
            "valid_runs": len(valid),
            "models": sorted({r["model"] for r in valid}),
            "measured": [dates[0], dates[-1]] if dates else None,
            **{
                key: values(key)
                for key in (
                    "gpu",
                    "driver",
                    "torch",
                    "cuda",
                    "deepspeed",
                    "transformers",
                    "peft",
                    "fitcheck_commit",
                )
            },
            "cuda_check": check,
        },
    }
    with open(args.out, "w") as f:
        json.dump(calibration, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out}")


def _row(run: dict, models: dict) -> dict:
    """Prediction and measurements for one run, in bytes."""
    key = (run["model"], run["revision"])
    if key not in models:
        config, tensors = hub.load(run["model"], revision=run["revision"])
        models[key] = from_config(run["model"], config, tensors)
    est = estimate(models[key], Setup(**run["setup"]))
    ranks = run.get("ranks") or []
    applied = run.get("applied_steps") or 0
    row = {
        "run": run["run"],
        "model": run["model"],
        "status": run["status"],
        "applied": applied,
        # Bare theory: PyTorch's peak allocated, without the CUDA context.
        "predicted": est.allocated,
        "valid": run["status"] == "ok" and applied >= MIN_APPLIED_STEPS,
        "total_memory": min((r["total_memory"] for r in ranks), default=None),
        "offload": run["setup"]["offload_optimizer"] or run["setup"]["offload_param"],
        # Offloaded optimizer and parameter state on the host, all ranks.
        "cpu_predicted": est.cpu,
        # Peak resident memory summed over ranks, unless a rank wrote nothing
        # (killed by the host's OOM killer, for instance).
        "cpu_measured": (
            sum(r["peak_host_rss"] for r in ranks)
            if len(ranks) == run["setup"]["num_gpus"]
            and all("peak_host_rss" in r for r in ranks)
            else None
        ),
    }
    if run["status"] == "ok":
        # The rank that needed the most, as that is the one that runs out.
        row["allocated"] = max(r["max_allocated"] for r in ranks)
        row["reserved"] = max(r["max_reserved"] for r in ranks)
        row["outside"] = [r["device_used_end"] - r["reserved_end"] for r in ranks]
    return row


def _allocated_error(valid: list[dict]) -> dict:
    """Median and max absolute error of predicted against measured allocated."""

    def stats(rows):
        e = [abs(r["predicted"] / r["allocated"] - 1) for r in rows]
        return {
            "median_abs_error": round(statistics.median(e), 4),
            "max_abs_error": round(max(e), 4),
            "within_target": sum(x <= TARGET for x in e),
            "runs": len(e),
        }

    per_model = {
        m: stats([r for r in valid if r["model"] == m])
        for m in sorted({r["model"] for r in valid})
    }
    return {**stats(valid), "per_model": per_model}


def _summary(e: dict) -> str:
    return (
        f"median {e['median_abs_error']:.1%}, max {e['max_abs_error']:.1%}, "
        f"{e['within_target']} of {e['runs']} within {TARGET:.0%}"
    )


def _cuda_check(path: str, runs: list[dict]) -> tuple[dict, str | None]:
    """What the CUDA saved-tensor check found, and why it blocks a fit, if it does."""
    if not os.path.exists(path):
        return {"file": None}, f"no CUDA saved-tensor check at {path}"
    with open(path, "rb") as f:
        raw = f.read()
    check = json.loads(raw)
    info = {
        "file": os.path.basename(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        **{k: check.get(k) for k in ("device", "gpu", "torch", "all_match")},
    }
    if check.get("device") != "cuda":
        return info, f"{path} is a {check.get('device')} check, not a CUDA one"
    if not check.get("all_match"):
        bad = [
            f"{c['config']} {'lora' if c['lora_rank'] else 'full'} {c['attn']}"
            for c in check["cases"]
            if not c["ok"]
        ]
        return (
            info,
            f"{len(bad)} CUDA saved-tensor case(s) don't match: {', '.join(bad)}",
        )
    torches = sorted({run["env"]["torch"] for run in runs if run.get("env")})
    if torches != [check.get("torch")]:
        return info, (
            f"the CUDA check ran on torch {check.get('torch')}, "
            f"the runs on {', '.join(torches)}"
        )
    return info, None


def _print_report(rows: list[dict]) -> None:
    print(
        f"{'run':<42} {'status':<8}{'applied':>7}{'predicted':>11}{'allocated':>11}"
        f"{'error':>8}{'reserved':>10}{'res/alloc':>10}"
    )
    for r in rows:
        line = f"{r['run']:<42} {r['status']:<8}{r['applied']:>7}"
        line += f"{r['predicted'] / GIB:>11.2f}"
        if "allocated" in r:
            line += f"{r['allocated'] / GIB:>11.2f}"
            line += f"{r['predicted'] / r['allocated'] - 1:>+8.1%}"
            line += f"{r['reserved'] / GIB:>10.2f}"
            line += f"{r['reserved'] / r['allocated']:>10.2f}"
        if r["status"] == "ok" and not r["valid"]:
            line += f"  (invalid: under {MIN_APPLIED_STEPS} applied steps)"
        print(line)
    print(
        "GiB per GPU. error: predicted against measured allocated, the target. "
        "reserved and res/alloc are for information only."
    )


def _print_host_memory(rows: list[dict]) -> None:
    """Offload runs: predicted against measured host memory. Reported, not fitted."""
    offload = [r for r in rows if r["offload"]]
    if not offload:
        return
    print()
    print("Host memory of the offload runs, all ranks (reported, not fitted):")
    print(f"{'run':<42} {'status':<8}{'predicted':>11}{'measured':>10}")
    for r in offload:
        measured = r["cpu_measured"]
        print(
            f"{r['run']:<42} {r['status']:<8}{r['cpu_predicted'] / GIB:>11.2f}"
            + (f"{measured / GIB:>10.2f}" if measured is not None else f"{'n/a':>10}")
        )
    print(
        "GiB. predicted: the offloaded optimizer and parameter state. measured: "
        "each rank's peak resident memory, summed, which also holds the model "
        "as loaded on the CPU and the Python, PyTorch and CUDA runtimes."
    )


def _print_verdicts(rows: list[dict], context: int) -> None:
    """Whether predicted allocated plus the CUDA context gets fits / doesn't fit right on every run."""
    wrong = []
    checked = 0
    for r in rows:
        if r["status"] not in ("ok", "oom") or r["total_memory"] is None:
            continue
        checked += 1
        fits = r["predicted"] + context <= r["total_memory"]
        if fits != (r["status"] == "ok"):
            wrong.append(
                f"{r['run']} ({r['status']}, predicted {'fit' if fits else 'no fit'})"
            )
    print(f"Fits verdict right on {checked - len(wrong)} of {checked} runs.")
    for w in wrong:
        print(f"  wrong: {w}")
