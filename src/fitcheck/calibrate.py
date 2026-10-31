"""fitcheck calibrate RESULTS: compare predictions with measured runs, then fit
the CALIBRATE constants into calibration.json.

RESULTS is the results.jsonl that calibrate/sweep.py writes, one line per run.
For every run this prints the predicted and measured peaks. It then fits:

- RESERVED_OVER_ALLOCATED, the factor from what PyTorch allocates to what its
  caching allocator reserves from the device, by least squares on the relative
  error against each run's measured max_memory_reserved;
- CUDA_CONTEXT_BYTES, device memory outside the allocator (CUDA context, cuBLAS
  and NCCL buffers): the median over ranks of device memory in use minus memory
  reserved, at the end of each run.

The accuracy it reports is held out: for each model, the factor is fitted on the
other models' runs and tested on this one's. calibration.json then uses every
valid run. A run is valid when it finished with at least MIN_APPLIED_STEPS
optimizer steps that fp16 overflow did not skip.

It refuses to fit until the CUDA saved-tensor check (scripts/saved_tensors.py
--device cuda --json) has passed on the same torch as the runs, because the
fitted factor would otherwise absorb a formula error. --ignore-cuda-check fits
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
TARGET = 0.10  # CLAUDE.md: within 10% of measured reserved memory


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
    held_out = _held_out(valid)
    _print_report(rows)
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

    ratio = _fit_ratio(valid)
    outside = [x for r in valid for x in r["outside"]]
    context = round(statistics.median(outside))
    print()
    print(f"Fitted on {len(valid)} valid runs of {len(rows)}:")
    print(f"  reserved / allocated  {ratio:.4f}")
    print(f"  CUDA context          {context / GIB:.2f} GiB (median over ranks)")
    for model, h in held_out.items():
        print(
            f"Held out {model}: factor {h['reserved_over_allocated']:.4f} from "
            f"{', '.join(h['fit_on'])}; median error {h['median_abs_error']:.1%}, "
            f"max {h['max_abs_error']:.1%}, {h['within_10_percent']} of "
            f"{h['runs']} within {TARGET:.0%}"
        )
    if not held_out:
        print("No held-out error: the valid runs cover only one model.")
    _print_verdicts(rows, ratio, context)

    envs = [run["env"] for run in runs if run.get("env")]

    def values(key):
        return sorted({e[key] for e in envs if e.get(key)})

    dates = values("date")
    calibration = {
        "reserved_over_allocated": round(ratio, 4),
        "cuda_context_bytes": context,
        "held_out": held_out,
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
        # Bare theory: without the constants being fitted.
        "predicted": est.gpu_total - est.overhead,
        "valid": run["status"] == "ok" and applied >= MIN_APPLIED_STEPS,
        "total_memory": min((r["total_memory"] for r in ranks), default=None),
    }
    if run["status"] == "ok":
        # The rank that needed the most, as that is the one that runs out.
        row["allocated"] = max(r["max_allocated"] for r in ranks)
        row["reserved"] = max(r["max_reserved"] for r in ranks)
        row["outside"] = [r["device_used_end"] - r["reserved_end"] for r in ranks]
    return row


def _fit_ratio(rows: list[dict]) -> float:
    """The factor r minimizing sum(((r * predicted - reserved) / reserved) ** 2)."""
    x = [r["predicted"] / r["reserved"] for r in rows]
    return sum(x) / sum(v * v for v in x)


def _held_out(valid: list[dict]) -> dict:
    """Per model: the factor fitted on the other models, and its error on this one.

    Also stores each test row's error under "held_out_error" for the report.
    """
    out = {}
    for model in sorted({r["model"] for r in valid}):
        train = [r for r in valid if r["model"] != model]
        test = [r for r in valid if r["model"] == model]
        if not train:
            continue
        ratio = _fit_ratio(train)
        errors = []
        for r in test:
            r["held_out_error"] = ratio * r["predicted"] / r["reserved"] - 1
            errors.append(abs(r["held_out_error"]))
        out[model] = {
            "fit_on": sorted({r["model"] for r in train}),
            "reserved_over_allocated": round(ratio, 4),
            "runs": len(test),
            "median_abs_error": round(statistics.median(errors), 4),
            "max_abs_error": round(max(errors), 4),
            "within_10_percent": sum(e <= TARGET for e in errors),
        }
    return out


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
        f"{'error':>8}{'reserved':>10}{'held out':>10}"
    )
    for r in rows:
        line = f"{r['run']:<42} {r['status']:<8}{r['applied']:>7}"
        line += f"{r['predicted'] / GIB:>11.2f}"
        if "allocated" in r:
            line += f"{r['allocated'] / GIB:>11.2f}"
            line += f"{r['predicted'] / r['allocated'] - 1:>+8.1%}"
            line += f"{r['reserved'] / GIB:>10.2f}"
        if "held_out_error" in r:
            line += f"{r['held_out_error']:>+10.1%}"
        if r["status"] == "ok" and not r["valid"]:
            line += f"  (invalid: under {MIN_APPLIED_STEPS} applied steps)"
        print(line)
    print(
        "GiB per GPU. error: bare prediction against measured allocated. "
        "held out: fitted prediction against measured reserved."
    )


def _print_verdicts(rows: list[dict], ratio: float, context: int) -> None:
    """Whether the fitted estimate gets fits / doesn't fit right on every run."""
    wrong = []
    checked = 0
    for r in rows:
        if r["status"] not in ("ok", "oom") or r["total_memory"] is None:
            continue
        checked += 1
        fits = ratio * r["predicted"] + context <= r["total_memory"]
        if fits != (r["status"] == "ok"):
            wrong.append(
                f"{r['run']} ({r['status']}, predicted {'fit' if fits else 'no fit'})"
            )
    print(f"Fits verdict right on {checked - len(wrong)} of {checked} runs.")
    for w in wrong:
        print(f"  wrong: {w}")
