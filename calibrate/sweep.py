"""Run the Phase 3 calibration sweep on Kaggle's 2x T4 and append each run to
results.jsonl.

    python calibrate/sweep.py --results /kaggle/working/results.jsonl \\
        --work-dir /kaggle/working/runs

Each run is one launch of calibrate/run_one.py through the DeepSpeed launcher.
A run still going after TIMEOUT_S is killed with all its ranks. Runs already in
results.jsonl are skipped, so the sweep resumes after a restart. --dry-run prints
every command without running anything.
"""

import argparse
import dataclasses
import json
import os
import shlex
import signal
import subprocess
import time

from fitcheck.estimate import Setup

# Pinned revisions, so a re-upload can't change what was measured.
MODELS = {
    "0.5B": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    "1.5B": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
}
STEPS = 8
TIMEOUT_S = 900
RUN_ONE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_one.py")


def grid() -> list[tuple[str, Setup]]:
    """The 55 runs. Everything is fp16, SDPA, micro-batch 1 and 2 GPUs unless
    a block changes it. fitcheck predicts that 7 won't fit in a T4's 15 GiB;
    they stay in to check the fits / doesn't-fit verdict."""
    runs = []
    for size in MODELS:
        # The base for the other blocks: full fine-tuning of 1.5B only fits
        # with the optimizer offloaded.
        base = {"zero_stage": 2, "offload_optimizer": size == "1.5B"}
        zero = [(0, False, False), (1, False, False), (1, True, False)]
        zero += [(2, False, False), (2, True, False)]
        zero += [(3, False, False), (3, True, False), (3, True, True)]
        if size == "1.5B":
            zero.remove((1, False, False))  # far over, like z0 and z2 already are
        for stage, opt, param in zero:
            runs.append(
                (
                    size,
                    {
                        "zero_stage": stage,
                        "offload_optimizer": opt,
                        "offload_param": param,
                    },
                )
            )
        for seq_len in (512, 1024, 2048, 4096):
            for ckpt in (False, True):
                runs.append((size, dict(base, seq_len=seq_len, checkpointing=ckpt)))
        for stage in (0, 2):
            for seq_len in (1024, 2048):
                for ckpt in (False, True):
                    lora = {"zero_stage": stage, "lora_rank": 16, "seq_len": seq_len}
                    runs.append((size, {**lora, "checkpointing": ckpt}))
        for seq_len in (1024, 2048):
            runs.append((size, dict(base, seq_len=seq_len, attn="eager")))
        for ckpt in (False, True):
            runs.append((size, dict(base, micro_batch=2, checkpointing=ckpt)))
        # One GPU: no NCCL buffers, for the context term.
        runs.append((size, {"zero_stage": 0, "lora_rank": 16, "num_gpus": 1}))

    unique = {}
    for size, kw in runs:
        setup = Setup(**{"num_gpus": 2, "dtype": "fp16", "seq_len": 1024, **kw})
        unique.setdefault(run_id(size, setup), (size, setup))
    # 1.5B with offload last: those runs need the most host RAM (Kaggle has
    # about 29 GB), so if one takes the session down, every other run is
    # already in results.jsonl. sorted is stable, so the rest keep their order.
    return sorted(unique.values(), key=lambda r: _host_heavy(*r))


def _host_heavy(size: str, s: Setup) -> bool:
    return size == "1.5B" and (s.offload_optimizer or s.offload_param)


def run_id(size: str, s: Setup) -> str:
    parts = [size, f"z{s.zero_stage}"]
    parts += ["oo"] * s.offload_optimizer + ["op"] * s.offload_param
    parts += [f"lora{s.lora_rank}"] if s.lora_rank else []
    parts += [f"s{s.seq_len}", f"mb{s.micro_batch}", s.attn]
    parts += ["ckpt"] * s.checkpointing + [f"g{s.num_gpus}"]
    return "-".join(parts)


def command(size: str, s: Setup, out_dir: str, port: int) -> list[str]:
    model, revision = MODELS[size]
    cmd = ["deepspeed", "--num_gpus", str(s.num_gpus), "--master_port", str(port)]
    cmd += [RUN_ONE, "--model", model, "--revision", revision]
    cmd += ["--zero", str(s.zero_stage)]
    cmd += ["--offload-optimizer"] * s.offload_optimizer
    cmd += ["--offload-param"] * s.offload_param
    cmd += ["--lora-rank", str(s.lora_rank)] if s.lora_rank else []
    cmd += ["--seq-len", str(s.seq_len), "--micro-batch", str(s.micro_batch)]
    cmd += ["--attn", s.attn] + ["--checkpointing"] * s.checkpointing
    return cmd + ["--steps", str(STEPS), "--out-dir", out_dir]


def launch(cmd: list[str], out_dir: str) -> tuple[int | None, float]:
    """Exit code (None on timeout) and seconds taken."""
    env = dict(os.environ)
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")  # JIT-build DeepSpeed ops for T4
    start = time.time()
    with open(os.path.join(out_dir, "log.txt"), "w") as log:
        # Its own session, so on timeout one killpg reaches the launcher and
        # every rank it started (runner.py and launch.py use plain Popen).
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            code = proc.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = None
    return code, time.time() - start


def merge(size: str, s: Setup, out_dir: str, code, seconds: float) -> dict:
    """One results.jsonl line from the per-rank files run_one.py wrote."""
    ranks = []
    for r in range(s.num_gpus):
        path = os.path.join(out_dir, f"rank{r}.json")
        if os.path.exists(path):
            with open(path) as f:
                ranks.append(json.load(f))
    statuses = {r["status"] for r in ranks}
    if "oom" in statuses:
        status = "oom"
    elif code is None:
        status = "timeout"
    elif code == 0 and len(ranks) == s.num_gpus and statuses == {"ok"}:
        status = "ok"
    else:
        status = "error"  # includes a rank killed by the host's OOM killer
    model, revision = MODELS[size]
    shared = ("model", "revision", "setup", "steps", "env", "ds_config")
    line = {
        "run": run_id(size, s),
        "status": status,
        "exit_code": code,
        "seconds": round(seconds),
        "model": model,
        "revision": revision,
        "setup": dataclasses.asdict(s),
        "steps": STEPS,
        # The fewest applied steps on any rank; fitcheck calibrate drops runs
        # with fewer than 4.
        "applied_steps": (
            min(sum(r["applied"]) for r in ranks) if status == "ok" else None
        ),
        "env": ranks[0].get("env") if ranks else None,
        "ds_config": ranks[0].get("ds_config") if ranks else None,
        "ranks": [{k: v for k, v in r.items() if k not in shared} for r in ranks],
    }
    if status != "ok":
        log = os.path.join(out_dir, "log.txt")
        if os.path.exists(log):
            with open(log, errors="replace") as f:
                line["log_tail"] = f.read()[-3000:]
    return line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="results.jsonl to append to")
    p.add_argument("--work-dir", required=True, help="per-run rank files and logs")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = p.parse_args()

    done = set()
    if os.path.exists(args.results):
        with open(args.results) as f:
            done = {json.loads(line)["run"] for line in f if line.strip()}
    runs = grid()
    for i, (size, setup) in enumerate(runs):
        rid = run_id(size, setup)
        if rid in done:
            continue
        out_dir = os.path.join(args.work_dir, rid)
        # A fresh port per run, so a socket left over from a killed run can't
        # block the next rendezvous.
        cmd = command(size, setup, out_dir, 29500 + i)
        print(f"[{i + 1}/{len(runs)}] {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        if os.path.isdir(out_dir):  # rank files from an interrupted attempt
            for name in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, name))
        os.makedirs(out_dir, exist_ok=True)
        code, seconds = launch(cmd, out_dir)
        line = merge(size, setup, out_dir, code, seconds)
        with open(args.results, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"    {line['status']} in {seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
