import json
import re
import sys
from pathlib import Path

import pytest
import sweep


def test_grid_is_the_55_planned_runs():
    runs = sweep.grid()
    ids = [sweep.run_id(size, s) for size, s in runs]
    assert len(ids) == len(set(ids)) == 55
    assert [size for size, _ in runs].count("0.5B") == 28
    assert all(s.dtype == "fp16" for _, s in runs)
    assert "1.5B-z2-oo-lora16-s1024-mb1-sdpa-g2" not in ids  # LoRA never offloads
    assert "1.5B-z0-lora16-s1024-mb1-sdpa-g1" in ids


def test_host_heavy_runs_come_last():
    heavy = [
        size == "1.5B" and (s.offload_optimizer or s.offload_param)
        for size, s in sweep.grid()
    ]
    n = sum(heavy)
    assert n == 15
    assert heavy == [False] * (55 - n) + [True] * n


def test_command_passes_every_setting():
    size, s = next(
        (size, s)
        for size, s in sweep.grid()
        if s.offload_param and s.zero_stage == 3 and size == "1.5B"
    )
    cmd = sweep.command(size, s, "/out", 29600)
    assert cmd[:5] == ["deepspeed", "--num_gpus", "2", "--master_port", "29600"]
    assert "--offload-optimizer" in cmd and "--offload-param" in cmd
    assert cmd[cmd.index("--revision") + 1] == sweep.MODELS["1.5B"][1]
    assert cmd[-2:] == ["--out-dir", "/out"]


def write_ranks(folder, *records):
    for i, record in enumerate(records):
        (folder / f"rank{i}.json").write_text(json.dumps({"rank": i, **record}))
    (folder / "log.txt").write_text("Traceback ...\nCUDA out of memory")


OK = {"status": "ok", "applied": [True] * 6 + [False] * 2, "env": {"torch": "2.14.1"}}


@pytest.mark.parametrize(
    ("records", "code", "status"),
    [
        ([OK, OK], 0, "ok"),
        ([OK, {"status": "oom"}], 1, "oom"),
        ([{"status": "oom"}], None, "oom"),  # one rank OOMed, the other hung
        ([OK], None, "timeout"),
        ([OK], 0, "error"),  # a rank wrote nothing
        ([], -9, "error"),  # e.g. the host's OOM killer
    ],
)
def test_merge_status(tmp_path, records, code, status):
    size, setup = sweep.grid()[3]  # a 2-GPU run
    write_ranks(tmp_path, *records)
    line = sweep.merge(size, setup, str(tmp_path), code, 12.3)
    assert line["status"] == status
    assert line["run"] == sweep.run_id(size, setup)
    assert line["applied_steps"] == (6 if status == "ok" else None)
    assert ("log_tail" in line) == (status != "ok")
    assert all("env" not in r for r in line["ranks"])


def test_only_runs_the_named_runs(monkeypatch, capsys):
    smoke = ["0.5B-z3-oo-op-s1024-mb1-sdpa-g2", "0.5B-z2-s1024-mb1-sdpa-g2"]
    argv = ["sweep.py", "--results", "/nonexistent", "--work-dir", "/w", "--dry-run"]
    monkeypatch.setattr(sys, "argv", [*argv, "--only", *smoke])
    sweep.main()
    lines = capsys.readouterr().out.splitlines()
    # In grid order, whatever order they were named in.
    assert [line.rsplit("/", 1)[1] for line in lines] == smoke[::-1]
    monkeypatch.setattr(sys, "argv", [*argv, "--only", "0.5B-z9-s1024-mb1-sdpa-g2"])
    with pytest.raises(SystemExit):
        sweep.main()


def test_notebook_smoke_runs_are_in_the_grid():
    nb = json.loads(
        (Path(__file__).parents[1] / "calibrate/calibration.ipynb").read_text()
    )
    source = "".join(line for c in nb["cells"] for line in c["source"])
    smoke = re.search(r'SMOKE_RUNS = "([^"]+)"', source).group(1).split()
    assert len(smoke) == 2
    assert set(smoke) <= {sweep.run_id(*r) for r in sweep.grid()}
    # The smoke push flips exactly this line.
    assert source.splitlines().count("SMOKE = False") == 1


def test_recommend_grid_ids_and_commands():
    runs = sweep.recommend_grid()
    ids = [sweep.run_id(size, s) for size, s in runs]
    assert len(set(ids)) == len(ids) == 5
    assert "0.5B-z1-s2048-mb1-sdpa-ga4-g2" in ids
    size, s = runs[2]
    cmd = sweep.command(size, s, "/out", 29600)
    assert cmd[cmd.index("--grad-accum") + 1] == "4"
    # Existing calibration ids don't change.
    assert not any("ga" in sweep.run_id(*r) for r in sweep.grid())


@pytest.mark.network
def test_recommend_grid_matches_what_recommend_picks(monkeypatch):
    from fitcheck import estimate, hub, recommend
    from fitcheck.model import from_config

    # The CLI's numbers: the fitted CUDA context, not conftest's bare theory.
    context = estimate.CALIBRATION["cuda_context_bytes"]
    monkeypatch.setattr(estimate, "CUDA_CONTEXT_BYTES", context)
    runs = sweep.recommend_grid()
    for i in (0, 2, 4):  # picks; the run before each pick is the rejected one
        size, s = runs[i]
        name, revision = sweep.MODELS[size]
        model = from_config(name, *hub.load(name, revision=revision))
        opts = recommend.options(model, s, sweep.T4_BYTES)
        pick = recommend.best(opts)
        if s.grad_accum > 1:
            # Picked when measured; after it measured 1 GiB over, unknown.
            assert pick is None and opts[1].setup == s and opts[1].fits == "unknown"
            continue
        assert pick.setup == s
        if i < 4:
            before = opts[opts.index(pick) - 1]
            assert before.setup == runs[i + 1][1]
            assert (before.fits, before.reason) == ("no", "GPU memory")
