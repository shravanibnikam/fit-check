import json

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
