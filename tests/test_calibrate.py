import json

import pytest

from fitcheck import hub
from fitcheck.calibrate import main
from fitcheck.estimate import Setup, estimate
from fitcheck.model import from_config

GIB = 1024**3


@pytest.fixture
def models(tmp_path, make_safetensors):
    """Two small Llama-style models in local folders, as hub.load reads them."""
    paths = []
    for name, hidden in (("small", 256), ("large", 512)):
        folder = tmp_path / name
        folder.mkdir()
        config = {
            "model_type": "llama",
            "hidden_size": hidden,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4 * hidden,
            "vocab_size": 32000,
        }
        (folder / "config.json").write_text(json.dumps(config))
        shapes = {"embed": [32000, hidden], "layers": [4 * 12 * hidden, hidden]}
        (folder / "model.safetensors").write_bytes(make_safetensors(shapes))
        paths.append(str(folder))
    return paths


def bare_prediction(model, setup):
    config, tensors = hub.load(model)
    return estimate(from_config(model, config, tensors), setup).gpu_total


def run(model, factor, *, seq_len=1024, applied=8, status="ok", torch="2.14.1"):
    """A results.jsonl line whose ranks reserved factor x the bare prediction."""
    setup = Setup(num_gpus=2, zero_stage=2, dtype="fp16", seq_len=seq_len)
    predicted = bare_prediction(model, setup)
    rank = {
        "status": status,
        "total_memory": 15 * GIB,
        "max_allocated": predicted,
        "max_reserved": round(factor * predicted),
        "reserved_end": round(factor * predicted),
        "device_used_end": round(factor * predicted) + 400 * 2**20,
    }
    return {
        "run": f"{model}-s{seq_len}-a{applied}",
        "status": status,
        "model": model,
        "revision": "main",
        "setup": setup.__dict__,
        "applied_steps": applied if status == "ok" else None,
        "env": {"torch": torch, "gpu": "Tesla T4", "date": "2026-11-02T10:00:00+00:00"},
        "ranks": [{**rank, "rank": 0}, {**rank, "rank": 1}],
    }


def write(tmp_path, runs, check=None):
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in runs))
    if check is not None:
        cases = [{"config": "llama", "lora_rank": None, "attn": "sdpa", "ok": check}]
        (tmp_path / "saved_tensors_cuda.json").write_text(
            json.dumps(
                {
                    "device": "cuda",
                    "torch": "2.14.1",
                    "all_match": check,
                    "cases": cases,
                }
            )
        )
    return str(results)


def fit(tmp_path, results, *extra):
    out = tmp_path / "calibration.json"
    main([results, "--out", str(out), *extra])
    return json.loads(out.read_text())


def test_fits_constants_with_held_out_error(tmp_path, models):
    small, large = models
    runs = [run(small, 1.2, seq_len=s) for s in (512, 1024)]
    runs += [run(large, 1.3, seq_len=s) for s in (512, 1024)]
    # Too few applied steps: dropped from the fit, however wrong it looks.
    runs.append(run(large, 5.0, seq_len=2048, applied=3))
    cal = fit(tmp_path, write(tmp_path, runs, check=True))

    assert 1.2 < cal["reserved_over_allocated"] < 1.3
    assert cal["cuda_context_bytes"] == 400 * 2**20
    # Fitted on the other model only: 1.3 / 1.2 - 1 and 1.2 / 1.3 - 1.
    assert cal["held_out"][small]["median_abs_error"] == pytest.approx(0.0833, abs=1e-4)
    assert cal["held_out"][large]["max_abs_error"] == pytest.approx(0.0769, abs=1e-4)
    assert cal["held_out"][small]["within_10_percent"] == 2
    prov = cal["provenance"]
    assert (prov["runs"], prov["valid_runs"]) == (5, 4)
    assert prov["torch"] == ["2.14.1"] and prov["gpu"] == ["Tesla T4"]
    assert len(prov["results_sha256"]) == 64
    assert prov["cuda_check"]["all_match"] is True


@pytest.mark.parametrize(
    ("check", "torch", "message"),
    [
        (None, "2.14.1", "no CUDA saved-tensor check"),
        (False, "2.14.1", "don't match: llama full sdpa"),
        (True, "2.11.0+cu128", "the runs on 2.11.0"),
    ],
)
def test_refuses_to_fit_until_the_cuda_check_passes(
    tmp_path, models, check, torch, message
):
    results = write(tmp_path, [run(m, 1.2, torch=torch) for m in models], check)
    with pytest.raises(SystemExit, match=message):
        fit(tmp_path, results)
    assert not (tmp_path / "calibration.json").exists()
    # The override fits anyway and says why.
    cal = fit(tmp_path, results, "--ignore-cuda-check")
    assert message.split(":")[0] in cal["provenance"]["cuda_check"]["overridden"]


def test_reports_verdicts_on_out_of_memory_runs(tmp_path, models, capsys):
    small, large = models
    runs = [run(small, 1.2), run(large, 1.2)]
    # Measured OOM although the estimate, far under 15 GiB, says it fits.
    runs.append(run(large, 1.2, seq_len=2048, status="oom"))
    fit(tmp_path, write(tmp_path, runs, check=True))
    out = capsys.readouterr().out
    assert "Fits verdict right on 2 of 3 runs." in out
    assert "wrong: " in out and "(oom, predicted fit)" in out
