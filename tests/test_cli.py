import json

import pytest

from fitcheck.cli import main

CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
}


@pytest.fixture
def folder(tmp_path, make_safetensors):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    # 1024 * 1024 params, so full fine-tuning in bf16 is exactly 16 MiB.
    (tmp_path / "model.safetensors").write_bytes(make_safetensors({"w": [1024, 1024]}))
    return str(tmp_path)


def row(out, label):
    """The value column of a breakdown row."""
    line = next(x for x in out.splitlines() if x.strip().startswith(label))
    return line.split()[-1]


def test_prints_breakdown_and_verdict(folder, capsys):
    main([folder, "--gpu-mem", "1"])
    out = capsys.readouterr().out
    assert "1.0M params" in out
    assert "4 heads (2 KV) of 16" in out
    assert "micro-batch 1 x 2048 tokens, sdpa attention" in out
    # No model_type, so no activation formula: model states only.
    assert row(out, "activations") == "n/a"
    assert "model states only" in out
    assert "Fits in 1 GiB: yes" in out


def test_counts_activations_for_known_families(tmp_path, make_safetensors, capsys):
    config = {
        **CONFIG,
        "model_type": "llama",
        "intermediate_size": 128,
        "vocab_size": 100,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(make_safetensors({"w": [8, 8]}))
    main([str(tmp_path), "--seq-len", "4096", "--micro-batch", "2", "--checkpointing"])
    out = capsys.readouterr().out
    assert "micro-batch 2 x 4096 tokens, sdpa attention, checkpointing" in out
    assert row(out, "activations") != "n/a"


def test_invalid_setup_exits_with_message(folder):
    with pytest.raises(SystemExit, match="parameter offload needs ZeRO stage 3"):
        main([folder, "--zero", "2", "--offload-param"])
