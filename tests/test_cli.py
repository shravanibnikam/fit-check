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


def test_prints_breakdown_and_verdict(folder, capsys):
    main([folder, "--gpu-mem", "1"])
    out = capsys.readouterr().out
    assert "1.0M params" in out
    assert "4 heads (2 KV) of 16" in out
    assert "total" in out and "0.02" in out
    assert "Fits in 1 GiB: yes" in out


def test_invalid_setup_exits_with_message(folder):
    with pytest.raises(SystemExit, match="parameter offload needs ZeRO stage 3"):
        main([folder, "--zero", "2", "--offload-param"])
