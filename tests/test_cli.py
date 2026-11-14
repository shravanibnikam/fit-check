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
    # No model_type, so no activation formula: model states only, and the
    # 16 MiB that fit are not enough to say yes.
    assert row(out, "activations") == "n/a"
    assert "Not modeled: no activation formula for this model yet" in out
    assert "Fits in 1 GiB: unknown" in out


def test_says_no_when_the_counted_part_already_exceeds(folder, capsys):
    main([folder, "--gpu-mem", "0.01"])
    assert (
        "Fits in 0.01 GiB: no, even without what is not modeled"
        in capsys.readouterr().out
    )


def test_warns_about_layers_lora_cannot_wrap(tmp_path, make_safetensors, capsys):
    config = {
        **CONFIG,
        "num_hidden_layers": 6,
        "model_type": "llama",
        "intermediate_size": 128,
        "vocab_size": 100,
    }
    shapes = {
        f"model.layers.{i}.self_attn.{t}.weight": [8, 8]
        for i in range(6)
        for t in ("q_proj", "k_proj", "v_proj", "o_proj")
        # Layers 2-3 and 5 have no K/V projections.
        if t in ("q_proj", "o_proj") or i in (0, 1, 4)
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(make_safetensors(shapes))
    main([str(tmp_path), "--lora-rank", "8", "--gpu-mem", "1"])
    out = capsys.readouterr().out
    assert "LoRA: 3 of 6 layers have no k_proj, v_proj (layers 2-3, 5)" in out
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
    assert "checkpointing, gradient accumulation 1" in out
    assert row(out, "activations") != "n/a"


def test_invalid_setup_exits_with_message(folder):
    with pytest.raises(SystemExit, match="parameter offload needs ZeRO stage 3"):
        main([folder, "--zero", "2", "--offload-param"])


def test_grad_accum_flag_shows_in_header(folder, capsys):
    main([folder, "--grad-accum", "4"])
    out = capsys.readouterr().out
    assert "gradient accumulation 4" in out


def test_loss_tokens_is_shown_and_shrinks_the_head(folder, capsys):
    main([folder, "--gpu-mem", "1"])
    full = capsys.readouterr().out
    main([folder, "--gpu-mem", "1", "--loss-tokens", "8"])
    out = capsys.readouterr().out
    assert "loss on last 8 tokens" in out
    assert "loss on last" not in full


@pytest.fixture
def llama_folder(tmp_path, make_safetensors):
    config = {
        **CONFIG,
        "model_type": "llama",
        "intermediate_size": 128,
        "vocab_size": 100,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(make_safetensors({"w": [1024, 1024]}))
    return tmp_path


def test_recommend_writes_the_best_config(llama_folder, capsys):
    out_file = llama_folder / "ds.json"
    main(["recommend", str(llama_folder), "--gpu-mem", "1", "--out", str(out_file)])
    out = capsys.readouterr().out
    # Everything fits in 1 GiB, so the cheapest setup wins.
    assert "* ZeRO-0 " in out
    assert "Best: ZeRO-0. Wrote" in out
    config = json.loads(out_file.read_text())
    assert config["zero_optimization"] == {"stage": 0}
    assert config["bf16"] == {"enabled": True}


def test_recommend_falls_back_to_offload(llama_folder, make_safetensors, capsys):
    # 67M params: 1.5 GiB without ZeRO, about 0.75 GiB per GPU with ZeRO-1 or 2
    # on 2 GPUs, and ZeRO-3's reduce bucket alone is 0.93 GiB. Only optimizer
    # offload fits in 0.5 GiB.
    (llama_folder / "model.safetensors").write_bytes(
        make_safetensors({"w": [8192, 8192]})
    )
    out_file = llama_folder / "ds.json"
    argv = ["recommend", str(llama_folder), "--gpus", "2", "--gpu-mem", "0.5"]
    main([*argv, "--seq-len", "64", "--out", str(out_file)])
    out = capsys.readouterr().out
    assert "* ZeRO-1, offload optimizer " in out
    assert "yes (GPU; host RAM unchecked)" in out
    assert "GiB of host RAM over all ranks; check that you have about twice" in out
    zero = json.loads(out_file.read_text())["zero_optimization"]
    assert zero["offload_optimizer"]["device"] == "cpu"
    # A limit below the offloaded state rules it out.
    with pytest.raises(SystemExit, match="no setup fits"):
        main([*argv, "--seq-len", "64", "--host-mem", "0.1", "--out", str(out_file)])
    assert "no (host RAM)" in capsys.readouterr().out


def test_recommend_lists_what_the_script_must_set(llama_folder, capsys):
    out_file = llama_folder / "ds.json"
    argv = ["recommend", str(llama_folder), "--gpu-mem", "1", "--out", str(out_file)]
    main([*argv, "--lora-rank", "4", "--attn", "eager", "--loss-tokens", "8"])
    out = capsys.readouterr().out
    assert 'attn_implementation="eager"' in out
    assert "peft LoraConfig(r=4, target_modules=['q_proj'" in out
    assert "logits_to_keep=9" in out


def test_recommend_needs_gpu_mem(folder):
    with pytest.raises(SystemExit):
        main(["recommend", folder])


def test_recommend_writes_nothing_for_unmodeled_models(folder, tmp_path, capsys):
    out_file = tmp_path / "ds.json"
    with pytest.raises(SystemExit, match="can't recommend a setup"):
        main(["recommend", folder, "--gpu-mem", "1", "--out", str(out_file)])
    assert not out_file.exists()
    assert "Not modeled: no activation formula" in capsys.readouterr().out


def test_recommend_says_when_nothing_fits(llama_folder, tmp_path):
    with pytest.raises(SystemExit, match="no setup fits"):
        main(["recommend", str(llama_folder), "--gpu-mem", "0.001"])
