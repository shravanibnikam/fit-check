import pytest

from fitcheck.model import from_config

# The fields of Qwen/Qwen2.5-0.5B's config.json that we read.
QWEN_05B = {
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
}
TENSORS = {"w": {"dtype": "BF16", "shape": [3, 7]}}


def test_reads_fields_and_derives_head_dim():
    m = from_config("qwen", QWEN_05B, TENSORS)
    assert (m.hidden_size, m.num_layers, m.num_heads, m.num_kv_heads) == (
        896,
        24,
        14,
        2,
    )
    assert m.head_dim == 64
    assert m.num_params == 21


def test_explicit_head_dim_wins():
    cfg = {**QWEN_05B, "hidden_size": 2304, "num_attention_heads": 8, "head_dim": 256}
    assert from_config("gemma", cfg, TENSORS).head_dim == 256


def test_kv_heads_default_to_query_heads():
    cfg = {k: v for k, v in QWEN_05B.items() if k != "num_key_value_heads"}
    assert from_config("mha", cfg, TENSORS).num_kv_heads == 14


def test_reads_nested_text_config():
    cfg = {"model_type": "gemma3", "text_config": QWEN_05B, "vision_config": {}}
    assert from_config("vlm", cfg, TENSORS).num_layers == 24


def test_missing_field_names_the_key():
    with pytest.raises(ValueError, match="'num_hidden_layers'"):
        from_config(
            "gpt2", {"n_layer": 12, "hidden_size": 1, "num_attention_heads": 1}, {}
        )


def test_lora_shapes_and_weights_transformers_drops():
    def w(*shape):
        return {"dtype": "BF16", "shape": list(shape)}

    cfg = {**QWEN_05B, "model_type": "gemma4_text", "num_kv_shared_layers": 1}
    cfg["num_hidden_layers"] = 2
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": w(128, 64),
        "model.layers.0.self_attn.k_proj.weight": w(32, 64),
        "model.layers.0.self_attn.q_proj.bias": w(128),
        # Layer 1 reuses layer 0's K/V; transformers drops its K/V weights.
        "model.layers.1.self_attn.k_proj.weight": w(32, 64),
        "model.layers.1.self_attn.o_proj.weight": w(64, 128),
        # Not a LoRA target: PEFT matches whole module names.
        "model.audio_tower.layers.0.relative_k_proj.weight": w(8, 8),
        # Qwen3.5's multi-token-prediction head is never loaded.
        "mtp.layers.0.mlp.up_proj.weight": w(4, 4),
    }
    m = from_config("m", cfg, tensors)
    assert m.family == "gemma4"
    assert sorted(m.lora_shapes) == [(64, 32), (64, 128), (128, 64)]
    assert m.num_params == 128 * 64 + 32 * 64 + 128 + 64 * 128 + 8 * 8


def test_unknown_model_type_has_no_family():
    assert (
        from_config("x", {**QWEN_05B, "model_type": "gpt_neox"}, TENSORS).family is None
    )
