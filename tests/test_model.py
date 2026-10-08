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
