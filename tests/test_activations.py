import pytest

from fitcheck.activations import FAMILIES
from fitcheck.estimate import Setup, estimate
from fitcheck.model import Model

# Fields of the real config.json files that the formulas read.
QWEN25_05B = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "intermediate_size": 4864,
    "vocab_size": 151936,
}
QWEN35_08B = {
    "model_type": "qwen3_5_text",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "intermediate_size": 3584,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "vocab_size": 248320,
    "rope_parameters": {"partial_rotary_factor": 0.25},
}
QWEN35_08B["layer_types"] = QWEN35_08B["layer_types"] * 6
GEMMA4_E2B = {
    "model_type": "gemma4_text",
    "hidden_size": 1536,
    "num_hidden_layers": 35,
    "layer_types": (["sliding_attention"] * 4 + ["full_attention"]) * 7,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
    "intermediate_size": 6144,
    "sliding_window": 512,
    "hidden_size_per_layer_input": 256,
    "num_kv_shared_layers": 20,
    "use_double_wide_mlp": True,
    "final_logit_softcapping": 30.0,
    "vocab_size": 262144,
}


def make(cfg, family, params=0):
    h, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    kv = cfg.get("num_key_value_heads", heads)
    d = cfg.get("head_dim", h // heads)
    n = cfg["num_hidden_layers"]
    return Model("m", params, h, n, heads, kv, d, family=family, config=cfg)


QWEN = make(QWEN25_05B, "llama", 494_032_768)


def test_worked_example_in_docs():
    # docs/memory-model.md: Qwen2.5-0.5B, bf16, one sequence of 2048, SDPA.
    e = estimate(QWEN, Setup())
    # 24 layers also keep SDPA's 16-byte philox seed and offset on CUDA.
    assert e.activations == 2_822_242_304 + 24 * 16
    assert e.logits == 4_371_013_636


def test_checkpointing_keeps_layer_inputs():
    e = estimate(QWEN, Setup(checkpointing=True))
    # 24 layer inputs of 896 bf16 values per token, plus the RoPE cos/sin table.
    assert e.activations == 24 * 2048 * 2 * 896 + 2 * 2 * 2048 * 64
    # The output head is not checkpointed.
    assert e.logits == 4_371_013_636


def test_checkpointing_peak_can_be_a_recomputed_layer():
    # With a tiny vocab the head is smaller than one layer, so the peak is the
    # recompute of the largest layer.
    m = make({**QWEN25_05B, "vocab_size": 10}, "llama")
    e = estimate(m, Setup(checkpointing=True))
    one_layer = FAMILIES["llama"](m.config, Setup(), 2).layers[0]
    assert e.activations > one_layer
    assert e.logits == 0


def test_eager_attention_grows_with_square_of_sequence():
    def extra(s):
        eager = estimate(QWEN, Setup(seq_len=s, attn="eager")).activations
        return eager - estimate(QWEN, Setup(seq_len=s)).activations

    # Mostly the fp32 softmax and its bf16 copy, heads x s per token.
    assert extra(4096) / extra(2048) > 3.9


def test_lora_keeps_less_than_full_fine_tuning():
    full = estimate(QWEN, Setup())
    lora = estimate(QWEN, Setup(lora_rank=16))
    assert lora.activations < full.activations
    assert lora.logits < full.logits


def test_peak_is_backward_or_optimizer_step():
    # One token: activations are tiny, so the fp32 gradient copy at the
    # optimizer step sets the peak.
    tiny = estimate(QWEN, Setup(seq_len=1))
    assert tiny.step == 4 * QWEN.num_params
    assert tiny.peak_extra == tiny.step
    long = estimate(QWEN, Setup())
    assert long.peak_extra == long.activations + long.logits
    assert estimate(QWEN, Setup(num_gpus=2, zero_stage=2)).step == 2 * QWEN.num_params
    assert estimate(QWEN, Setup(zero_stage=2, offload_optimizer=True)).step == 0


def test_unsupported_architecture_still_gets_model_states():
    m = Model("gpt2", 124_000_000, 768, 12, 12, 12, 64)
    e = estimate(m, Setup())
    assert e.activations is None and e.logits is None
    assert e.gpu_total == e.weights + e.grads + e.optimizer + e.step


def test_qwen35_linear_attention_layers_dominate():
    layers = FAMILIES["qwen3_5"](QWEN35_08B, Setup(), 2).layers
    linear, full = layers[0], layers[3]
    # The PyTorch Gated DeltaNet fallback keeps fp32 chunk matrices and
    # recurrent states, over twice what a full-attention layer keeps.
    assert linear > 2 * full


def test_qwen35_lora_saves_nothing_before_first_adapter():
    layers = FAMILIES["qwen3_5"](QWEN35_08B, Setup(lora_rank=8), 2).layers
    # Layers 0-2 are frozen DeltaNet layers fed by frozen embeddings.
    assert layers[:3] == [0, 0, 0]
    assert layers[3] > 0


def test_gemma4_sliding_mask_starts_at_the_window():
    def layer0(s):
        return FAMILIES["gemma4"](GEMMA4_E2B, Setup(seq_len=s), 2).layers[0]

    per_token = layer0(511) - layer0(510)
    # From s = sliding_window on, SDPA keeps a float copy of the s x s mask.
    assert layer0(512) - layer0(511) - per_token == 512 * 512 * 2


def test_mistral_sliding_mask_starts_at_the_window():
    mistral = {**QWEN25_05B, "model_type": "mistral", "sliding_window": 1024}

    def layer0(cfg, s):
        return FAMILIES["llama"](cfg, Setup(seq_len=s), 2).layers[0]

    # Below the window Mistral runs exactly like Llama.
    assert layer0(mistral, 1023) == layer0(QWEN25_05B, 1023)
    # From the window on: the s x s float mask, and K/V copied from 2 to 14 heads.
    s, q, kv = 1024, 14 * 64, 2 * 64
    extra = s * s * 2 + 2 * 2 * (q - kv) * s
    assert layer0(mistral, s) - layer0(QWEN25_05B, s) == extra
    # Without the key, MistralConfig's default window of 4096 applies.
    no_key = {**mistral}
    del no_key["sliding_window"]
    assert layer0(no_key, 4096) > layer0(QWEN25_05B, 4096)


def test_qwen2_slides_only_with_use_sliding_window():
    def layers(**cfg):
        return FAMILIES["llama"]({**QWEN25_05B, **cfg}, Setup(seq_len=4096), 2).layers

    dense = layers()
    assert layers(sliding_window=4096) == dense  # Qwen2.5 configs ship this
    # Layers from max_window_layers on slide.
    sliding = layers(use_sliding_window=True, sliding_window=4096, max_window_layers=20)
    assert sliding[:20] == dense[:20]
    assert all(a > b for a, b in zip(sliding[20:], dense[20:]))


def test_gemma4_shared_kv_layers_have_no_kv_terms():
    parts = FAMILIES["gemma4"]({**GEMMA4_E2B, "use_double_wide_mlp": False}, Setup(), 2)
    # Layer 15 is the first that reuses K/V; both are sliding layers.
    assert parts.layers[15] < parts.layers[0]


@pytest.mark.parametrize(
    "cfg,family", [(GEMMA4_E2B, "gemma4"), (QWEN35_08B, "qwen3_5")]
)
def test_families_cover_every_layer(cfg, family):
    parts = FAMILIES[family](cfg, Setup(), 2)
    assert len(parts.layers) == cfg["num_hidden_layers"]
    assert all(n > 0 for n in parts.layers)
