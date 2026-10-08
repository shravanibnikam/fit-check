import pytest

from fitcheck.estimate import Setup, estimate, lora_params
from fitcheck.model import Model

# Llama-2-7B shapes: 32 layers, hidden 4096, 32 heads of 128, no GQA.
LLAMA2_7B = Model("llama2-7b", 6_738_415_616, 4096, 32, 32, 32, 128)
# Qwen2.5-0.5B shapes: 24 layers, hidden 896, 14 query heads, 2 KV heads of 64.
QWEN_05B = Model("qwen2.5-0.5b", 494_032_768, 896, 24, 14, 2, 64)


def model_states(m, **setup):
    e = estimate(m, Setup(**setup))
    return e.weights + e.grads + e.optimizer


@pytest.mark.parametrize(
    ("stage", "paper_gb"),
    [(0, 120), (1, 31.4), (2, 16.6), (3, 1.88)],
)
def test_matches_zero_paper_figure_1(stage, paper_gb):
    # ZeRO paper, Figure 1: 7.5B parameters, 64 GPUs, mixed-precision Adam.
    m = Model("7.5b", 7_500_000_000, 1, 1, 1, 1, 1)
    gb = model_states(m, num_gpus=64, zero_stage=stage) / 1e9
    assert gb == pytest.approx(paper_gb, rel=0.005)


def test_lora_params_without_gqa():
    # 4 projections of 4096x4096, each adding 8 * (4096 + 4096), over 32 layers.
    # PEFT reports half of this (4,194,304) for the common q_proj/v_proj-only setup.
    assert lora_params(LLAMA2_7B, 8) == 8_388_608


def test_lora_params_with_gqa():
    # k_proj and v_proj map 896 -> 2 * 64 = 128.
    per_layer = (896 + 896) + 2 * (896 + 128) + (896 + 896)
    assert lora_params(QWEN_05B, 16) == 16 * per_layer * 24


def test_lora_trains_only_adapters():
    e = estimate(LLAMA2_7B, Setup(lora_rank=8))
    adapters = 8_388_608
    assert e.trainable_params == adapters
    assert e.weights == 2 * (LLAMA2_7B.num_params + adapters)
    assert e.grads == 2 * adapters
    assert e.optimizer == 12 * adapters


def test_dtype_changes_where_bytes_go():
    p = QWEN_05B.num_params
    # Mixed precision: 2 + 2 + 12. fp32 without ZeRO: 4 + 4 + 8 (Adam moments only).
    assert model_states(QWEN_05B, dtype="bf16") == 16 * p
    assert model_states(QWEN_05B, dtype="fp16") == 16 * p
    assert model_states(QWEN_05B, dtype="fp32") == 16 * p
    # ZeRO-1 on 2 GPUs: fp32 now carries a cloned master partition as well.
    assert model_states(QWEN_05B, dtype="bf16", num_gpus=2, zero_stage=1) == 10 * p
    assert model_states(QWEN_05B, dtype="fp32", num_gpus=2, zero_stage=1) == 14 * p


def test_optimizer_offload():
    p = QWEN_05B.num_params
    s1 = estimate(QWEN_05B, Setup(num_gpus=2, zero_stage=1, offload_optimizer=True))
    assert (s1.optimizer, s1.grads, s1.cpu) == (0, 2 * p, 16 * p)
    s2 = estimate(QWEN_05B, Setup(num_gpus=2, zero_stage=2, offload_optimizer=True))
    assert (s2.weights, s2.grads, s2.optimizer) == (2 * p, 0, 0)


def test_param_offload_moves_weights_to_cpu():
    p = QWEN_05B.num_params
    e = estimate(
        QWEN_05B,
        Setup(num_gpus=2, zero_stage=3, offload_optimizer=True, offload_param=True),
    )
    assert e.gpu_total == 0
    assert e.cpu == 16 * p + 2 * p


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ({"offload_param": True, "zero_stage": 2}, "stage 3"),
        ({"offload_optimizer": True}, "stage 1, 2 or 3"),
        ({"dtype": "int8"}, "dtype"),
        ({"zero_stage": 4}, "zero_stage"),
        ({"num_gpus": 0}, "num_gpus"),
        ({"lora_rank": 0}, "lora_rank"),
    ],
)
def test_rejects_invalid_setups(setup, message):
    with pytest.raises(ValueError, match=message):
        Setup(**setup)
