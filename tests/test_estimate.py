import pytest

from fitcheck.estimate import (
    REDUCE_BUCKET,
    Setup,
    _gathered_at_backward,
    estimate,
    lora_params,
)
from fitcheck.model import Model


def attn_shapes(layers, h, q, kv):
    """(in, out) of q/k/v/o in every layer, as from_config reads them."""
    return ((h, q), (h, kv), (h, kv), (q, h)) * layers


# Llama-2-7B shapes: 32 layers, hidden 4096, 32 heads of 128, no GQA.
LLAMA2_7B = Model(
    "llama2-7b",
    6_738_415_616,
    4096,
    32,
    32,
    32,
    128,
    lora_shapes=attn_shapes(32, 4096, 4096, 4096),
    largest_param=32000 * 4096,
    embedding_params=32000 * 4096,
)
# Qwen2.5-0.5B shapes: 24 layers, hidden 896, 14 query heads, 2 KV heads of 64.
QWEN_05B = Model(
    "qwen2.5-0.5b",
    494_032_768,
    896,
    24,
    14,
    2,
    64,
    lora_shapes=attn_shapes(24, 896, 896, 128),
    largest_param=151936 * 896,
    embedding_params=151936 * 896,
    tied_embeddings=True,
)


def model_states(m, **setup):
    e = estimate(m, Setup(**setup))
    return e.weights + e.grads + e.optimizer


@pytest.mark.parametrize(
    ("stage", "paper_gb"),
    [(0, 120), (2, 16.6), (3, 1.88)],
)
def test_matches_zero_paper_figure_1(stage, paper_gb):
    # ZeRO paper, Figure 1: 7.5B parameters, 64 GPUs, mixed-precision Adam.
    m = Model("7.5b", 7_500_000_000, 1, 1, 1, 1, 1)
    gb = model_states(m, num_gpus=64, zero_stage=stage) / 1e9
    assert gb == pytest.approx(paper_gb, rel=0.005)


def test_zero1_holds_only_the_gradient_partition_at_the_step():
    # The paper's ZeRO-1 keeps full gradients (31.4 GB in Figure 1). DeepSpeed
    # with gradient accumulation 1 reduces them into the bucket at the end of
    # backward, and ZeRO-1 measured the same as ZeRO-2 on the T4.
    m = Model("7.5b", 7_500_000_000, 1, 1, 1, 1, 1)
    assert model_states(m, num_gpus=64, zero_stage=1) == model_states(
        m, num_gpus=64, zero_stage=2
    )


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
    assert model_states(QWEN_05B, dtype="bf16", num_gpus=2, zero_stage=1) == 9 * p
    assert model_states(QWEN_05B, dtype="fp32", num_gpus=2, zero_stage=1) == 12 * p


def test_optimizer_offload():
    p = QWEN_05B.num_params
    s1 = estimate(QWEN_05B, Setup(num_gpus=2, zero_stage=1, offload_optimizer=True))
    assert (s1.optimizer, s1.grads, s1.cpu) == (0, 0, 16 * p)
    s2 = estimate(QWEN_05B, Setup(num_gpus=2, zero_stage=2, offload_optimizer=True))
    assert (s2.weights, s2.grads, s2.optimizer) == (2 * p, 0, 0)
    # The GPU staging buffer the size of the largest trainable tensor.
    assert s2.buffers == 2 * QWEN_05B.largest_param


def test_param_offload_moves_weights_to_cpu():
    p = QWEN_05B.num_params
    e = estimate(
        QWEN_05B,
        Setup(num_gpus=2, zero_stage=3, offload_optimizer=True, offload_param=True),
    )
    assert (e.weights, e.grads, e.optimizer) == (0, 0, 0)
    assert e.cpu == 16 * p + 2 * p
    # Still on the GPU: the reduce bucket and the parameters gathered for use.
    assert e.buffers == 2 * REDUCE_BUCKET
    assert e.gathered > 0


def test_zero3_keeps_parameters_gathered_within_the_reuse_distance():
    # Qwen2.5-0.5B: forward plus backward traverse under 1e9 elements, so every
    # parameter is still gathered when backward starts.
    total = QWEN_05B.num_params
    assert _gathered_at_backward(QWEN_05B, total) == total
    # A 7B model: only the last layers and the head come back within 1e9.
    kept = _gathered_at_backward(LLAMA2_7B, LLAMA2_7B.num_params)
    assert 0 < kept < LLAMA2_7B.num_params / 4


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ({"offload_param": True, "zero_stage": 2}, "stage 3"),
        ({"offload_optimizer": True}, "stage 1, 2 or 3"),
        ({"dtype": "int8"}, "dtype"),
        ({"zero_stage": 4}, "zero_stage"),
        ({"num_gpus": 0}, "num_gpus"),
        ({"lora_rank": 0}, "lora_rank"),
        ({"grad_accum": 0}, "grad_accum"),
    ],
)
def test_rejects_invalid_setups(setup, message):
    with pytest.raises(ValueError, match=message):
        Setup(**setup)


def held(zero_stage, grad_accum, offload_optimizer=False, num_gpus=2):
    setup = Setup(
        num_gpus=num_gpus,
        zero_stage=zero_stage,
        offload_optimizer=offload_optimizer,
        grad_accum=grad_accum,
    )
    e = estimate(QWEN_05B, setup)
    # What backward holds beyond the other terms: the gradients.
    held = e.backward - e.weights - e.optimizer - e.buffers - e.gathered
    return held - (e.activations or 0) - (e.logits or 0), e


def test_grad_accum_one_holds_no_gradients_at_backward_start():
    assert [held(z, 1)[0] for z in (0, 1, 2)] == [0, 0, 0]


def test_grad_accum_holds_earlier_gradients_by_stage():
    p = QWEN_05B.num_params
    assert held(0, 4)[0] == 2 * p  # full, in fp16/bf16
    assert held(1, 4)[0] == 2 * p  # ZeRO-1 reduces only at the boundary
    assert held(2, 4)[0] == 2 * p // 2  # ZeRO-2 keeps its partition
    assert held(2, 4)[0] == held(2, 4)[1].grads


def test_grad_accum_with_optimizer_offload():
    p = QWEN_05B.num_params
    h1, e1 = held(1, 4, offload_optimizer=True)
    assert h1 == 2 * p  # ZeRO-1 offload still holds full gradients on the GPU
    assert e1.cpu == 16 * p + 2 * p
    h2, e2 = held(2, 4, offload_optimizer=True)
    assert h2 == 0  # ZeRO-2 offload accumulates on the CPU
    assert e2.cpu == 16 * p + 2 * p


def test_grad_accum_does_not_change_zero3():
    assert held(3, 4)[0] == held(3, 1)[0]
