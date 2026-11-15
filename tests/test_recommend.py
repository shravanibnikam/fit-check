import pytest

from fitcheck import recommend
from fitcheck.estimate import Setup
from fitcheck.model import from_config
from fitcheck.recommend import ZERO_CHOICES, best, cost, ds_config, options

GIB = 1024**3
CONFIG = {
    "model_type": "llama",
    "hidden_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 512,
    "vocab_size": 1000,
}


@pytest.fixture(autouse=True)
def no_margin(monkeypatch):
    monkeypatch.setattr(recommend, "MARGIN", 0.0)


def tensors(**shapes):
    return {name: {"dtype": "BF16", "shape": shape} for name, shape in shapes.items()}


@pytest.fixture
def model():
    shapes = {f"w{i}": [1024, 1024] for i in range(8)}
    return from_config("tiny", CONFIG, tensors(embed=[1000, 256], **shapes))


def needed(o):
    return o.estimate.allocated + o.estimate.overhead


def test_every_setup_cheapest_first(model):
    opts = options(model, Setup(num_gpus=2), gpu_mem=100 * GIB)
    assert len(opts) == 2 * len(ZERO_CHOICES)
    assert [cost(o.setup) for o in opts] == sorted(cost(o.setup) for o in opts)
    first, last = opts[0].setup, opts[-1].setup
    assert (first.zero_stage, first.offload_optimizer, first.checkpointing) == (
        0,
        False,
        False,
    )
    assert last.offload_param and last.checkpointing
    assert best(opts) == opts[0]


def test_best_is_the_cheapest_that_fits(model):
    base = Setup(num_gpus=2, seq_len=512)
    opts = options(model, base, gpu_mem=100 * GIB)
    # Just below the cheapest setup's need, something further down the order.
    tight = options(model, base, gpu_mem=needed(opts[0]) - 1)
    pick = best(tight)
    assert pick is not None and pick.setup != opts[0].setup
    assert all(o.fits == "no" for o in tight[: tight.index(pick)])
    assert needed(pick) <= needed(opts[0]) - 1


def test_nothing_fits(model):
    opts = options(model, Setup(), gpu_mem=1)
    assert best(opts) is None
    assert {o.reason for o in opts} == {"GPU memory"}


def test_host_ram_rules_out_offload(model):
    opts = options(model, Setup(num_gpus=2), gpu_mem=100 * GIB, host_mem=0)
    for o in opts:
        offload = o.setup.offload_optimizer or o.setup.offload_param
        assert (o.fits, o.reason) == (("no", "host RAM") if offload else ("yes", ""))


def test_margin_needs_headroom(model, monkeypatch):
    first = options(model, Setup(), gpu_mem=100 * GIB)[0]
    assert options(model, Setup(), gpu_mem=needed(first))[0].fits == "yes"
    monkeypatch.setattr(recommend, "MARGIN", 0.05)
    assert options(model, Setup(), gpu_mem=needed(first))[0].fits == "no"


def test_unmodeled_model_is_never_recommended():
    config = {k: v for k, v in CONFIG.items() if k != "model_type"}
    m = from_config("plain", config, tensors(w=[8, 8]))
    opts = options(m, Setup(), gpu_mem=100 * GIB)
    assert {o.fits for o in opts} == {"unknown"}
    assert best(opts) is None
    # "no" still holds when the counted part alone is too big.
    assert {o.fits for o in options(m, Setup(), gpu_mem=1)} == {"no"}


def test_config_fields():
    c = ds_config(Setup(zero_stage=0, dtype="fp32", micro_batch=2, grad_accum=4))
    assert c["zero_optimization"] == {"stage": 0}
    assert "fp32" not in c and "bf16" not in c and "fp16" not in c
    assert c["train_micro_batch_size_per_gpu"] == 2
    assert c["gradient_accumulation_steps"] == 4
    z = ds_config(Setup(zero_stage=3, offload_optimizer=True, dtype="fp16"))
    assert z["fp16"] == {"enabled": True}
    assert z["data_types"] == {"grad_accum_dtype": "fp16"}
    assert z["zero_optimization"]["offload_optimizer"] == {
        "device": "cpu",
        "pin_memory": True,
    }
    assert z["zero_optimization"]["overlap_comm"] is True
    assert "offload_param" not in z["zero_optimization"]


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize(("stage", "opt", "param"), ZERO_CHOICES)
def test_deepspeed_accepts_every_config(stage, opt, param, dtype):
    """Each config parses with DeepSpeed's own parser, to the setup we meant."""
    pytest.importorskip("deepspeed")
    from deepspeed.runtime.config import DeepSpeedConfig

    setup = Setup(
        zero_stage=stage,
        offload_optimizer=opt,
        offload_param=param,
        dtype=dtype,
        micro_batch=2,
        grad_accum=3,
    )
    parsed = DeepSpeedConfig(ds_config(setup))
    zero = parsed.zero_config
    assert parsed.zero_optimization_stage == stage
    assert parsed.train_micro_batch_size_per_gpu == 2
    assert parsed.gradient_accumulation_steps == 3
    assert parsed.float16_config.enabled == (dtype == "fp16")
    assert parsed.bfloat16_config.enabled == (dtype == "bf16")
    assert (zero.offload_optimizer is not None) == opt
    assert (zero.offload_param is not None) == param
    if opt:
        assert zero.offload_optimizer.pin_memory
    assert parsed.optimizer_name == "adamw"
    assert parsed.grad_accum_dtype == dtype
    # What the estimate assumes, as DeepSpeed reads it.
    assert zero.contiguous_gradients
    assert zero.overlap_comm == (stage == 3)
    assert zero.reduce_bucket_size == recommend.REDUCE_BUCKET
    assert zero.max_reuse_distance == recommend.MAX_REUSE_DISTANCE
    assert zero.sub_group_size == recommend.ZERO3_SUB_GROUP


def test_gradient_accumulation_only_at_zero1(model):
    # Only ZeRO-1 with accumulation is measured; the other stages are unknown.
    opts = options(model, Setup(num_gpus=2, grad_accum=4), gpu_mem=100 * GIB)
    for o in opts:
        assert o.fits == ("yes" if o.setup.zero_stage == 1 else "unknown")
    assert best(opts).setup.zero_stage == 1
