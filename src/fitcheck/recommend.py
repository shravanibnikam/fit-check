"""Rank the DeepSpeed setups that fit, and write the best one's ds_config.json.

Every ZeRO stage, offload choice and activation checkpointing setting is
estimated for the same model, task and batch. The setups are ranked by a cost
order that is a rule of thumb, not measured (see "Choosing a setup" in
docs/memory-model.md):

1. Offload first, because it moves the optimizer step to the CPU and every
   step's gradients and weights over PCIe: none, then the optimizer, then the
   optimizer and the parameters.
2. Then checkpointing, which runs each layer's forward again in backward.
3. Then the ZeRO stage. Stages 1 and 2 communicate as much as plain data
   parallelism, stage 3 about 1.5 times as much (ZeRO paper, section 7).
"""

from dataclasses import dataclass, replace

from fitcheck.estimate import (
    CALIBRATION,
    MAX_REUSE_DISTANCE,
    REDUCE_BUCKET,
    ZERO3_SUB_GROUP,
    Estimate,
    Setup,
    estimate,
)
from fitcheck.model import Model

# The worst measured error of the allocated estimate, |predicted / measured - 1|,
# from calibration.json. A setup is recommended only if it fits even when the
# true peak is that much above the estimate: measured <= predicted / (1 - e).
MARGIN = CALIBRATION["allocated_error"]["max_abs_error"] if CALIBRATION else 0.0

# (zero_stage, offload_optimizer, offload_param), the combinations DeepSpeed
# accepts (DeepSpeedZeroConfig in deepspeed/runtime/zero/config.py).
ZERO_CHOICES = [
    (0, False, False),
    (1, False, False),
    (2, False, False),
    (3, False, False),
    (1, True, False),
    (2, True, False),
    (3, True, False),
    (3, True, True),
]


@dataclass(frozen=True)
class Option:
    setup: Setup
    estimate: Estimate
    fits: str  # "yes", "no" or "unknown"
    reason: str = ""  # why not: "GPU memory" or "host RAM"


def cost(setup: Setup) -> tuple:
    """Smaller is cheaper; see the module docstring."""
    offload = setup.offload_optimizer + setup.offload_param
    return (offload, setup.checkpointing, setup.zero_stage)


def options(
    model: Model, base: Setup, gpu_mem: int, host_mem: int | None = None
) -> list[Option]:
    """Every setup for base's model, task and batch, cheapest first.

    gpu_mem is bytes per GPU, host_mem bytes of host RAM for all ranks
    together. base's ZeRO stage, offload and checkpointing are ignored.
    """
    out = []
    for stage, opt, param in ZERO_CHOICES:
        for ckpt in (False, True):
            setup = replace(
                base,
                zero_stage=stage,
                offload_optimizer=opt,
                offload_param=param,
                checkpointing=ckpt,
            )
            est = estimate(model, setup)
            needed = est.allocated / (1 - MARGIN) + est.overhead
            if needed > gpu_mem:
                fits, reason = "no", "GPU memory"
            # est.cpu leaves out the model as loaded and the runtimes, so it
            # can only rule a setup out (docs/memory-model.md, "Calibration").
            elif host_mem is not None and est.cpu > host_mem:
                fits, reason = "no", "host RAM"
            # What is not modeled only adds memory, so "no" holds; "yes" would not.
            elif model.unmodeled or est.activations is None:
                fits, reason = "unknown", ""
            else:
                fits, reason = "yes", ""
            out.append(Option(setup, est, fits, reason))
    return sorted(out, key=lambda o: cost(o.setup))


def best(opts: list[Option]) -> Option | None:
    """The cheapest setup that fits, or None."""
    return next((o for o in opts if o.fits == "yes"), None)


def ds_config(setup: Setup, lr: float = 1e-5) -> dict:
    """A DeepSpeed config for deepspeed.initialize, as FitCheck estimates it.

    The ZeRO settings the estimate depends on are written out at their
    DeepSpeed v0.19.9 defaults, so the estimate stays right if a default
    changes. The training dtype must match the model's (DeepSpeed casts the
    model to it), and gradients accumulate in it (data_types.grad_accum_dtype):
    with fp32 accumulation, bf16 ZeRO-1 switches to BF16_Optimizer
    (_do_optimizer_sanity_check in deepspeed/runtime/engine.py), which FitCheck
    does not model. AdamW from the config is FusedAdam on the GPU, or
    DeepSpeedCPUAdam with optimizer offload, the 16 bytes per parameter the
    estimate counts (get_optimizer_configuration in engine.py).
    """
    zero = {"stage": setup.zero_stage}
    if setup.zero_stage > 0:
        # deepspeed/runtime/zero/config.py. overlap_comm defaults to on only
        # at stage 3 (overlap_comm_valid), and with it off ZeRO-1 reduces only
        # at the accumulation boundary, which the gradient terms assume.
        zero["contiguous_gradients"] = True
        zero["overlap_comm"] = setup.zero_stage == 3
        zero["reduce_bucket_size"] = REDUCE_BUCKET
    if setup.zero_stage == 3:
        zero["sub_group_size"] = ZERO3_SUB_GROUP
        zero["stage3_max_reuse_distance"] = MAX_REUSE_DISTANCE
    # pin_memory stays at its default, off, as in the calibration runs.
    if setup.offload_optimizer:
        zero["offload_optimizer"] = {"device": "cpu"}
    if setup.offload_param:
        zero["offload_param"] = {"device": "cpu"}
    config = {
        "train_micro_batch_size_per_gpu": setup.micro_batch,
        "gradient_accumulation_steps": setup.grad_accum,
        "data_types": {"grad_accum_dtype": setup.dtype},
        "optimizer": {"type": "AdamW", "params": {"lr": lr}},
        "zero_optimization": zero,
    }
    if setup.dtype != "fp32":
        config[setup.dtype] = {"enabled": True}
    return config
