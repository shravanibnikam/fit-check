"""Per-GPU training memory: model states, activations and the training peak.

Model states (weights, gradients, Adam states) follow the ZeRO paper's
accounting (Rajbhandari et al. 2020, arXiv:1910.02054), checked against
DeepSpeed's source. Activations come from activations.py. Derivation and
sources are in docs/memory-model.md.
"""

import math
from dataclasses import dataclass

from fitcheck.activations import activations
from fitcheck.model import Model

# Bytes per element of the weights and gradients during training. DeepSpeed
# casts every parameter, LoRA adapters included, to fp16/bf16 when either is
# enabled (DeepSpeedEngine._cast_module_mixed_precision in
# deepspeed/runtime/engine.py), so one dtype covers the whole model.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}
F32_BYTES = 4

# ZeRO-3 steps the optimizer one sub-group of parameters at a time
# (zero_optimization.sub_group_size, default 1e9 elements).
ZERO3_SUB_GROUP = 1_000_000_000

# Placeholders until Phase 3 fits them from measured runs. The neutral values
# leave the estimate as bare theory.
RESERVED_OVER_ALLOCATED = 1.0  # CALIBRATE: allocator cache and fragmentation
CUDA_CONTEXT_BYTES = 0  # CALIBRATE: CUDA context, outside PyTorch's allocator


@dataclass(frozen=True)
class Setup:
    num_gpus: int = 1
    zero_stage: int = 0
    offload_optimizer: bool = False
    offload_param: bool = False
    lora_rank: int | None = None  # None means full fine-tuning
    dtype: str = "bf16"
    micro_batch: int = 1  # sequences per GPU per forward pass
    seq_len: int = 2048
    attn: str = "sdpa"  # attn_implementation, "sdpa" or "eager"
    checkpointing: bool = False  # model.gradient_checkpointing_enable()

    def __post_init__(self):
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"dtype must be one of {sorted(DTYPE_BYTES)}")
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError("zero_stage must be 0, 1, 2 or 3")
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if self.lora_rank is not None and self.lora_rank < 1:
            raise ValueError("lora_rank must be at least 1")
        if self.micro_batch < 1 or self.seq_len < 1:
            raise ValueError("micro_batch and seq_len must be at least 1")
        if self.attn not in ("sdpa", "eager"):
            raise ValueError('attn must be "sdpa" or "eager"')
        # Both rules come from DeepSpeedZeroConfig in deepspeed/runtime/zero/config.py.
        if self.offload_optimizer and self.zero_stage == 0:
            raise ValueError("optimizer offload needs ZeRO stage 1, 2 or 3")
        if self.offload_param and self.zero_stage != 3:
            raise ValueError("parameter offload needs ZeRO stage 3")


@dataclass(frozen=True)
class Estimate:
    """Bytes per GPU, except `cpu`, which is host RAM summed over all ranks."""

    weights: int
    grads: int
    optimizer: int
    activations: int | None  # None when the architecture has no formula
    logits: int | None  # output head: logits, loss and its backward
    step: int  # fp32 gradient copy made at the optimizer step
    overhead: int
    cpu: int
    trainable_params: int

    @property
    def peak_extra(self) -> int:
        """What sits on top of the model states at the peak.

        Activations peak at the start of backward and are gone by the
        optimizer step, so only the larger of the two counts.
        """
        return max((self.activations or 0) + (self.logits or 0), self.step)

    @property
    def gpu_total(self) -> int:
        states = self.weights + self.grads + self.optimizer
        return states + self.peak_extra + self.overhead


def lora_params(model: Model, rank: int) -> int:
    """Adapter parameters for LoRA on every q, k, v and o projection.

    LoRA freezes a Linear(in, out) and trains A (rank x in) and B (out x rank)
    beside it, adding rank * (in + out) parameters (Hu et al. 2021,
    arXiv:2106.09685). The shapes come from the checkpoint, so grouped-query
    attention, Qwen3.5's gated q_proj and Gemma 4's layers without K/V weights
    are counted as they are.
    """
    return rank * sum(n_in + n_out for n_in, n_out in model.lora_shapes)


def estimate(model: Model, setup: Setup) -> Estimate:
    b = DTYPE_BYTES[setup.dtype]
    n = setup.num_gpus
    stage = setup.zero_stage

    if setup.lora_rank is None:
        total = trainable = model.num_params
    else:
        trainable = lora_params(model, setup.lora_rank)
        total = model.num_params + trainable

    # Weights: one copy in the training dtype. Only ZeRO-3 partitions it, and
    # with parameter offload it lives on the CPU between uses.
    if setup.offload_param:
        weights = 0
    elif stage == 3:
        weights = math.ceil(b * total / n)
    else:
        weights = b * total

    # Gradients: one per trainable parameter. ZeRO-2 and 3 partition them. With
    # optimizer offload at those stages each rank's partition is copied straight
    # into an fp32 CPU buffer (copy_grads_in_partition in stage_1_and_2.py).
    if stage >= 2 and setup.offload_optimizer:
        grads = 0
    elif stage >= 2:
        grads = math.ceil(b * trainable / n)
    else:
        grads = b * trainable

    # Optimizer: an fp32 master copy plus Adam's two fp32 moments, 12 bytes per
    # trainable parameter. ZeRO clones the master partition even for an fp32
    # model (stage_1_and_2.py, single_partition_of_fp32_groups). Only fp32
    # without ZeRO steps the parameters directly and keeps just the moments.
    k = 8 if setup.dtype == "fp32" and stage == 0 else 12
    if setup.offload_optimizer:
        optimizer = 0
    elif stage >= 1:
        optimizer = math.ceil(k * trainable / n)
    else:
        optimizer = k * trainable

    # Offloaded optimizer: fp32 master, two moments and an fp32 gradient
    # partition per rank (initialize_optimizer_states in stage_1_and_2.py).
    cpu = 0
    if setup.offload_optimizer:
        cpu += 16 * trainable
    if setup.offload_param:
        cpu += b * total

    # At the step, DeepSpeed copies the gradients to fp32 for Adam: all of them
    # without ZeRO (FP16_Optimizer.step in runtime/fp16/fused_optimizer.py),
    # each rank's partition with ZeRO-1/2 (stage_1_and_2.py), one sub-group at
    # a time with ZeRO-3 (_prepare_fp32_grad_for_sub_group in stage3.py). With
    # offload the copy is on the CPU, and in fp32 training there is no copy.
    step = 0
    if setup.dtype != "fp32" and not setup.offload_optimizer:
        if stage == 0:
            step = F32_BYTES * trainable
        elif stage < 3:
            step = math.ceil(F32_BYTES * trainable / n)
        else:
            step = F32_BYTES * min(math.ceil(trainable / n), ZERO3_SUB_GROUP)

    act = activations(model, setup, b)
    layers, logits = (act.layers, act.logits) if act else (None, None)
    extra = max((layers or 0) + (logits or 0), step)  # as Estimate.peak_extra
    allocated = weights + grads + optimizer + extra
    overhead = round((RESERVED_OVER_ALLOCATED - 1) * allocated) + CUDA_CONTEXT_BYTES
    return Estimate(
        weights, grads, optimizer, layers, logits, step, overhead, cpu, trainable
    )
