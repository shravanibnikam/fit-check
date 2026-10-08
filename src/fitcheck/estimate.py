"""Per-GPU training memory for model states: weights, gradients and Adam states.

Follows the ZeRO paper's accounting (Rajbhandari et al. 2020,
arXiv:1910.02054), checked against DeepSpeed's source. Activations are not
modeled yet. Derivation and sources are in docs/memory-model.md.
"""

import math
from dataclasses import dataclass

from fitcheck.model import Model

# Bytes per element of the weights and gradients during training. DeepSpeed
# casts every parameter, LoRA adapters included, to fp16/bf16 when either is
# enabled (DeepSpeedEngine._cast_module_mixed_precision in
# deepspeed/runtime/engine.py), so one dtype covers the whole model.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

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

    def __post_init__(self):
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"dtype must be one of {sorted(DTYPE_BYTES)}")
        if self.zero_stage not in (0, 1, 2, 3):
            raise ValueError("zero_stage must be 0, 1, 2 or 3")
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be at least 1")
        if self.lora_rank is not None and self.lora_rank < 1:
            raise ValueError("lora_rank must be at least 1")
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
    overhead: int
    cpu: int
    trainable_params: int

    @property
    def gpu_total(self) -> int:
        return self.weights + self.grads + self.optimizer + self.overhead


def lora_params(model: Model, rank: int) -> int:
    """Adapter parameters for LoRA on the q, k, v and o projections of every layer.

    LoRA freezes a Linear(in, out) and trains A (rank x in) and B (out x rank)
    beside it, adding rank * (in + out) parameters (Hu et al. 2021,
    arXiv:2106.09685). With grouped-query attention k and v are narrower.
    """
    h = model.hidden_size
    q = model.num_heads * model.head_dim
    kv = model.num_kv_heads * model.head_dim
    per_layer = (h + q) + 2 * (h + kv) + (q + h)  # q_proj, k_proj, v_proj, o_proj
    return rank * per_layer * model.num_layers


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

    allocated = weights + grads + optimizer
    overhead = round((RESERVED_OVER_ALLOCATED - 1) * allocated) + CUDA_CONTEXT_BYTES
    return Estimate(weights, grads, optimizer, overhead, cpu, trainable)
