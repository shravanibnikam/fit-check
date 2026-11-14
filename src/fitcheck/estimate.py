"""Per-GPU training memory: model states, activations and the training peak.

Model states (weights, gradients, Adam states) follow the ZeRO paper's
accounting (Rajbhandari et al. 2020, arXiv:1910.02054), checked against
DeepSpeed's source. Activations come from activations.py. Derivation and
sources are in docs/memory-model.md.
"""

import json
import math
from dataclasses import dataclass
from importlib.resources import files

from fitcheck.activations import activations
from fitcheck.model import Model

# Bytes per element of the weights and gradients during training. DeepSpeed
# casts every parameter, LoRA adapters included, to fp16/bf16 when either is
# enabled (DeepSpeedEngine._cast_module_mixed_precision in
# deepspeed/runtime/engine.py), so one dtype covers the whole model.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}
F32_BYTES = 4

# DeepSpeed's zero_optimization defaults (deepspeed/runtime/zero/config.py),
# in elements. ZeRO-3 steps the optimizer one sub-group at a time.
ZERO3_SUB_GROUP = 1_000_000_000  # sub_group_size
REDUCE_BUCKET = 500_000_000  # reduce_bucket_size
MAX_REUSE_DISTANCE = 1_000_000_000  # stage3_max_reuse_distance


def _calibration() -> dict | None:
    """What `fitcheck calibrate` fitted from measured runs, if it has run."""
    path = files("fitcheck") / "calibration.json"
    return json.loads(path.read_text()) if path.is_file() else None


# Fitted by `fitcheck calibrate` into calibration.json, with provenance. Until
# then the neutral values leave the estimate as bare theory.
CALIBRATION = _calibration()
# CALIBRATE: allocator cache and fragmentation, reserved / allocated.
RESERVED_OVER_ALLOCATED = CALIBRATION["reserved_over_allocated"] if CALIBRATION else 1.0
# CALIBRATE: device memory outside PyTorch's allocator: CUDA context, NCCL.
CUDA_CONTEXT_BYTES = CALIBRATION["cuda_context_bytes"] if CALIBRATION else 0


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
    grads: int  # held at the optimizer step
    optimizer: int
    activations: int | None  # None when the architecture has no formula
    logits: int | None  # output head: logits, loss and its backward
    step: int  # temporary fp32 gradient copies made at the optimizer step
    buffers: int  # DeepSpeed buffers on the GPU for the whole run
    gathered: int  # ZeRO-3 parameters still gathered at the start of backward
    allocated: int  # the peak: the larger of the two below
    backward: int  # at the start of backward
    at_step: int  # at the optimizer step
    overhead: int
    cpu: int
    trainable_params: int

    @property
    def peak_at(self) -> str:
        return "backward" if self.backward >= self.at_step else "step"

    @property
    def gpu_total(self) -> int:
        return self.allocated + self.overhead


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

    # Gradients held once backward ends: each rank's reduced partition with
    # ZeRO-1, 2 and 3. ZeRO-1 drops the full gradients too: with
    # contiguous_gradients (the default) they are copied into the reduce bucket
    # (reduce_independent_p_g_buckets_and_remove_grads in stage_1_and_2.py), and
    # ZeRO-1 measured the same as ZeRO-2 on the T4. With optimizer offload each
    # rank's partition goes straight into an fp32 CPU buffer
    # (copy_grads_in_partition in stage_1_and_2.py).
    if setup.offload_optimizer:
        grads = 0
    elif stage >= 1:
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

    # Temporary fp32 gradient copies at the step. Without ZeRO, FP16_Optimizer
    # converts each gradient to fp32 and then concatenates the copies into one
    # flat tensor, so both exist for a moment: 8 bytes per trainable parameter
    # (FP16_Optimizer.step in runtime/fp16/fused_optimizer.py). With ZeRO each
    # rank flattens its gradient partition and converts it to fp32: b + 4
    # bytes per parameter in the partition, one sub-group at a time at ZeRO-3
    # (stage_1_and_2.py step, _prepare_fp32_grad_for_sub_group in stage3.py).
    # Offload makes the copy on the CPU, and fp32 training makes none.
    step = 0
    if setup.dtype != "fp32" and not setup.offload_optimizer:
        if stage == 0:
            step = 2 * F32_BYTES * trainable
        elif stage < 3:
            step = (b + F32_BYTES) * math.ceil(trainable / n)
        else:
            step = (b + F32_BYTES) * min(math.ceil(trainable / n), ZERO3_SUB_GROUP)

    # Buffers DeepSpeed keeps on the GPU for the whole run. ZeRO-3 allocates a
    # reduce bucket of reduce_bucket_size elements at init, whatever the model
    # size (_setup_for_real_optimizer in stage3.py). ZeRO-1 and 2 with
    # optimizer offload keep a GPU buffer the size of the largest trainable
    # tensor (temp_grad_buffer_for_gpu_offload in stage_1_and_2.py). The ZeRO-1
    # and 2 reduce bucket only exists during backward, after the output head
    # has freed more than it takes (setup_buckets, _release_ipg_buffers).
    buffers = 0
    if stage == 3:
        buffers = b * REDUCE_BUCKET
    elif stage >= 1 and setup.offload_optimizer:
        buffers = b * _largest_trainable(model, setup)

    # ZeRO-3 preallocates the gradient partition at init and keeps it for the
    # whole run (grad_partitions_flat_buffer in stage3.py), on the CPU with
    # optimizer offload. ZeRO-0, 1 and 2 only create gradients during backward.
    held_in_backward = grads if stage == 3 else 0
    gathered = b * _gathered_at_backward(model, total) if stage == 3 else 0

    act = activations(model, setup, b)
    layers, logits = (act.layers, act.logits) if act else (None, None)
    # Activations peak at the start of backward and are gone by the optimizer
    # step, where the gradients and the fp32 copies are. ZeRO-3 releases the
    # gathered parameters when backward ends.
    backward = weights + optimizer + held_in_backward + buffers + gathered
    backward += (layers or 0) + (logits or 0)
    at_step = weights + optimizer + grads + step + buffers
    allocated = max(backward, at_step)
    overhead = round((RESERVED_OVER_ALLOCATED - 1) * allocated) + CUDA_CONTEXT_BYTES
    return Estimate(
        weights=weights,
        grads=grads,
        optimizer=optimizer,
        activations=layers,
        logits=logits,
        step=step,
        buffers=buffers,
        gathered=gathered,
        allocated=allocated,
        backward=backward,
        at_step=at_step,
        overhead=overhead,
        cpu=cpu,
        trainable_params=trainable,
    )


def _largest_trainable(model: Model, setup: Setup) -> int:
    """Elements in the largest tensor the optimizer updates."""
    if setup.lora_rank is None:
        return model.largest_param
    return setup.lora_rank * max((max(s) for s in model.lora_shapes), default=0)


def _gathered_at_backward(model: Model, total: int) -> int:
    """ZeRO-3 parameters still gathered when backward starts, in elements.

    After a module runs, ZeRO-3 keeps its parameters gathered if the trace uses
    them again within stage3_max_reuse_distance elements of the modules that
    come next (__params_to_release in partitioned_param_coordinator.py). The
    trace runs the embedding, the layers and the head in forward, then the
    head, the layers and the embedding in backward. So layer i is reused after
    2 * (layers after it + head) elements. This counts whole layers, not single
    Linears, and the layers as equal in size.
    """
    emb = model.embedding_params
    head = emb  # tied: the embedding itself; untied: assume the same shape
    per_layer = (
        total - emb - (0 if model.tied_embeddings else head)
    ) / model.num_layers
    kept = head  # runs last in forward and first in backward
    for after in range(model.num_layers):  # layers after this one
        if 2 * (after * per_layer + head) < MAX_REUSE_DISTANCE:
            kept += per_layer
    # An untied embedding runs again only at the very end of backward.
    if not model.tied_embeddings and 2 * (total - emb) < MAX_REUSE_DISTANCE:
        kept += emb
    return min(round(kept), total)
