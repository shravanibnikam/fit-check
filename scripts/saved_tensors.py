"""Check fitcheck's activation formulas against what PyTorch autograd saves.

For small random-weight models that keep each family's structure (grouped-query
attention, sliding windows, shared K/V, per-layer embeddings, linear attention),
this runs one training forward pass on CPU and adds up the bytes of every
distinct storage that autograd keeps for backward, by decoder layer. It then
compares those bytes with fitcheck.activations, bucket by bucket.

These are exact byte counts of saved tensors, not GPU measurements: allocator
overhead, CUDA workspaces and transient peaks are left to the Phase 3
calibration runs.

Run:  uv run --group check python scripts/saved_tensors.py [-v]
"""

import sys

import torch
import transformers.integrations.sdpa_attention as hf_sdpa
from transformers import (
    AutoModelForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    MistralConfig,
    Qwen2Config,
    Qwen3_5TextConfig,
)

from fitcheck import activations, estimate, model

# HF picks SDPA's native grouped-query path per device. On macOS the MPS check
# also enables it for masked CPU calls. Pin the CUDA rule (use_gqa_in_sdpa in
# transformers/integrations/sdpa_attention.py) so CPU runs match a GPU.
hf_sdpa.use_gqa_in_sdpa = lambda mask, k, v: (
    mask is None and k.shape[-1] == v.shape[-1] <= 256
)

SLIDING, FULL, LINEAR = "sliding_attention", "full_attention", "linear_attention"

CONFIGS = {
    "llama": LlamaConfig(
        hidden_size=64,
        intermediate_size=176,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=500,
        tie_word_embeddings=False,
    ),
    # Like Mistral-7B-v0.1: every layer slides, so past the window SDPA gets a
    # mask and K/V are copied to every query head.
    "mistral": MistralConfig(
        hidden_size=64,
        intermediate_size=176,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=500,
        sliding_window=16,
        tie_word_embeddings=False,
    ),
    # Qwen2 with use_sliding_window: only the layers from max_window_layers on
    # slide. Also covers the q/k/v biases.
    "qwen2-sliding": Qwen2Config(
        hidden_size=64,
        intermediate_size=176,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=500,
        use_sliding_window=True,
        sliding_window=16,
        max_window_layers=1,
        tie_word_embeddings=False,
    ),
    # Like gemma-4-E2B: per-layer embeddings, shared K/V, double-wide MLP.
    "gemma4-e2b": Gemma4TextConfig(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        # The last layer must be full attention, or transformers forces it.
        layer_types=[SLIDING, SLIDING, FULL, SLIDING, SLIDING, FULL, SLIDING, FULL],
        sliding_window=16,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=500,
        num_kv_shared_layers=4,
        use_double_wide_mlp=True,
        final_logit_softcapping=30.0,
    ),
    # Like gemma-4-31B: K = V on global layers with fewer KV heads, and a global
    # head_dim over 256, which makes HF copy K/V to every head before SDPA.
    "gemma4-31b": Gemma4TextConfig(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=288,
        num_global_key_value_heads=2,
        layer_types=[SLIDING, SLIDING, FULL] * 2,
        sliding_window=16,
        hidden_size_per_layer_input=0,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
    ),
    # Like Qwen3.5: three Gated DeltaNet layers per gated full-attention layer.
    "qwen3.5": Qwen3_5TextConfig(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        layer_types=[LINEAR, LINEAR, LINEAR, FULL] * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    ),
}
# Sequence lengths past the sliding window and not a whole number of 64-token
# DeltaNet chunks, to exercise masks and padding.
SEQ = {
    "llama": 48,
    "mistral": 48,
    "qwen2-sliding": 48,
    "gemma4-e2b": 40,
    "gemma4-31b": 40,
    "qwen3.5": 100,
}
BATCH = 2
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def measure(config, attn, lora, dtype):
    """Saved bytes per bucket: shared, embed, L0..Ln, head, held."""
    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_config(config, attn_implementation=attn)
    if lora:
        from peft import LoraConfig, get_peft_model

        targets = list(model.LORA_TARGETS)
        lm = get_peft_model(
            lm, LoraConfig(r=lora, target_modules=targets, lora_dropout=0.0)
        )
    # deepspeed.initialize casts every parameter, LoRA adapters included, to the
    # training dtype (DeepSpeedEngine._cast_module_mixed_precision).
    lm = lm.to(DTYPES[dtype]).train()
    base = lm.get_base_model() if lora else lm
    states = {x.untyped_storage().data_ptr() for x in [*lm.parameters(), *lm.buffers()]}

    where = ["embed"]
    for i, layer in enumerate(base.model.layers):
        layer.register_forward_pre_hook(lambda m, a, i=i: where.__setitem__(0, f"L{i}"))
    base.model.norm.register_forward_pre_hook(lambda m, a: where.__setitem__(0, "head"))

    first, kept = {}, []

    def pack(x):
        ptr = x.untyped_storage().data_ptr()
        first.setdefault(ptr, (where[0], x.untyped_storage().nbytes()))
        first[ptr] = (
            first[ptr][0] if first[ptr][0] == where[0] else "shared",
            first[ptr][1],
        )
        kept.append(x)  # keep storages alive so their addresses stay unique
        return x

    s = SEQ[next(k for k, v in CONFIGS.items() if v is config)]
    ids = torch.randint(0, config.vocab_size, (BATCH, s))
    states.add(ids.untyped_storage().data_ptr())
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        out = lm(input_ids=ids, labels=ids)

    # Count only storages the backward graph can still reach: tensors saved by
    # branches the loss does not depend on are freed with those branches.
    live, stack, seen = set(), [out.loss.grad_fn], set()
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        for attr in dir(fn):
            if attr.startswith("_saved_"):
                try:
                    v = getattr(fn, attr)
                except RuntimeError:
                    continue
                for x in v if isinstance(v, tuple | list) else [v]:
                    if isinstance(x, torch.Tensor):
                        live.add(x.untyped_storage().data_ptr())
        stack.extend(f for f, _ in fn.next_functions)

    buckets = {}
    for ptr, (bucket, nbytes) in first.items():
        if ptr in live and ptr not in states:
            buckets[bucket] = buckets.get(bucket, 0) + nbytes
    logits = out.logits.untyped_storage()
    if logits.data_ptr() not in first:
        buckets["held"] = logits.nbytes()
    return buckets, s


def predict(name, config, attn, lora, s, dtype):
    cfg = config.to_dict()
    family = model.FAMILIES[cfg["model_type"]]
    setup = estimate.Setup(
        lora_rank=lora, micro_batch=BATCH, seq_len=s, attn=attn, dtype=dtype
    )
    p = estimate.DTYPE_BYTES[dtype]
    parts = activations.FAMILIES[family](cfg, setup, p)
    buckets = {"shared": parts.shared, "embed": parts.embed, "head": parts.head}
    buckets.update({f"L{i}": n for i, n in enumerate(parts.layers)})
    buckets["held"] = parts.held
    if name == "qwen3.5" and attn == "sdpa":
        # CPU-only: PyTorch's CPU flash kernel returns its output in the query's
        # layout, and Qwen3.5's partial-rotary cat leaves q contiguous, so the
        # transpose before the output gate copies. CUDA kernels return the
        # output transposed already, so the formula has no such copy.
        q = config.num_attention_heads * config.head_dim
        for i, kind in enumerate(config.layer_types):
            if kind == FULL:
                buckets[f"L{i}"] += BATCH * s * p * q
    return buckets


def main():
    verbose = "-v" in sys.argv
    failures = 0
    for name, config in CONFIGS.items():
        for lora in (None, 8):
            for attn, dtype in (("sdpa", "bf16"), ("eager", "bf16"), ("sdpa", "fp32")):
                measured, s = measure(config, attn, lora, dtype)
                predicted = predict(name, config, attn, lora, s, dtype)
                keys = sorted(set(measured) | set(predicted), key=_order)
                bad = [k for k in keys if measured.get(k, 0) != predicted.get(k, 0)]
                total = sum(measured.values())
                kind = "lora" if lora else "full"
                label = f"{name:13} {kind:4} {attn:5} {dtype:4} s={s:<4}"
                print(f"{label} {total:>10,} bytes  {'ok' if not bad else 'MISMATCH'}")
                for k in keys if verbose or bad else []:
                    m, p = measured.get(k, 0), predicted.get(k, 0)
                    print(
                        f"    {k:7} measured {m:>9,}  predicted {p:>9,}  diff {p - m:+,}"
                    )
                failures += bool(bad)
    if failures:
        sys.exit(f"{failures} case(s) do not match")
    print("all cases match exactly")


def _order(key):
    names = ["shared", "embed"]
    return (
        (0, names.index(key))
        if key in names
        else (2, 0)
        if key in ("head", "held")
        else (1, int(key[1:]))
    )


if __name__ == "__main__":
    main()
