"""Activation memory: the tensors autograd keeps from the forward pass for backward.

Each term below is one tensor saved by one op of the Hugging Face modeling code
(transformers 5.19), counted under DeepSpeed's training setup: every parameter
cast to the training dtype, no dropout, no padding, and the loss computed by the
model from `labels`. An op only saves what its required gradients need, so
frozen weights (LoRA) and inputs without gradients save less. Every formula is
checked tensor by tensor against PyTorch by scripts/saved_tensors.py.
Derivation and sources: docs/memory-model.md.
"""

from dataclasses import dataclass

F32 = 4  # the modeling code upcasts norms, softmax and the loss to fp32
I64 = 8  # the shifted labels kept by the loss

# HF's linear-attention fallback chunks the sequence (torch_chunk_gated_delta_rule).
DELTA_CHUNK = 64


@dataclass(frozen=True)
class Parts:
    """Saved bytes for one forward pass, before combining into a peak."""

    shared: int  # used by many layers, stored once: RoPE tables, Gemma 4 PLE inputs
    embed: int  # saved between the embedding and the first layer
    layers: list[int]  # saved inside each decoder layer
    layer_input: int  # one layer's input hidden states, what checkpointing keeps
    head: int  # final norm, logits and loss, saved
    held: int  # logits returned to the caller, alive during backward
    loss_backward: int  # transient buffers of the loss backward


@dataclass(frozen=True)
class Activations:
    """Activation bytes per GPU at the training peak, split for display."""

    layers: int  # everything before the output head
    logits: int  # output head: logits, loss and the loss backward


def activations(model, setup, dtype_bytes: int) -> Activations | None:
    """Peak activation memory, or None for an architecture without a formula."""
    if model.family is None:
        return None
    parts = FAMILIES[model.family](model.config, setup, dtype_bytes)
    logits = parts.head + parts.held + parts.loss_backward
    if not setup.checkpointing:
        return Activations(parts.shared + parts.embed + sum(parts.layers), logits)
    # HF checkpoints each decoder layer (GradientCheckpointingLayer, use_reentrant
    # False): the forward keeps only each layer's input, and backward recomputes
    # one layer at a time. The output head runs first in backward and frees its
    # buffers before the first recompute, so the peak is whichever is larger.
    kept = parts.shared + parts.embed + len(parts.layers) * parts.layer_input
    if logits >= max(parts.layers):
        return Activations(kept, logits)
    return Activations(kept + max(parts.layers), 0)


def _sizes(setup):
    b, s = setup.micro_batch, setup.seq_len
    return b, s, b * s, setup.lora_rank is None, setup.lora_rank or 0


def _head(t, p, vocab, final_norm, full, h, softcap=False):
    """Final norm, logits and loss (LlamaForCausalLM.forward, ForCausalLMLoss)."""
    saved = final_norm
    if full:
        # Norm output, kept by lm_head for its weight gradient.
        saved += t * p * h
    if softcap:
        # Gemma's tanh(logits / cap) * cap keeps the tanh output.
        saved += t * p * vocab
    # The loss upcasts logits to fp32 and keeps the log-softmax output.
    saved += t * F32 * vocab + t * I64 + F32  # and nll_loss's total_weight scalar
    held = t * p * vocab
    # Backward of log-softmax holds the incoming gradient and its result, both
    # fp32 and logits-sized, next to the saved output.
    return saved, held, 2 * t * F32 * vocab


def _attention(t, b, s, heads, q, kv, p, attn, repeat, mask):
    """Scaled dot-product attention, after RoPE (which keeps only cos and sin).

    `repeat`: SDPA gets K/V copied to every query head (see _copies_kv).
    """
    if attn == "eager":
        # matmul keeps q and the repeated k; softmax keeps its fp32 output; the
        # second matmul keeps its dtype copy and the repeated v; the next layer
        # keeps the transposed copy of the output.
        probs = p if p != F32 else 0  # .to(dtype) is a no-op in fp32
        return t * (4 * p * q + (F32 + probs) * heads * s)
    # SDPA keeps q, k, v, its output and the fp32 logsumexp. On CUDA its output
    # already has the (batch, seq, heads, dim) layout, so the transpose that
    # feeds o_proj makes no copy.
    k = q if repeat else kv
    saved = t * (2 * p * q + 2 * p * k + F32 * heads)
    # CUDA's memory-efficient kernel (the T4's) also keeps its philox seed and
    # offset, two int64 scalars on the GPU, even without dropout
    # (_efficient_attention_forward in aten/src/ATen/native/transformers/cuda/
    # attention.cu). Measured on a T4 by scripts/saved_tensors.py --device cuda.
    saved += 2 * I64
    if mask:
        # SDPA converts the boolean mask to a float bias on every call.
        saved += b * s * s * p
    return saved


def _copies_kv(heads, kv_heads, d, mask):
    """Whether HF copies K/V to every query head before SDPA.

    On CUDA, HF lets SDPA do grouped-query attention natively only without a
    mask and with head_dim <= 256 (use_gqa_in_sdpa in
    integrations/sdpa_attention.py); otherwise repeat_kv expands K/V. With a
    single KV head that expand is a stride-0 view, so nothing is copied.
    """
    return heads > kv_heads > 1 and (mask or d > 256)


# Llama, Mistral, Qwen2 -------------------------------------------------------


def _llama_norm(t, w, grad_in, trainable, p):
    """LlamaRMSNorm: fp32 copy of x, rsqrt per row, then weight * x.to(dtype)."""
    if not grad_in:
        return 0
    return t * (F32 * w + F32 + (p * w if trainable else 0))


def _windows(cfg, n_layers):
    """Each layer's sliding window, or None, as transformers configures them.

    Mistral slides every layer, over 4096 tokens unless config.json says
    otherwise (MistralConfig). Qwen2 slides only with use_sliding_window, and
    then the layers its layer_types marks, by default those from
    max_window_layers on (Qwen2Config.__post_init__).
    """
    kind = cfg.get("model_type")
    if kind == "mistral":
        return [cfg.get("sliding_window", 4096)] * n_layers
    if kind == "qwen2" and cfg.get("use_sliding_window"):
        window = cfg.get("sliding_window", 4096)
        first = cfg.get("max_window_layers", 28)
        types = cfg.get("layer_types") or [
            "sliding_attention" if i >= first else "full_attention"
            for i in range(n_layers)
        ]
        return [window if x == "sliding_attention" else None for x in types]
    return [None] * n_layers


def _llama(cfg, setup, p):
    b, s, t, full, r = _sizes(setup)
    h, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads") or heads
    d = cfg.get("head_dim") or h // heads
    q, kv, inter = heads * d, kv_heads * d, cfg["intermediate_size"]
    layers = []
    for i, window in enumerate(_windows(cfg, n_layers)):
        # A sliding layer needs an explicit mask once the sequence reaches the
        # window (_ignore_causal_mask_sdpa in masking_utils.py), as in Gemma 4.
        mask = window is not None and s >= window
        repeat = _copies_kv(heads, kv_heads, d, mask)
        # With LoRA the embeddings are frozen, so nothing before layer 0's
        # adapters needs a gradient.
        n = _llama_norm(t, h, full or i > 0, full, p)
        # Norm output, kept by q/k/v (or by LoRA's A); each adapter's A output,
        # kept by its B.
        n += t * p * h + (0 if full else 4 * t * p * r)
        n += _attention(t, b, s, heads, q, kv, p, setup.attn, repeat, mask)
        n += _llama_norm(t, h, True, full, p)
        # MLP input, kept by gate_proj and up_proj when they train.
        n += t * p * h if full else 0
        # Gate output (SiLU input), SiLU output and up output (kept by the
        # product), and the product (kept by down_proj when it trains).
        n += t * p * inter * (4 if full else 3)
        layers.append(n)
    final_norm = _llama_norm(t, h, True, full, p)
    head = _head(t, p, cfg["vocab_size"], final_norm, full, h)
    # cos and sin, batch 1 since position_ids are shared across the batch.
    return Parts(2 * p * s * d, 0, layers, t * p * h, *head)


# Gemma 4 ----------------------------------------------------------------------


def _gemma_norm(t, rows, d, trainable, p, scale=True, grad_in=True):
    """Gemma4RMSNorm: x.float() * pow(mean(x^2) + eps, -0.5) * weight.float()."""
    if not grad_in:
        return 0
    # fp32 x, plus the mean square and its power per row.
    n = t * rows * (F32 * d + 2 * F32)
    if scale:
        # weight.float(), one copy per call unless the weight is already fp32,
        # and the normalized x for the weight gradient.
        n += (F32 * d if p != F32 else 0) + (t * rows * F32 * d if trainable else 0)
    return n


def _gemma4(cfg, setup, p):
    b, s, t, full, r = _sizes(setup)
    h, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    # Defaults are Gemma4TextConfig's (configuration_gemma4.py).
    window = cfg.get("sliding_window", 512)
    types = cfg.get("layer_types") or [
        "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
        for i in range(n_layers)
    ]
    # Full-attention layers override head_dim and, with attention_k_eq_v, the
    # KV head count. config.json gives the global values; an in-memory config
    # carries the derived per_layer_config (Gemma4TextConfig.__post_init__).
    global_d = cfg.get("global_head_dim") or 512
    global_kv = cfg.get("num_global_key_value_heads")
    if not cfg.get("attention_k_eq_v", False) or global_kv is None:
        global_kv = kv_heads
    overrides = cfg.get("per_layer_config") or {}
    n_shared = cfg.get("num_kv_shared_layers", 0)
    ple = cfg.get("hidden_size_per_layer_input", 256)
    inter = cfg["intermediate_size"]
    # Each attention type's last layer before the shared ones lends its K/V to
    # the shared layers of that type (store_full_length_kv in Gemma4TextAttention).
    first_shared = n_layers - n_shared if n_shared else n_layers
    lenders = {kind: i for i, kind in enumerate(types[:first_shared])}
    borrowed = {kind for kind in types[first_shared:]}

    shared = 0
    ds = {}  # head_dim per attention type, for the RoPE tables
    layers = []
    for i, kind in enumerate(types):
        sliding = kind == "sliding_attention"
        d = cfg["head_dim"] if sliding else global_d
        kvh = kv_heads if sliding else global_kv
        if overrides:
            o = overrides.get(i) or overrides.get(str(i)) or {}
            d = o.get("head_dim", cfg["head_dim"])
            kvh = o.get("num_key_value_heads", kv_heads)
        ds[kind] = d
        q, kv = heads * d, kvh * d
        # The last num_kv_shared_layers layers reuse an earlier layer's K/V.
        reuses_kv = n_shared > 0 and i >= n_layers - n_shared
        # Shorter sequences need no sliding mask and fall back to is_causal.
        mask = sliding and s >= window
        repeat = _copies_kv(heads, kvh, d, mask)

        # LoRA targets in this layer: q and o always, k unless K/V are borrowed,
        # v unless it is also k (attention_k_eq_v on full-attention layers).
        k_eq_v = cfg.get("attention_k_eq_v", False) and not sliding
        adapters = 2 + (0 if reuses_kv else 1 + (0 if k_eq_v else 1))

        n = _gemma_norm(t, 1, h, full, p, grad_in=full or i > 0)  # input_layernorm
        # Its output, kept by the projections, and LoRA's A outputs.
        n += t * p * h + (0 if full else t * p * r * adapters)
        n += _gemma_norm(t, heads, d, full, p)  # q_norm
        if not reuses_kv:
            n += _gemma_norm(t, kvh, d, full, p)  # k_norm
            n += _gemma_norm(t, kvh, d, full, p, scale=False)  # v_norm, no weight
            if k_eq_v and p == F32:
                # v_norm's fp32 input is k_norm's input itself (.float() is a
                # no-op), already counted.
                n -= t * F32 * kv
        if setup.attn == "sdpa" and not repeat and kind in borrowed:
            if lenders.get(kind) == i:
                # The lender's K/V are kept by every layer that reads them.
                shared += 2 * t * p * kv
            if reuses_kv or lenders.get(kind) == i:
                kv = 0
        n += _attention(t, b, s, heads, q, kv, p, setup.attn, repeat, mask)
        n += _gemma_norm(t, 1, h, full, p)  # post_attention_layernorm
        n += _gemma_norm(t, 1, h, full, p)  # pre_feedforward_layernorm
        n += t * p * h if full else 0  # MLP input, kept by gate_proj and up_proj
        wide = inter * (2 if cfg.get("use_double_wide_mlp") and reuses_kv else 1)
        n += t * p * wide * (4 if full else 3)  # as Llama, with GELU(tanh)
        n += _gemma_norm(t, 1, h, full, p)  # post_feedforward_layernorm
        if ple:
            # Per-layer embeddings: gate (h -> ple), GELU, times this layer's
            # slice of the per-layer inputs, projection back to h, then a norm.
            # Gate input, GELU output and projection input are kept only for
            # weight and PLE gradients, i.e. in full fine-tuning.
            n += t * p * ple + (t * p * (h + 2 * ple) if full else 0)
            n += _gemma_norm(t, 1, h, full, p)  # post_per_layer_input_norm
        layers.append(n)

    # RoPE cos and sin for each attention type, batch 1.
    shared += sum(2 * p * s * d for d in ds.values())
    embed = 0
    if ple:
        shared += t * p * n_layers * ple  # the per-layer inputs, one slice per layer
        if full:
            # Gemma4TextModel.project_per_layer_inputs: projection of the token
            # embeddings, then a norm over each layer's slice.
            embed = t * p * h + _gemma_norm(t, n_layers, ple, True, p)
            if p == F32:
                # In fp32, layer 0's norm keeps the embeddings themselves
                # (.float() is a no-op): the same tensor the projection keeps.
                embed -= t * p * h
                shared += t * p * h
                layers[0] -= t * F32 * h
    softcap = cfg.get("final_logit_softcapping") is not None
    final_norm = _gemma_norm(t, 1, h, full, p)
    head = _head(t, p, cfg["vocab_size"], final_norm, full, h, softcap)
    return Parts(shared, embed, layers, t * p * h, *head)


# Qwen3.5 ----------------------------------------------------------------------


def _qwen_norm(t, rows, d, grad_in, trainable):
    """Qwen3_5RMSNorm: x.float() * rsqrt(mean(x^2) + eps) * (1 + weight.float())."""
    if not grad_in:
        return 0
    # fp32 x, rsqrt per row, and 1 + weight.float() once per call.
    n = t * rows * (F32 * d + F32) + F32 * d
    # The normalized x, for the weight gradient.
    return n + (t * rows * F32 * d if trainable else 0)


def _delta_net(t, b, s, h, cfg, p, full):
    """Gated DeltaNet through HF's PyTorch fallback (no flash-linear-attention).

    Qwen3_5GatedDeltaNet.forward and torch_chunk_gated_delta_rule in
    modeling_qwen3_5.py. Everything inside the chunked rule is fp32.
    """
    hk, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    conv = 2 * hk * dk + hv * dv
    kernel = cfg.get("linear_conv_kernel_dim", 4)
    chunks = -(-s // DELTA_CHUNK)
    tp = b * chunks * DELTA_CHUNK  # tokens after padding to whole chunks

    # Norm output, kept by the four input projections when they train, and the
    # causal conv1d input, which ConvolutionBackward keeps even with a frozen
    # weight.
    n = t * p * (h if full else 0) + t * p * conv
    # SiLU input: the conv output before its kernel - 1 padding is trimmed.
    n += b * p * conv * (s + kernel - 1)
    n += t * p * hv  # sigmoid(b), the beta gate
    # g = -exp(A_log) * softplus(a + dt_bias). softplus keeps its input; the
    # product keeps both factors when A_log trains, else just -exp(A_log).
    n += (t + 1) * F32 * hv * (2 if full else 1)
    # q and k, repeated to the value heads, cast to fp32 and L2-normalized:
    # each keeps its fp32 input and its inverse norm.
    n += 2 * t * F32 * hv * (dk + 1)
    # Scaled q (kept by q @ k^T and its decay), normalized k (kept by k_beta and
    # k_beta @ k^T), k_beta, and the two decayed copies of q and k.
    n += 5 * tp * F32 * hv * dk
    n += tp * F32 * hv * (dv + 1)  # v and beta, kept by v * beta
    # Five chunk-local C x C matrices per head: the pairwise decay, k_beta @ k^T,
    # the UT system, q @ k^T and the intra-chunk attention.
    n += 5 * tp * F32 * hv * DELTA_CHUNK
    n += DELTA_CHUNK * DELTA_CHUNK  # the boolean causal mask of one chunk
    n += 3 * tp * F32 * hv  # exp of the cumulative decay, three variants
    n += b * chunks * F32 * hv  # each chunk's total decay
    n += tp * F32 * hv * (dv + dk)  # the two triangular-solve results
    # The scan over chunks keeps each chunk's recurrent state (hv x dk x dv)
    # and the corrected values.
    n += b * chunks * F32 * hv * dk * dv + tp * F32 * hv * dv
    # Qwen3_5RMSNormGated over each value head: fp32 x and rsqrt, x.to(dtype)
    # for the weight gradient, then weight * x, fp32 z and silu(z), kept by the
    # gate product.
    n += t * hv * (F32 * dv + F32) + (t * p * hv * dv if full else 0)
    n += t * hv * dv * (p + 2 * F32)
    n += t * p * hv * dv if full else 0  # out_proj input
    return n


def _qwen3_5(cfg, setup, p):
    b, s, t, full, r = _sizes(setup)
    h, n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    # Defaults are Qwen3_5TextConfig's (configuration_qwen3_5.py).
    d, inter = cfg.get("head_dim", 256), cfg["intermediate_size"]
    q, kv = heads * d, kv_heads * d
    interval = cfg.get("full_attention_interval", 4)
    types = cfg.get("layer_types") or [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n_layers)
    ]
    rope = cfg.get("rope_parameters") or {}
    rot = int(d * rope.get("partial_rotary_factor", 0.25))
    repeat = _copies_kv(heads, kv_heads, d, False)

    layers = []
    grad = full  # with LoRA, gradients start at the first adapter
    for kind in types:
        n = _qwen_norm(t, 1, h, grad, full)  # input_layernorm
        if kind == "linear_attention":
            n += _delta_net(t, b, s, h, cfg, p, full) if grad else 0
        else:
            # Norm output, kept by q/k/v (or LoRA's A), and A's outputs.
            n += t * p * h + (0 if full else 4 * t * p * r)
            n += _qwen_norm(t, heads, d, True, full)  # q_norm
            if p == F32:
                # In fp32, q_norm keeps a view of q_proj's output (.float() is a
                # no-op), whose storage also holds the gate half.
                n += t * F32 * q
            n += _qwen_norm(t, kv_heads, d, True, full)  # k_norm
            n += _attention(t, b, s, heads, q, kv, p, setup.attn, repeat, False)
            # Output gate: sigmoid(gate), kept by sigmoid and by the product,
            # and the product, kept by o_proj.
            n += 2 * t * p * q
            grad = True
        n += _qwen_norm(t, 1, h, grad, full)  # post_attention_layernorm
        if grad:
            n += t * p * h if full else 0
            n += t * p * inter * (4 if full else 3)  # SwiGLU, as Llama
        layers.append(n)
    final_norm = _qwen_norm(t, 1, h, True, full)
    head = _head(t, p, cfg["vocab_size"], final_norm, full, h)
    # Multimodal RoPE builds cos and sin per batch row, for the rotary part only.
    return Parts(2 * p * t * rot, 0, layers, t * p * h, *head)


FAMILIES = {"llama": _llama, "gemma4": _gemma4, "qwen3_5": _qwen3_5}
