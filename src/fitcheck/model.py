"""The architecture fields the estimator needs, read from config.json and the
safetensors headers.

Uses the standard Hugging Face field names. `family` picks the activation
formula in activations.py. Other decoder-only models still get the model-state
estimate, just no activation estimate. `unmodeled` says in plain words what the
estimate leaves out, so the CLI never prints a confident verdict over a gap.
"""

import re
from dataclasses import dataclass, field

from fitcheck.safetensors_header import count_params

# Text model_type -> activation formula. Mistral and Qwen2 run the same ops as
# Llama: Qwen2's q/k/v biases don't change what autograd keeps for backward.
FAMILIES = {
    "llama": "llama",
    "mistral": "llama",
    "qwen2": "llama",
    "gemma4_text": "gemma4",
    "gemma4_unified_text": "gemma4",
    "qwen3_5_text": "qwen3_5",
}

# The modules LoRA wraps. PEFT matches a target against the last component of
# a module's dotted name, so "k_proj" does not match "relative_k_proj".
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

_SHARED_KV_WEIGHT = re.compile(
    r"(?:^model|language_model)\.layers\.(\d+)\.self_attn\.(?:k_proj|v_proj|k_norm|v_norm)\."
)
_DECODER_LAYER = re.compile(r"(?:^model|language_model)\.layers\.(\d+)\.")

# Expert counts as Qwen MoE, OLMoE and Gemma 4 (num_experts), Mixtral
# (num_local_experts) and DeepSeek (n_routed_experts) name them.
EXPERT_FIELDS = ("num_experts", "num_local_experts", "n_routed_experts")


@dataclass(frozen=True)
class Model:
    name: str
    num_params: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    family: str | None = None
    config: dict = field(default_factory=dict)  # the text config, for activations.py
    lora_shapes: tuple[tuple[int, int], ...] = ()  # (in, out) of each LoRA target
    lora_missing: dict = field(default_factory=dict)  # target -> layers without it
    unmodeled: tuple[str, ...] = ()  # what the estimate leaves out, in plain words


def from_config(name: str, config: dict, tensors: dict[str, dict]) -> Model:
    # Multimodal checkpoints (Gemma 4, Qwen3.5) nest the language model's fields
    # under text_config.
    cfg = config.get("text_config", config)

    def need(key: str):
        if key not in cfg:
            raise ValueError(
                f"{name}: config.json has no {key!r}. Only decoder-only models "
                "with standard Hugging Face field names are supported."
            )
        return cfg[key]

    hidden = need("hidden_size")
    heads = need("num_attention_heads")
    n_layers = need("num_hidden_layers")
    loaded = {k: v for k, v in tensors.items() if _is_loaded(k, cfg)}
    targets = {
        k: info
        for k, info in loaded.items()
        if k.endswith(".weight") and k.split(".")[-2] in LORA_TARGETS
        if len(info["shape"]) == 2
    }
    # Which decoder layers have each target, to warn when LoRA can't wrap all four.
    found = set()
    for k in targets:
        if m := _DECODER_LAYER.search(k):
            found.add((k.split(".")[-2], int(m.group(1))))
    missing = {
        t: [i for i in range(n_layers) if (t, i) not in found] for t in LORA_TARGETS
    }
    experts = next((cfg[k] for k in EXPERT_FIELDS if (cfg.get(k) or 0) > 1), None)
    # A dense formula would silently undercount an MoE layer (Gemma 4 26B-A4B
    # shares gemma4_text with the dense models), so MoE gets no formula yet.
    family = None if experts else FAMILIES.get(cfg.get("model_type"))
    return Model(
        name=name,
        num_params=count_params(loaded),
        hidden_size=hidden,
        num_layers=n_layers,
        num_heads=heads,
        # Without grouped-query attention every query head has its own K/V head.
        num_kv_heads=cfg.get("num_key_value_heads") or heads,
        # Usually hidden / heads, but Gemma and Qwen3 set it explicitly to
        # something else (e.g. Gemma 2 2B: hidden 2304, 8 heads, head_dim 256).
        head_dim=cfg.get("head_dim") or hidden // heads,
        family=family,
        config=cfg,
        lora_shapes=tuple(
            (info["shape"][1], info["shape"][0]) for info in targets.values()
        ),
        lora_missing={t: layers for t, layers in missing.items() if layers},
        unmodeled=_unmodeled(config, cfg, family, experts),
    )


def _unmodeled(config: dict, cfg: dict, family: str | None, experts) -> tuple[str, ...]:
    reasons = []
    if experts:
        reasons.append(
            f"mixture-of-experts layers ({experts} experts): every expert is "
            "counted in the model states, but activations and expert "
            "parallelism are not modeled yet"
        )
    if cfg.get("kv_lora_rank"):
        # DeepSeek's multi-head latent attention: q/k heads of qk_nope_head_dim
        # + qk_rope_head_dim, K/V projected through a low-rank latent.
        reasons.append(
            "MLA attention (kv_lora_rank): the head size shown is wrong, and "
            "LoRA on q/k/v/o misses its latent K/V projections"
        )
    # A top-level field: quantization applies to the whole checkpoint.
    quant = config.get("quantization_config")
    if quant:
        reasons.append(
            f"quantized checkpoint ({quant.get('quant_method', 'unknown method')}): "
            "packed formats such as GPTQ and AWQ undercount the parameters"
        )
    if family is None and not experts:
        kind = cfg.get("model_type")
        reasons.append(
            f"no activation formula for {repr(kind) if kind else 'this model'} yet: "
            "the total is model states only"
        )
    return tuple(reasons)


def _is_loaded(name: str, cfg: dict) -> bool:
    """False for checkpoint tensors that transformers drops when loading."""
    # Qwen3.5's multi-token-prediction head, never loaded for training
    # (_keys_to_ignore_on_load_unexpected in modeling_qwen3_5.py).
    if name.startswith("mtp."):
        return False
    # Gemma 4 layers that reuse an earlier layer's K/V have no K/V weights, but
    # checkpoints such as gemma-4-E2B still ship them (Gemma4TextModel.__init__).
    shared = cfg.get("num_kv_shared_layers") or 0
    m = _SHARED_KV_WEIGHT.search(name)
    return not (shared and m and int(m.group(1)) >= cfg["num_hidden_layers"] - shared)
