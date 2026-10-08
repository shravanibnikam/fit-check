"""The architecture fields the estimator needs, read from config.json.

Uses the standard Hugging Face field names shared by LlamaConfig, Qwen2Config,
MistralConfig and most decoder-only models.
"""

from dataclasses import dataclass

from fitcheck.safetensors_header import count_params


@dataclass(frozen=True)
class Model:
    name: str
    num_params: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def from_config(name: str, config: dict, tensors: dict[str, dict]) -> Model:
    # Multimodal checkpoints (Gemma 3, Llama 4, Qwen2.5-VL) nest the language
    # model's fields under text_config.
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
    return Model(
        name=name,
        num_params=count_params(tensors),
        hidden_size=hidden,
        num_layers=need("num_hidden_layers"),
        num_heads=heads,
        # Without grouped-query attention every query head has its own K/V head.
        num_kv_heads=cfg.get("num_key_value_heads") or heads,
        # Usually hidden / heads, but Gemma and Qwen3 set it explicitly to
        # something else (e.g. Gemma 2 2B: hidden 2304, 8 heads, head_dim 256).
        head_dim=cfg.get("head_dim") or hidden // heads,
    )
