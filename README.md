# FitCheck

FitCheck estimates how much GPU memory fine-tuning a Hugging Face LLM takes on a given setup, and shows where it goes. It reads only `config.json` and the safetensors headers, a few KB per file fetched with HTTP range requests. It never downloads the weights.

## What it does today

It estimates the training peak per GPU:

- Model states (weights, gradients and Adam states) for DeepSpeed ZeRO stages 0-3, with optimizer and parameter offload, for full fine-tuning or LoRA on the q, k, v and o projections, in bf16 (the default), fp16 or fp32.
- Activations for a given micro-batch and sequence length, with SDPA or eager attention and with or without activation checkpointing, including the logits and loss, which are often the largest single allocation.
- The peak as the model states plus the larger of the activations and the fp32 gradient copy made at the optimizer step.

Activation formulas cover Llama 3 and other Llama-architecture models (also Mistral and Qwen2.5), Gemma 4 (E2B, E4B, 12B, 31B) and Qwen3.5 (0.8B to 27B, and Qwen3.8-27B). Other architectures, including mixture-of-experts models, get model states only. The formulas count the tensors PyTorch keeps for backward, and `scripts/saved_tensors.py` checks them against autograd byte for byte. They are not yet calibrated against GPU runs, so allocator overhead and short-lived buffers are missing. Calibration, DeepSpeed config generation and vLLM serving estimates are planned.

## Usage

From a clone, with [uv](https://docs.astral.sh/uv/):

```
$ uv run fitcheck Qwen/Qwen2.5-0.5B --gpus 2 --gpu-mem 15 --zero 2 --dtype fp16
Qwen/Qwen2.5-0.5B: 494.0M params, 24 layers, hidden 896, 14 heads (2 KV) of 64
full fine-tune (494.0M trainable), fp16, ZeRO-2 on 2 GPU(s)
micro-batch 1 x 2048 tokens, sdpa attention, no checkpointing

Per GPU               GiB
  weights            0.92
  gradients          0.46
  optimizer          2.76
  activations        2.63
  logits and loss    4.07
  overhead           0.00
  total             10.84

Fits in 15 GiB: yes
Not yet calibrated against measured runs.
```

Options: `--gpus`, `--gpu-mem` (GiB per GPU), `--zero {0,1,2,3}`, `--offload-optimizer`, `--offload-param`, `--lora-rank R`, `--dtype {bf16,fp16,fp32}`, `--seq-len` (default 2048), `--micro-batch` (default 1), `--attn {sdpa,eager}`, `--checkpointing`, `--revision`. The model can be a Hub id or a local folder. For gated models such as Llama, accept the license on the Hub and set `HF_TOKEN`.

The formulas, their sources and the known gaps are in [docs/memory-model.md](docs/memory-model.md).

## Development

The `fitcheck` package has no runtime dependencies, so it can run in the browser via Pyodide later.

```
uv sync
uv run pytest               # offline tests
uv run pytest -m network    # only the tests that hit the Hugging Face Hub
uv run ruff check && uv run ruff format --check
uv run --group check python scripts/saved_tensors.py   # needs torch, transformers, peft
```

The `check` group installs torch, transformers and peft for the saved-tensor check only. CI doesn't install it.
