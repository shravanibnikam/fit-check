# FitCheck

FitCheck estimates how much GPU memory fine-tuning a Hugging Face LLM takes on a given setup, and shows where it goes. It reads only `config.json` and the safetensors headers, a few KB per file fetched with HTTP range requests. It never downloads the weights.

## What it does today

It counts model states per GPU: weights, gradients and Adam optimizer states. It covers:

- DeepSpeed ZeRO stages 0-3, with optimizer offload and parameter offload
- full fine-tuning, or LoRA on the q, k, v and o projections
- bf16 (the default), fp16 or fp32 training

Activations are not counted yet, so treat every result as a lower bound. Activation memory, calibration against measured runs, DeepSpeed config generation and vLLM serving estimates are planned.

## Usage

From a clone, with [uv](https://docs.astral.sh/uv/):

```
$ uv run fitcheck Qwen/Qwen2.5-0.5B --gpus 2 --gpu-mem 15 --zero 2 --dtype fp16
Qwen/Qwen2.5-0.5B: 494.0M params, 24 layers, hidden 896, 14 heads (2 KV) of 64
full fine-tune (494.0M trainable), fp16, ZeRO-2 on 2 GPU(s)

Per GPU            GiB
  weights         0.92
  gradients       0.46
  optimizer       2.76
  overhead        0.00
  total           4.14

Fits in 15 GiB: yes
Activations are not counted yet, so treat this as a lower bound.
```

Options: `--gpus`, `--gpu-mem` (GiB per GPU), `--zero {0,1,2,3}`, `--offload-optimizer`, `--offload-param`, `--lora-rank R`, `--dtype {bf16,fp16,fp32}`, `--revision`. The model can be a Hub id or a local folder. For gated models such as Llama, accept the license on the Hub and set `HF_TOKEN`.

The formulas, their sources and the known gaps are in [docs/memory-model.md](docs/memory-model.md).

## Development

The `fitcheck` package has no runtime dependencies, so it can run in the browser via Pyodide later.

```
uv sync
uv run pytest               # offline tests
uv run pytest -m network    # only the tests that hit the Hugging Face Hub
uv run ruff check && uv run ruff format --check
```
