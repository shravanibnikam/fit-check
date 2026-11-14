# FitCheck

FitCheck estimates how much GPU memory fine-tuning a Hugging Face LLM takes on a given setup, shows where it goes, and writes a DeepSpeed config for the cheapest setup that fits. It reads only `config.json` and the safetensors headers, a few KB per file fetched with HTTP range requests. It never downloads the weights.

## What it does today

It estimates the training peak per GPU:

- Model states (weights, gradients and Adam states) for DeepSpeed ZeRO stages 0-3, with optimizer and parameter offload, for full fine-tuning or LoRA on the q, k, v and o projections, in bf16 (the default), fp16 or fp32.
- Activations for a given micro-batch and sequence length, with SDPA or eager attention and with or without activation checkpointing, including the logits and loss, which are often the largest single allocation.
- The peak as the model states plus the larger of the activations and the fp32 gradient copy made at the optimizer step.
- With `fitcheck recommend`, every ZeRO stage, offload and checkpointing setting for your GPUs, ranked from cheapest to most expensive, and a `ds_config.json` for the cheapest one that fits.

Activation formulas cover Llama 3 and other Llama-architecture models (also Mistral and Qwen2, including their sliding-window layers), Gemma 4 (E2B, E4B, 12B, 31B) and Qwen3.5 (0.8B to 27B, and Qwen3.8-27B). Other architectures, including mixture-of-experts models, get model states only.

When a model uses something FitCheck doesn't model yet (mixture-of-experts layers, MLA attention, a quantized checkpoint, or an architecture without an activation formula), the output says so, and the verdict is "unknown" unless the part it counts already exceeds the GPU. With LoRA, it also lists the layers that lack any of q_proj, k_proj, v_proj or o_proj. The formulas count the tensors PyTorch keeps for backward, and `scripts/saved_tensors.py` checks them against autograd byte for byte. The estimate is of the peak memory PyTorch allocates, plus the CUDA context. The target is within 2% of measured `max_memory_allocated`; reserved memory is reported but not used, because DeepSpeed leaves freed blocks in PyTorch's cache and PyTorch frees them before it raises an OOM. Measured on Kaggle's 2x T4 (Qwen2.5-0.5B and 1.5B, DeepSpeed ZeRO 0-3, offload, LoRA, checkpointing, eager attention), the prediction is within 2% on 33 of 35 runs, median 0.8% and worst 3.4% (both misses at sequence length 512), and the fits verdict was right on all 40 runs that finished or ran out of GPU memory. The 1.5B runs with optimizer offload could not be measured: they need more than Kaggle's 31 GB of host RAM. The runs are in `calibrate/results/`. vLLM serving estimates are planned.

## Usage

From a clone, with [uv](https://docs.astral.sh/uv/):

```
$ uv run fitcheck Qwen/Qwen2.5-0.5B --gpus 2 --gpu-mem 15 --zero 2 --dtype fp16
Qwen/Qwen2.5-0.5B: 494.0M params, 24 layers, hidden 896, 14 heads (2 KV) of 64
full fine-tune (494.0M trainable), fp16, ZeRO-2 on 2 GPU(s)
micro-batch 1 x 2048 tokens, sdpa attention, no checkpointing, gradient accumulation 1

Per GPU, at the start of backward   GiB
  weights              0.92
  gradients            0.00
  optimizer            2.76
  DeepSpeed buffers    0.00
  activations          2.63
  logits and loss      4.07
  overhead             0.24
  total               10.62

Fits in 15 GiB: yes
Calibrated on Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B runs on Tesla T4; allocated memory within 3% on 35 measured runs, the allocator may reserve more. Other models and GPUs are extrapolated.
```

Options: `--gpus`, `--gpu-mem` (GiB per GPU), `--zero {0,1,2,3}`, `--offload-optimizer`, `--offload-param`, `--lora-rank R`, `--dtype {bf16,fp16,fp32}`, `--seq-len` (default 2048), `--micro-batch` (default 1), `--grad-accum` (default 1), `--attn {sdpa,eager}`, `--checkpointing`, `--loss-tokens N` (loss on the last N tokens only, via `logits_to_keep=N + 1`; default is HF's loss over every token), `--revision`. The model can be a Hub id or a local folder. For gated models such as Llama, accept the license on the Hub and set `HF_TOKEN`.

To find a setup, `fitcheck recommend` takes the same model, task and batch options (but not `--zero`, the offload flags or `--checkpointing`, which it searches), plus `--gpu-mem` (required), `--host-mem` (GiB of host RAM for all GPUs together), `--lr` (default 1e-5) and `--out` (default `ds_config.json`):

```
$ uv run fitcheck recommend Qwen/Qwen2.5-0.5B --gpus 2 --gpu-mem 15 --dtype fp16 --seq-len 4096
Qwen/Qwen2.5-0.5B: 494.0M params, 24 layers, hidden 896, 14 heads (2 KV) of 64
full fine-tune (494.0M trainable), fp16 on 2 GPU(s) of 15 GiB, micro-batch 1 x 4096 tokens, sdpa attention, gradient accumulation 1

  setup, cheapest first                                GPU GiB  host GiB  fits
  ZeRO-0                                                 20.08         -  no (GPU memory)
  ZeRO-1                                                 17.32         -  no (GPU memory)
  ZeRO-2                                                 17.32         -  no (GPU memory)
  ZeRO-3                                                 19.17         -  no (GPU memory)
  ZeRO-0, checkpointing                                  14.99         -  no (GPU memory)
* ZeRO-1, checkpointing                                  12.22         -  yes
  ZeRO-2, checkpointing                                  12.22         -  yes
  ZeRO-3, checkpointing                                  14.08         -  yes
  ZeRO-1, offload optimizer                              14.81      7.36  no (GPU memory)
  ZeRO-2, offload optimizer                              14.81      7.36  no (GPU memory)
  ZeRO-3, offload optimizer                              15.95      7.36  no (GPU memory)
  ZeRO-1, offload optimizer, checkpointing                9.72      7.36  yes (GPU; host RAM unchecked)
  ZeRO-2, offload optimizer, checkpointing                9.72      7.36  yes (GPU; host RAM unchecked)
  ZeRO-3, offload optimizer, checkpointing               10.86      7.36  yes (GPU; host RAM unchecked)
  ZeRO-3, offload optimizer and params                   15.49      8.28  no (GPU memory)
  ZeRO-3, offload optimizer and params, checkpointing    10.40      8.28  yes (GPU; host RAM unchecked)

A setup fits when its peak allocated memory plus 3.4%, the worst calibration error, and the CUDA context fit in 15 GiB. The order is a rule of thumb: offload, then checkpointing, then a higher ZeRO stage.
Host GiB is the offloaded state only. It leaves out the model as loaded on the CPU and the Python and CUDA runtimes, so it is a lower bound: a setup over --host-mem is ruled out, one under it is not checked. The measured offload runs needed about twice the estimate (docs/memory-model.md).
Calibrated on Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B runs on Tesla T4; allocated memory within 3% on 35 measured runs, the allocator may reserve more. Other models and GPUs are extrapolated.

Best: ZeRO-1, checkpointing. Wrote ds_config.json.
Also set in the training script, which the config can't hold:
  from_pretrained(..., attn_implementation="sdpa")
  model.gradient_checkpointing_enable(), then model.train()
The estimate holds for the calibrated versions: torch 2.14.1+cu126, deepspeed 0.19.9.
```

The order is offload first, then checkpointing, then a higher ZeRO stage. It is a rule of thumb, not measured. A setup must fit with the calibration's worst error to spare, which is why ZeRO-0 with checkpointing (14.99 GiB) doesn't count above. Host RAM is a lower bound, so `--host-mem` can only rule a setup out. When a model has something FitCheck doesn't model, it writes no config. The config is for `deepspeed.initialize` or the `deepspeed` launcher. The attention implementation, checkpointing and LoRA go in the training script, and FitCheck prints what to set.

The formulas, their sources and the known gaps are in [docs/memory-model.md](docs/memory-model.md).

## Development

The `fitcheck` package has no runtime dependencies, so it can run in the browser via Pyodide later.

```
uv sync
uv run pytest               # offline tests
uv run pytest -m network    # only the tests that hit the Hugging Face Hub
uv run ruff check && uv run ruff format --check
uv run --group check python scripts/saved_tensors.py   # needs torch, transformers, peft
uv run --group deepspeed pytest   # also loads generated configs with DeepSpeed's parser
```

Calibration runs on Kaggle's 2x T4, pushed with the Kaggle CLI (a separate tool: `uv tool install kaggle`). `calibrate/kaggle/kernel-metadata.json` asks for a private notebook with GPU T4 x2 and internet:

```
kaggle kernels push -p calibrate/kaggle
kaggle kernels status shravanibnikam/fitcheck-calibration
kaggle kernels output shravanibnikam/fitcheck-calibration -p calibrate/results
uv run fitcheck calibrate calibrate/results/results.jsonl
```

A smoke run pushes a copy of the notebook with `SMOKE = True`: the environment checks, the DeepSpeed op build, both saved-tensor checks and 2 sweep runs.

It prints predicted against measured allocated memory for every run, with reserved memory for information, and writes `src/fitcheck/calibration.json`. It won't fit until the CUDA saved-tensor check in `calibrate/results/saved_tensors_cuda.json` matches.

The `check` group installs torch, transformers and peft for the saved-tensor check only. CI doesn't install it. The `deepspeed` group installs DeepSpeed 0.19.9 and CPU-only torch, and a separate CI job runs the tests with it.
