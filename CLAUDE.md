# FitCheck

FitCheck predicts whether a Hugging Face LLM fits on a given GPU setup for training or serving, shows where the memory goes, and generates a working DeepSpeed config or vLLM command. It is a public tool people should be able to use, and also my vehicle for learning PyTorch internals, DeepSpeed, vLLM and Helion properly. I am an MS CS student targeting ML engineering roles.

## How to work with me

- Work one phase at a time. At the start of each phase, give me a short plan (files, approach, open questions) and wait for my approval before writing code.
- I am building this to learn. When code depends on non-obvious PyTorch, DeepSpeed or vLLM behavior, explain it in a brief comment or in `docs/`, and cite the source (paper, docs page, or file in the upstream repo).
- Prefer plain, readable code over abstraction. No frameworks or dependencies unless they earn their place.
- Never invent measured numbers. Calibration constants come only from results files I produce on real hardware. If a phase needs a measurement, write the script, tell me exactly how to run it, and stop.
- Small commits with clear messages. Run tests and lint before every commit.
- Docs: simple, direct wording, no emojis, few subsections.

## Hard constraints

- Zero cost. No paid services and nothing that requires a credit card. Allowed: GitHub (repo, Actions, Pages), Hugging Face Hub and free CPU Spaces, Kaggle and Colab free GPUs, PyPI.
- My machines are macOS and Arch Linux with no NVIDIA GPU. All GPU work runs in Kaggle notebooks (2x T4 or P100). T4 has no native bf16, so every estimate must take the training dtype as a parameter (fp16 for T4 calibration, bf16 default otherwise).
- The core `fitcheck` package must stay pure Python with zero runtime dependencies, so the same code can run in the browser via Pyodide. Network access stays isolated in `hub.py`.
- Never download model weights to estimate. Read safetensors headers with HTTP range requests and `config.json` only.

## Starting point

Phase 1 already exists in this repo: `safetensors_header.py` (header parsing), `hub.py` (range-request loading, sharded models, local folders), `model.py` (architecture fields from config.json), `estimate.py` (weights, grads and Adam states across ZeRO 0-3, optimizer and param offload, LoRA on q/k/v/o), `cli.py`, and tests. Read it first, keep its structure, and extend it rather than rewriting it. Constants marked `CALIBRATE` are placeholders.

## Phases

1. Harden the foundation. Switch tests to pytest, add ruff, add a GitHub Actions workflow (lint and tests on Python 3.10-3.12). Add a network smoke test, marked and skipped by default, that loads `Qwen/Qwen2.5-0.5B` from the Hub and checks the parameter count. Make the dtype configurable.

2. Activations. Model per-GPU activation memory for decoder-only transformers as a function of micro-batch size, sequence length, attention implementation (eager vs SDPA/flash), and activation checkpointing. Account for the logits and loss, which are often the largest single allocation at long context and large vocab. Write the derivation in `docs/memory-model.md`.

3. Calibration harness. Write `calibrate/run_one.py`, a minimal training loop in plain PyTorch with `deepspeed.initialize` (no HF Trainer), launched with the DeepSpeed launcher, and a Kaggle notebook that sweeps model size (0.5B, 1.5B), ZeRO stage, offload, LoRA, sequence length and checkpointing. Each run appends one JSON line with the config and measured peak `max_memory_allocated` and `max_memory_reserved` per rank. Add `fitcheck calibrate results.jsonl`, which reports prediction error per config and fits the `CALIBRATE` constants into `calibration.json` with provenance (date, GPU, library versions). Target: within 10% of measured reserved memory.

4. Recommendations and config generation. Given a model, hardware and goal, rank valid setups by cost (least offload, highest stage needed) and emit a ready-to-use `ds_config.json` for the best one. Validate generated configs by loading them with DeepSpeed's own config parser in tests (DeepSpeed as a dev-only dependency).

5. Serving. Estimate vLLM memory: weights, KV cache per token from layers, KV heads, head dim and KV dtype, plus `gpu_memory_utilization` headroom. Output max context and max concurrent sequences, and emit a `vllm serve` command. Calibrate against vLLM's reported KV cache capacity on a Kaggle T4.

6. Helion kernel. Implement a fused linear plus cross-entropy kernel in Helion that avoids materializing full logits. Test correctness against PyTorch (forward and backward, tolerance appropriate to dtype) and measure peak memory and speed on Kaggle. Feed the measured saving into the estimator as a "fused loss" option. First check that Helion runs on T4; if it doesn't, use Triton and document why.

7. Web app. A static site on GitHub Pages that runs the `fitcheck` package in the browser via Pyodide, fetching headers from the Hub with range requests (verify CORS early with a spike). Inputs: model id, GPU count and memory, task (full fine-tune, LoRA, serving), sequence length, batch size. Output: fits or not, a stacked memory breakdown, and the generated config to copy. Propose the layout and component structure; I will own the visual design.

## Definition of done for each phase

Tests pass in CI, the README reflects what the tool can do now, `docs/` explains any new formula, and the phase is summarized in a short commit series I can review.
