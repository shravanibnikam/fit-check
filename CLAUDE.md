# FitCheck

FitCheck predicts whether a Hugging Face LLM fits on a given GPU setup for training or serving, shows where the memory goes, and generates a working DeepSpeed config or vLLM command. It is a public tool people should be able to use, and also my vehicle for learning PyTorch internals, DeepSpeed, vLLM and Helion properly. I am an MS CS student targeting ML engineering roles.

## How to work with me

- Work one phase at a time. At the start of each phase, give me a short plan (files, approach, open questions) and wait for my approval before writing code.
- I am building this to learn. When code depends on non-obvious PyTorch, DeepSpeed or vLLM behavior, explain it in a brief comment or in `docs/`, and cite the source (paper, docs page, or file in the upstream repo).
- Prefer plain, readable code over abstraction. No frameworks or dependencies unless they earn their place.
- Never invent measured numbers. Calibration constants come only from results files I produce on real hardware. If a phase needs a measurement, write the script, tell me exactly how to run it, and stop.
- Small commits with clear messages. Run tests and lint before every commit.
- Docs: simple, direct wording, no emojis, few subsections.
- If a model uses something FitCheck does not model yet (for example MLA attention or sliding-window layers), the output must say so plainly instead of printing a confident number.
- Before planning any Kaggle run, confirm that the PyTorch, DeepSpeed, vLLM and Helion versions you plan to use support T4 (compute capability 7.5). If the latest release does not, find the newest one that does and pin it.
- At the end of each phase, add a few sentences to `docs/learning-log.md` on what the phase taught and what I should now be able to explain in an interview.

## Hard constraints

- Zero cost. No paid services and nothing that requires a credit card. Allowed: GitHub (repo, Actions, Pages), Hugging Face Hub and free CPU Spaces, Kaggle and Colab free GPUs, PyPI.
- My machines are macOS and Arch Linux with no NVIDIA GPU. All GPU work runs in Kaggle notebooks (2x T4). T4 has no native bf16, so every estimate must take the training dtype as a parameter (fp16 for T4 calibration, bf16 default otherwise).
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

6. Helion kernel. Implement a fused linear plus cross-entropy kernel in Helion that avoids materializing full logits. Test correctness against PyTorch (forward and backward, tolerance appropriate to dtype) and measure peak memory and speed on Kaggle. Feed the measured saving into the estimator as a "fused loss" option. First check that Helion runs on T4; if it doesn't, use Triton and document why. Record the Helion backend, the autotuned config and the autotuning time in every results file. Check whether Helion's CuteDSL backend supports T4 before trying it; if it does not, use the default backend and note why in `docs/`. Optional stretch, only after the GPU kernel passes its tests: a short spike on whether the same kernel runs on a free Kaggle TPU through Helion's TPU backend, reported to me before anything is built.

7. Web app. A static site on GitHub Pages that runs the `fitcheck` package in the browser via Pyodide, fetching headers from the Hub with range requests (verify CORS early with a spike). Inputs: model id, GPU count and memory, task (full fine-tune, LoRA, serving), sequence length, batch size. Output: fits or not, a stacked memory breakdown, and the generated config to copy. Propose the layout and component structure; I will own the visual design.

8. MoE models. Detect mixture-of-experts models from config.json (field names differ across Mixtral, Qwen MoE, DeepSeek and OLMoE) and confirm by classifying expert tensors in the safetensors header, so total and active parameter counts are exact. Report both. Training: every expert stays resident, so weights, gradients and optimizer state scale with total parameters, while activations follow the active ones. Add expert parallel degree and context (sequence) parallel degree as inputs and model how each divides memory, based on DeepSpeed's current MoE and sequence parallel source. Serving: weights use total parameters, and KV cache depends only on the attention layers. Calibrate on the smallest open MoE model that fits 2x T4, checking with FitCheck before choosing; `allenai/OLMoE-1B-7B-0924` is the first candidate. Motivation: the DeepSpeed session "Fitting Large MoE Fine-Tuning onto Few Nodes" (PyTorch Conference 2026) showed that HBM capacity, not compute, limits fine-tuning of large MoE models.

9. KV cache tiers and disaggregated serving. Extend the serving estimate with a CPU tier for vLLM's KV cache offloading: given host RAM, report how many tokens and concurrent sequences fit on GPU and on CPU, and emit the matching vLLM flags. Also report KV cache bytes per request for a given prompt length, which is what a prefill/decode disaggregated setup moves between instances. Take flag names and how vLLM reports offloaded capacity from the current vLLM docs and source, not from memory. Calibrate the CPU tier on Kaggle against what vLLM itself reports. Motivation: the talks "Native Tiered KV Cache Offloading in vLLM" and "State-of-the-Art KV Transfer for Disaggregated LLM Serving in vLLM" (PyTorch Conference 2026).

Phases 8 and 9 come after the web app so a usable v1 ships first. The web app runs the same package, so these features reach it once they land.

## Definition of done for each phase

Tests pass in CI, the README reflects what the tool can do now, `docs/` explains any new formula, and the phase is summarized in a short commit series I can review.
