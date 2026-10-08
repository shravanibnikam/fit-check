# Memory model

This is what FitCheck counts, where each formula comes from, and what it leaves out. Training memory per GPU is the model states (weights, gradients, optimizer states) plus, at the peak, either the activations or the optimizer step's fp32 gradient copy, whichever is larger.

Notation: Ψ is the parameter count, Ψt the trainable parameter count (Ψt = Ψ for full fine-tuning), N the number of GPUs (data-parallel ranks), and b the bytes per element of the training dtype (2 for fp16 and bf16, 4 for fp32).

## Parameter count

Ψ is the sum of `prod(shape)` over every tensor in the checkpoint's safetensors headers. The headers are fetched with HTTP range requests, so no weights are downloaded. The count is exact for unquantized checkpoints. When the embedding and output head are tied, the matrix is stored once and counted once.

Worked example, Qwen2.5-0.5B (hidden 896, 24 layers, 14 query heads and 2 KV heads of size 64, MLP width 4864, vocab 151936, biases on q/k/v, tied embeddings):

| Tensor | Parameters |
|---|---|
| Token embedding, 151936 x 896 | 136,134,656 |
| q_proj, 896 x 896 + bias | 803,712 |
| k_proj and v_proj, 896 x 128 + bias, each | 114,816 |
| o_proj, 896 x 896 | 802,816 |
| gate, up and down projections, 3 x 896 x 4864 | 13,074,432 |
| Two RMSNorms, 2 x 896 | 1,792 |
| One layer, total | 14,912,384 |
| 24 layers | 357,897,216 |
| Final norm | 896 |
| Model | 494,032,768 |

The network smoke test checks this number against the real Hub headers.

Some checkpoints ship weights that transformers drops when loading, and FitCheck drops them too: Qwen3.5's multi-token-prediction head (`mtp.*`, in `_keys_to_ignore_on_load_unexpected`), and the K/V weights of Gemma 4 layers that reuse an earlier layer's K/V (gemma-4-E2B stores them anyway).

## Bytes per parameter

Mixed-precision Adam keeps, per parameter, a weight and a gradient in the training dtype, plus an fp32 master copy of the weight and Adam's two fp32 moments. That is 2 + 2 + 12 = 16 bytes (ZeRO paper, section 3.1, where K = 12).

DeepSpeed casts every parameter to fp16 or bf16 when either is enabled, LoRA adapters included (`DeepSpeedEngine._cast_module_mixed_precision` in `deepspeed/runtime/engine.py`). So one dtype covers the whole model. fp16 and bf16 cost the same bytes. The dtype still matters because the T4 has no native bf16, so calibration runs there use fp16.

fp32 training without ZeRO steps the parameters directly, so the optimizer only adds the two moments: 4 + 4 + 8 = 16 bytes. With ZeRO, DeepSpeed clones an fp32 master partition even when the model is already fp32 (`single_partition_of_fp32_groups` in `deepspeed/runtime/zero/stage_1_and_2.py`), so the optimizer term is back to 12 bytes.

## ZeRO stages

Each ZeRO stage partitions one more kind of state across the N ranks (ZeRO paper, section 5). Per GPU, with K = 12 (or 8 for fp32 at stage 0):

| Stage | Weights | Gradients | Optimizer |
|---|---|---|---|
| 0 | bΨ | bΨt | KΨt |
| 1 | bΨ | bΨt | KΨt / N |
| 2 | bΨ | bΨt / N | KΨt / N |
| 3 | bΨ / N | bΨt / N | KΨt / N |

With Ψ = 7.5B, N = 64 and mixed precision this gives 120 GB, 31.4 GB, 16.6 GB and 1.88 GB, matching Figure 1 of the paper. The tests check these values.

## Offload

Optimizer offload is valid for stages 1-3, and parameter offload only for stage 3 (`DeepSpeedZeroConfig` in `deepspeed/runtime/zero/config.py`).

- Optimizer offload moves the master copy and the moments to CPU RAM and runs Adam there (ZeRO-Offload, Ren et al. 2021). At stages 2 and 3, each rank's gradient partition is also copied straight into an fp32 CPU buffer, so it leaves the GPU too (`copy_grads_in_partition` in `stage_1_and_2.py`). Stage 1 still holds full gradients on the GPU. CPU RAM needed: 16 bytes per trainable parameter (master, two moments, fp32 gradient), summed over all ranks.
- Parameter offload keeps the weights in CPU RAM between uses, which adds bΨ to CPU RAM and takes the weights term on the GPU to roughly zero.

## LoRA

LoRA freezes each targeted Linear(in, out) and trains A (r x in) and B (out x r) beside it, adding r(in + out) parameters (Hu et al. 2021). FitCheck targets the modules named q_proj, k_proj, v_proj and o_proj, as PEFT would with `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`, and reads their shapes from the safetensors headers. For a Llama-style layer with hidden size h, query width q = heads x head_dim and KV width kv = kv_heads x head_dim, that is

    Ψt = r * layers * [(h + q) + 2(h + kv) + (q + h)]

Reading shapes from the checkpoint also handles models where that formula is wrong: Qwen3.5's q_proj is twice as wide (it also produces an output gate), only its full-attention layers have these modules, and Gemma 4 layers that reuse K/V have no k_proj or v_proj.

The frozen base still costs bΨ in weights, partitioned only at stage 3. Gradients and optimizer states exist only for the adapters.

## Activations

During the forward pass, autograd keeps some tensors for the backward pass. Those saved tensors are the activations. They are all alive at the start of backward, which is where training memory peaks. FitCheck counts them one op at a time in the Hugging Face modeling code (transformers 5.19), using two rules from PyTorch autograd (`tools/autograd/derivatives.yaml`):

- An op saves only what its required gradients need. `x * y` saves y only if x needs a gradient, and x only if y does. A Linear layer saves its input only if its weight trains. That is why RoPE keeps almost nothing (cos and sin need no gradient), and why LoRA, with frozen weights, keeps less.
- Tensors are counted once per storage. A view or a reshape that doesn't copy shares memory with its source.

The setup is the one DeepSpeed training uses: every parameter cast to the training dtype, no dropout, sequences without padding, the loss computed by the model from `labels`, and the returned outputs kept alive until backward. Per token, with p = b bytes and F = 4 bytes for the fp32 upcasts in the modeling code, the building blocks are:

| Op | Saved per token | When |
|---|---|---|
| LlamaRMSNorm over width w | F·w (fp32 x) + F (rsqrt) + p·w (x before the weight) | last term only if the weight trains |
| Gemma4RMSNorm over w | F·w (fp32 x) + 2F + F·w (normalized x) | last term only if the weight trains |
| Qwen3_5RMSNorm over w | F·w + F + F·w | last term only if the weight trains |
| Linear | its input | only if its weight trains, or LoRA's A reads it |
| SDPA (flash or memory-efficient) | p·q + 2p·kv + p·q (output) + F·heads (logsumexp) | K/V at full q width if they are copied per head |
| Eager attention | 4p·q + (F + p)·heads·s | the softmax and its copy grow with sequence length s |
| SwiGLU or GeGLU MLP, width I | 4p·I | 3p·I when down_proj is frozen |

SDPA has no s² term, because the fused kernel recomputes attention scores in backward. Eager attention keeps an fp32 softmax and a dtype copy of it, heads x s values per token per layer. On a T4 or P100, SDPA uses the memory-efficient kernel, because FlashAttention needs sm80 or newer (`check_flash_attention_hardware_support` in PyTorch's `sdp_utils.cpp`). Both kernels keep the same tensors.

Llama, Mistral and Qwen2 add these up per layer: two norms, the projection inputs, SDPA and the MLP. For Qwen2.5-0.5B in bf16 that is 57,408 bytes per token per layer: 2.63 GiB for one 2048-token sequence over 24 layers.

### The output head

The logits are often the largest allocation. The model returns bf16 logits (p·V per token, V the vocabulary size), and `ForCausalLMLoss` upcasts them to fp32 and keeps the log-softmax output (F·V). At the start of backward, log-softmax's backward also holds the incoming gradient and its result, two more fp32 tensors of the same size. So the head peaks at (p + 3F)·V per token: 14 bytes per vocabulary entry in bf16. Gemma's final logit softcapping keeps one more p·V (the tanh output). For Qwen2.5-0.5B (V = 151,936) that is 4.07 GiB for 2048 tokens, more than all 24 layers together. Phase 6's fused linear plus cross-entropy kernel targets exactly this.

### Gemma 4

- Four norms per layer, plus norms on q, k and v. Every Gemma norm keeps two fp32 copies of its input when its weight trains, so norms cost more than in Llama.
- Sliding-window layers need an explicit mask once the sequence reaches the window (`_ignore_causal_mask_sdpa` in `masking_utils.py`). SDPA converts that boolean mask to a float bias on every call and keeps it: p·s² per sequence per sliding layer.
- On CUDA, HF lets SDPA do grouped-query attention natively only without a mask and with head_dim at most 256 (`use_gqa_in_sdpa` in `integrations/sdpa_attention.py`). Otherwise `repeat_kv` copies K/V to every query head first. That happens on every sliding layer and on the global layers, whose head_dim is 512. With a single KV head (E2B, E4B) the copy is a stride-0 view and costs nothing.
- The last `num_kv_shared_layers` layers reuse an earlier layer's K/V. They have no K/V projections or norms, and in E2B their MLP is twice as wide.
- E2B and E4B also have per-layer embeddings: a small gate, GELU, product and projection in every layer, plus one input tensor of layers x 256 values per token that all layers share.

### Qwen3.5

Three of every four layers use Gated DeltaNet, a linear-attention layer, instead of softmax attention. Without the `flash-linear-attention` package, transformers runs a PyTorch fallback (`torch_chunk_gated_delta_rule`). It splits the sequence into 64-token chunks and works in fp32: per head it keeps five 64 x 64 matrices per chunk, several fp32 copies of q, k and v, and the recurrent state (128 x 128 per head) after every chunk. For Qwen3.5-0.8B that is 238,209 bytes per token per DeltaNet layer, against 88,149 for a full-attention layer. FitCheck counts this fallback, because it is what runs when the package isn't installed. With the package installed, the true number is lower; measuring it is Phase 3 work. The full-attention layers add an output gate: sigmoid(gate) and the gated product, 2p·q per token.

### Checkpointing and the peak

With `model.gradient_checkpointing_enable()`, HF checkpoints each decoder layer (`GradientCheckpointingLayer`, `use_reentrant=False`). The forward pass keeps only each layer's input, p·h per token. Backward recomputes one layer at a time. The output head isn't checkpointed, and it runs first in backward. So the activation peak is the layer inputs plus the larger of the head's buffers and one recomputed layer.

The training peak is the model states plus the larger of two moments:

- the start of backward, holding all activations and the head's buffers;
- the optimizer step, after activations are freed, where DeepSpeed copies the gradients to fp32 for Adam. That copy is all trainable gradients at stage 0 (`FP16_Optimizer.step` in `deepspeed/runtime/fp16/fused_optimizer.py`), each rank's partition at stages 1-2 (`stage_1_and_2.py`), and one sub-group at a time at stage 3 (`_prepare_fp32_grad_for_sub_group` in `stage3.py`, at most `sub_group_size`, default 1e9 elements). It is 4Ψt bytes at stage 0, on the CPU with optimizer offload, and none in fp32 training.

### How the formulas are checked

`scripts/saved_tensors.py` builds small random-weight models that keep each family's structure: grouped-query attention, sliding windows, shared K/V, per-layer embeddings, K = V on global layers, global head_dim over 256, and DeltaNet layers. It runs a training forward pass on CPU, records every tensor autograd saves, and compares bytes per layer with the formulas. All 24 cases match exactly: four configs, full fine-tuning and LoRA, and SDPA bf16, eager bf16 and SDPA fp32.

    uv run --group check python scripts/saved_tensors.py -v

Two things differ between CPU and CUDA, and the script handles both. HF's native-GQA rule depends on the device, so the script pins the CUDA rule. And PyTorch's CPU attention kernel returns its output in the query's memory layout, which costs Qwen3.5 one extra copy that CUDA kernels don't make, so the script adds that copy to the prediction. These are exact counts of saved tensors, not GPU measurements. They don't cover the allocator, CUDA workspaces or other short-lived buffers.

## Not modeled yet

These are known gaps. Each should show up as error in the Phase 3 calibration runs or get its own term:

- Short-lived buffers other than the loss backward: matmul outputs before they are consumed, gradients of activations during backward, attention kernel workspaces.
- Communication buckets (`reduce_bucket_size`, `allgather_bucket_size`), and the parameters stage 3 gathers and prefetches for the layer it is running.
- Whether CUDA's SDPA output really needs no copy before `o_proj`, as assumed above. Phase 3 measures it.
- Qwen3.5 with `flash-linear-attention` installed, and kernels from the `kernels` package.
- Padding, which brings back masks and, on CUDA, K/V copies.
- The CUDA context, and the gap between memory the allocator has reserved and memory actually allocated. These are the `CALIBRATE` constants in `estimate.py`, left at neutral values until measured.
- Mixture-of-experts layers (Gemma 4 26B-A4B, Qwen3.5 MoE, Llama 4) and other architectures: FitCheck reports their model states only.
- Quantized checkpoints (GPTQ, AWQ), whose packed tensors undercount Ψ.

## Sources

- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2020. arXiv:1910.02054.
- Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training", 2021. arXiv:2101.06840.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. arXiv:2106.09685.
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", 2022. arXiv:2205.05198. Its per-layer counting method, applied here to the HF modeling code.
- DeepSpeed source, github.com/deepspeedai/DeepSpeed, read at v0.19.8.
- transformers source, github.com/huggingface/transformers, read at 5.19: `models/llama`, `models/gemma4`, `models/qwen3_5`, `integrations/sdpa_attention.py`, `masking_utils.py`, `loss/loss_utils.py`.
- PyTorch source, github.com/pytorch/pytorch: `tools/autograd/derivatives.yaml`, `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`.
- safetensors format, github.com/huggingface/safetensors.
