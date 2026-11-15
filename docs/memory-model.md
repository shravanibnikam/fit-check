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

Some checkpoints ship weights that transformers drops when loading, and FitCheck drops them too: Qwen3.5's multi-token-prediction head and vision tower (`mtp.*` and `model.visual.*`, in `Qwen3_5ForCausalLM._keys_to_ignore_on_load_unexpected`, which is what `AutoModelForCausalLM` builds), and the K/V weights of Gemma 4 layers that reuse an earlier layer's K/V (gemma-4-E2B stores them anyway).

## Bytes per parameter

Mixed-precision Adam keeps, per parameter, a weight and a gradient in the training dtype, plus an fp32 master copy of the weight and Adam's two fp32 moments. That is 2 + 2 + 12 = 16 bytes (ZeRO paper, section 3.1, where K = 12).

DeepSpeed casts every parameter to fp16 or bf16 when either is enabled, LoRA adapters included (`DeepSpeedEngine._cast_module_mixed_precision` in `deepspeed/runtime/engine.py`). So one dtype covers the whole model. fp16 and bf16 cost the same bytes. The dtype still matters because the T4 has no native bf16, so calibration runs there use fp16.

fp32 training without ZeRO steps the parameters directly, so the optimizer only adds the two moments: 4 + 4 + 8 = 16 bytes. With ZeRO, DeepSpeed clones an fp32 master partition even when the model is already fp32 (`single_partition_of_fp32_groups` in `deepspeed/runtime/zero/stage_1_and_2.py`), so the optimizer term is back to 12 bytes.

## ZeRO stages

Each ZeRO stage partitions one more kind of state across the N ranks (ZeRO paper, section 5). Per GPU, with K = 12 (or 8 for fp32 at stage 0):

| Stage | Weights | Gradients | Optimizer |
|---|---|---|---|
| 0 | bΨ | bΨt | KΨt |
| 1 | bΨ | bΨt / N | KΨt / N |
| 2 | bΨ | bΨt / N | KΨt / N |
| 3 | bΨ / N | bΨt / N | KΨt / N |

With Ψ = 7.5B, N = 64 and mixed precision, stages 0, 2 and 3 give 120 GB, 16.6 GB and 1.88 GB, matching Figure 1 of the paper. The tests check these values.

The paper's stage 1 keeps full gradients (31.4 GB). DeepSpeed doesn't, when gradient accumulation is 1: at the end of backward ZeRO-1 copies the gradients into its reduce bucket, reduces them and drops them, keeping only its own partition (`reduce_gradients` and `reduce_independent_p_g_buckets_and_remove_grads` in `stage_1_and_2.py`). On the T4, ZeRO-1 measured exactly the same as ZeRO-2. With `--grad-accum` above 1, the gradients of the earlier micro-batches are alive when the last backward starts, and FitCheck adds them to that moment. ZeRO-0 and 1 reduce only at the accumulation boundary, so they keep full gradients (`process_gradients` and `reduce_gradients`). ZeRO-2 reduces after every micro-batch and keeps its own partition (`independent_gradient_partition_epilogue`). ZeRO-2 with optimizer offload adds each partition into a CPU buffer (`async_accumulate_grad_in_cpu_via_gpu`), so nothing stays on the GPU, and the CPU estimate grows by one gradient copy. ZeRO-3 already holds its partition. The gradients stay in the training dtype, because DeepSpeed's default `gradient_accumulation_dtype` is the model dtype. This comes from reading the DeepSpeed v0.19.9 source. The first measurement, Qwen2.5-0.5B at sequence length 2048 and accumulation 4 on 2x T4 (`calibrate/results/2026-11-14-recommend/`), showed ZeRO-1 peaking at 12.31 GiB allocated against 11.30 predicted. An allocator trace of the same run (`calibrate/results/2026-11-15-ga-trace/`, `run_one.py --trace`) showed the missing 0.93 GiB: ZeRO-1's reduce bucket. The first gradient hook of each backward allocates it (`setup_buckets`, called from `process_gradients` in `stage_1_and_2.py`). Only the epilogue frees it (`independent_gradient_partition_epilogue` and `_release_ipg_buffers`), and ZeRO-1 runs the epilogue only at the accumulation boundary, from `reduce_gradients`. So the bucket from the first micro-batch is still allocated when every later backward starts. ZeRO-2 runs the epilogue after every micro-batch (`engine.allreduce_gradients`), so it doesn't keep the bucket. With that term the run is predicted at 12.23 GiB, 0.7% under the measured peak.

ZeRO-0 with accumulation 4 ran out of memory in its first step. Its error message reports 11.83 GiB allocated and a 1.16 GiB request, 12.99 GiB in all. FitCheck's formula for that moment, without Adam's moments, which don't exist yet, gives 10.38 GiB. That leaves about 2.6 GiB unexplained, and no trace of ZeRO-0 exists. ZeRO-2 and 3 with accumulation have not been measured. So FitCheck models gradient accumulation at ZeRO-1 only, and lists it as not modeled at the other stages.

## Offload

Optimizer offload is valid for stages 1-3, and parameter offload only for stage 3 (`DeepSpeedZeroConfig` in `deepspeed/runtime/zero/config.py`).

- Optimizer offload moves the master copy and the moments to CPU RAM and runs Adam there (ZeRO-Offload, Ren et al. 2021). Each rank's gradient partition is also copied straight into an fp32 CPU buffer, so it leaves the GPU too (`copy_grads_in_partition` in `stage_1_and_2.py`). CPU RAM needed: 16 bytes per trainable parameter (master, two moments, fp32 gradient), summed over all ranks.
- Parameter offload keeps the weights in CPU RAM between uses, which adds bΨ to CPU RAM and takes the weights term on the GPU to roughly zero.

## LoRA

LoRA freezes each targeted Linear(in, out) and trains A (r x in) and B (out x r) beside it, adding r(in + out) parameters (Hu et al. 2021). FitCheck targets the modules named q_proj, k_proj, v_proj and o_proj, as PEFT would with `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`, and reads their shapes from the safetensors headers. For a Llama-style layer with hidden size h, query width q = heads x head_dim and KV width kv = kv_heads x head_dim, that is

    Ψt = r * layers * [(h + q) + 2(h + kv) + (q + h)]

Reading shapes from the checkpoint also handles models where that formula is wrong: Qwen3.5's q_proj is twice as wide (it also produces an output gate), only its full-attention layers have these modules, and Gemma 4 layers that reuse K/V have no k_proj or v_proj.

The frozen base still costs bΨ in weights, partitioned only at stage 3. Gradients and optimizer states exist only for the adapters.

When a decoder layer lacks one of the four targets, the CLI lists those layers, so a smaller count than expected is visible rather than silent. That is expected for Gemma 4 layers that reuse K/V, Gemma 4 global layers where K = V, and Qwen3.5's DeltaNet layers, and it is a real gap for MLA models, whose K/V projections have other names.

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
| SDPA (flash or memory-efficient) | p·q + 2p·kv + p·q (output) + F·heads (logsumexp), plus 16 bytes per call | K/V at full q width if they are copied per head |
| Eager attention | 4p·q + (F + p)·heads·s | the softmax and its copy grow with sequence length s |
| SwiGLU or GeGLU MLP, width I | 4p·I | 3p·I when down_proj is frozen |

SDPA has no s² term, because the fused kernel recomputes attention scores in backward. The 16 bytes per call are the memory-efficient kernel's philox seed and offset, two int64 scalars it keeps on the GPU even without dropout (`_efficient_attention_forward` in `aten/src/ATen/native/transformers/cuda/attention.cu`); the CUDA saved-tensor check on a T4 found them. The kernel saves K/V at their own width under grouped-query attention, as the formula assumes. Eager attention keeps an fp32 softmax and a dtype copy of it, heads x s values per token per layer. On a T4 or P100, SDPA uses the memory-efficient kernel, because FlashAttention needs sm80 or newer (`check_flash_attention_hardware_support` in PyTorch's `sdp_utils.cpp`). Both kernels keep the same tensors.

Llama, Mistral and Qwen2 add these up per layer: two norms, the projection inputs, SDPA and the MLP. For Qwen2.5-0.5B in bf16 that is 57,408 bytes per token per layer: 2.63 GiB for one 2048-token sequence over 24 layers.

Mistral, and Qwen2 with `use_sliding_window`, have sliding-window layers. Once the sequence reaches the window, such a layer needs an explicit mask (`_ignore_causal_mask_sdpa` in `masking_utils.py`), which costs what it costs in Gemma 4 below: SDPA keeps a float copy of the s x s mask, and K/V are copied to every query head. Mistral slides every layer, over 4096 tokens unless `config.json` says otherwise (Mistral-7B-v0.1 sets 4096, v0.3 sets no window). Qwen2 slides the layers from `max_window_layers` on, or those `layer_types` marks. For Mistral-7B-v0.1 at 8192 tokens that is 4 GiB of masks and 3 GiB of K/V copies over 32 layers.

### The output head

The logits are often the largest allocation. The model returns bf16 logits (p·V per token, V the vocabulary size), and `ForCausalLMLoss` upcasts them to fp32 and keeps the log-softmax output (F·V). At the start of backward, log-softmax's backward also holds the incoming gradient and its result, two more fp32 tensors of the same size. So the head peaks at (p + 3F)·V per token: 14 bytes per vocabulary entry in bf16. Gemma's final logit softcapping keeps one more p·V (the tanh output). For Qwen2.5-0.5B (V = 151,936) that is 4.07 GiB for 2048 tokens, more than all 24 layers together. Phase 6's fused linear plus cross-entropy kernel targets exactly this.

### Loss on the last n tokens

Some training loops compute the loss on the answer only:

    logits = model(input_ids=ids, logits_to_keep=n + 1).logits[:, :-1].float()
    loss = F.cross_entropy(logits.reshape(-1, vocab), ids[:, -n:].reshape(-1))

`Setup.loss_tokens = n` (`--loss-tokens n`) models this. The decoder layers are unchanged. The head changes because `lm_head` runs on `hidden_states[:, -(n + 1):]` (`slice_indices` in the `ForCausalLM.forward` of Llama, Qwen2, Qwen3.5 and Gemma 4), and the loss runs on n positions. Counting what autograd saves, per sequence:

- Log-softmax keeps its fp32 output: F·V per kept token, n tokens. The slice `[:, :-1]` is a view, and `.float()` saves nothing.
- `nll_loss` keeps the targets and the `total_weight` scalar. With b = 1, `ids[:, -n:].reshape(-1)` is a view of the input ids, so nothing new is kept. With b > 1 the reshape copies: 8 bytes per kept token.
- `lm_head` keeps its input if its weight trains. With b = 1 the slice is contiguous, so it is a view and the whole final-norm output (p·h per token, all s tokens) stays alive. With b > 1 `linear` flattens the slice with a copy, so only b·(n + 1) rows are kept.
- Gemma's softcapping keeps the tanh output on the n + 1 positions.
- Nothing is held: the caller drops the bf16 logits after `.float()`. The backward buffers are the same two fp32 tensors as before, on n tokens.

For b = 1 and a Llama-style model the head is the final norm, p·s·h, F·n·V + F, and 2F·n·V of backward buffers, against (p + 3F)·s·V in the default. For Qwen2.5-0.5B at s = 2048 and n = 64 the logits and loss drop from 4.07 GiB to under 0.2 GiB. `scripts/saved_tensors.py` checks these byte for byte for Llama, Qwen3.5 and Gemma 4 (full and LoRA, b = 1 and 2). It assumes the caller does not keep the model output alive during backward.

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

The training peak is the larger of two moments:

- The start of backward: weights, optimizer states, all activations and the head's buffers, plus the DeepSpeed buffers below. No gradients exist yet at stages 0-2, because they are created during backward, as activations are freed. ZeRO-3 is different: it allocates its gradient partition at init and keeps it (`grad_partitions_flat_buffer` in `stage3.py`), on the CPU with optimizer offload.
- The optimizer step, after activations are freed: weights, optimizer states, gradients, the buffers, and temporary fp32 copies of the gradients. Without ZeRO, `FP16_Optimizer.step` (`deepspeed/runtime/fp16/fused_optimizer.py`) converts each gradient to fp32 and then concatenates the copies into one flat tensor, so both exist for a moment: 8Ψt bytes. With ZeRO each rank flattens its partition and converts it to fp32: (b + 4)Ψt / N bytes, one sub-group at a time at stage 3 (`_prepare_fp32_grad_for_sub_group` in `stage3.py`, at most `sub_group_size`, default 1e9 elements). With optimizer offload the copies are on the CPU, and fp32 training makes none.

DeepSpeed also keeps buffers that don't depend on the model's size, from its `zero_optimization` defaults (`deepspeed/runtime/zero/config.py`):

- ZeRO-3 allocates a reduce bucket of `reduce_bucket_size` (5e8) elements in the training dtype at init and never frees it (`_setup_for_real_optimizer` in `stage3.py`): 0.93 GiB in fp16, even for LoRA.
- ZeRO-3 keeps a module's parameters gathered after it runs if the trace uses them again within `stage3_max_reuse_distance` (1e9) elements of the modules that follow (`__params_to_release` in `partitioned_param_coordinator.py`). The trace goes through the embedding, the layers and the head, then back through the head, the layers and the embedding. So layer i comes back after 2 x (parameters after it, plus the head) elements, and the last layers and the head are still gathered, as full tensors, when backward starts. For Qwen2.5-0.5B that is the whole model, because forward plus backward covers under 1e9 elements. With parameter offload the gathered copies are still on the GPU. FitCheck counts whole layers of equal size.
- ZeRO-1 and 2 with optimizer offload keep a GPU staging buffer the size of the largest trainable tensor (`temp_grad_buffer_for_gpu_offload` in `stage_1_and_2.py`), for Qwen2.5 the embedding.
- The ZeRO-1 and 2 reduce bucket is allocated at the first gradient hook and freed when backward ends (`setup_buckets`, `_release_ipg_buffers`). By then the head has freed more than the bucket takes, so it doesn't raise the allocated peak, though it does raise reserved memory (see Calibration).

### How the formulas are checked

`scripts/saved_tensors.py` builds small random-weight models that keep each family's structure: grouped-query attention, sliding windows, shared K/V, per-layer embeddings, K = V on global layers, global head_dim over 256, and DeltaNet layers. It runs a training forward pass on CPU, records every tensor autograd saves, and compares bytes per layer with the formulas. All cases match exactly: six configs (Llama, Mistral, Qwen2 with sliding layers, two Gemma 4 and Qwen3.5), full fine-tuning and LoRA, and SDPA bf16, eager bf16 and SDPA fp32, plus the answer-only loss above.

    uv run --group check python scripts/saved_tensors.py -v

Three things differ between CPU and CUDA, and the script handles them on CPU. HF's native-GQA rule depends on the device, so the script pins the CUDA rule. PyTorch's CPU attention kernel returns its output in the query's memory layout, which costs Qwen3.5 one extra copy that CUDA kernels don't make, so the script adds that copy to the prediction. And the CPU kernel keeps no philox seed and offset, so the script subtracts those 16 bytes per call. With `--device cuda` none of these adjustments apply. These are exact counts of saved tensors, not GPU measurements. They don't cover the allocator, CUDA workspaces or other short-lived buffers.

## Calibration

The formulas above count what PyTorch allocates, so the estimate is of peak allocated memory. The GPU also holds the CUDA context, cuBLAS and NCCL buffers outside the allocator. FitCheck covers that with one `CALIBRATE` constant in `estimate.py`:

    GPU total = allocated + C

C is `CUDA_CONTEXT_BYTES`. It comes only from measured runs, through `calibration.json`, and stays at 0 until then. The fits verdict compares this total with the GPU's memory.

Reserved memory, which is what PyTorch's caching allocator has taken from the device, is reported but not used. DeepSpeed's initialize and the later phases leave freed blocks in the cache, 1 to 5 GiB of them: after initialize at ZeRO-0 for Qwen2.5-0.5B, 11.15 GiB are reserved against 6.44 GiB allocated. Cached free blocks are not a hard limit, because PyTorch frees its unused cache and retries before it raises an OOM ("Memory management" in PyTorch's CUDA notes; `c10/cuda/CUDACachingAllocator.cpp`). The reserved to allocated ratio therefore varies from run to run, and no single factor would predict it.

`calibrate/run_one.py` trains for 8 steps under DeepSpeed in fp16 and records, on every rank, `max_memory_allocated` and `max_memory_reserved` for the forward and backward pass and for the optimizer step separately, plus device memory in use (`cudaMemGetInfo`) and memory reserved at the end. Three details keep the measurement comparable to the formulas:

- The model is put in train mode. `from_pretrained` returns it in eval mode, and HF only checkpoints when `self.training` is set (`GradientCheckpointingLayer.__call__` in `transformers/modeling_layers.py`). DeepSpeed's engine doesn't call `train()`. The first sweep missed this, so its checkpointing runs measured the same bytes as the runs without checkpointing and were rerun.
- The model's outputs stay alive through backward and are freed before the optimizer step, as the formulas assume.
- The dynamic loss scale starts at 1 instead of DeepSpeed's 2^16 (`initial_scale_power` 0). A step skipped on fp16 overflow returns before the fp32 gradient copy (`FP16_Optimizer.step` in `deepspeed/runtime/fp16/fused_optimizer.py`), so its peak would be too low. Runs with fewer than 4 applied steps don't count.

`calibrate/sweep.py` runs 55 configurations on 2x T4: Qwen2.5-0.5B and 1.5B, each ZeRO stage and offload combination, sequence lengths 512 to 4096 with and without checkpointing, LoRA, eager attention, micro-batch 2, and one GPU. `fitcheck calibrate results.jsonl` then compares, for every valid run, predicted allocated with the largest `max_memory_allocated` over the ranks. The target is within 2%. It prints the median and maximum absolute error per model and overall, next to each run's measured reserved memory and reserved to allocated ratio. It sets C to the median over ranks of device memory in use minus memory reserved at the end of the run. For the offload runs it also prints FitCheck's host RAM estimate against each rank's peak resident memory (`ru_maxrss`), summed over ranks. Nothing is fitted to that yet: the measured number also holds the model as loaded on the CPU and the Python, PyTorch and CUDA runtimes. The 1.5B offload runs need the most host RAM, so the sweep runs them last. `calibration.json` records C, the allocated error, the results file's hash, the dates, the GPU, the driver and the library versions.

It refuses to fit until `scripts/saved_tensors.py --device cuda` has matched every case on the T4, on the same torch as the runs; `--ignore-cuda-check` overrides that and records why. Otherwise a formula error on CUDA would go unnoticed next to C. On CUDA the check uses sequence lengths that are multiples of 32, because the memory-efficient kernel pads its logsumexp to a multiple of 32 queries (`kAlignLSE` in `mem_eff_attention/kernel_forward.h`), which the formulas leave out; the sweep's lengths are multiples of 512.

### Why reserved memory is not predicted

Allocator traces of three runs (`run_one.py --trace`, snapshots in the `calibration-2026-11-14` GitHub release) show where the gap between reserved and allocated memory comes from. No segment is ever returned to the device during training, so reserved memory only grows, and every cudaMalloc stays reserved until the process ends.

- `deepspeed.initialize` leaves holes. The fp16 weights are copied to the GPU, flattened into one buffer (`stage_1_and_2.py`, `fused_optimizer.py`) and freed. At ZeRO-1 and 2 a partition clone is made and freed straight away. These holes have the exact sizes of model tensors, so later buffers of a different size can't use them.
- ZeRO-1 and 2 then allocate three buffers of nearly the same size during each step, one after another: the reduce bucket (5e8 elements, 0.93 GiB in fp16), the flattened gradients in `allreduce_bucket` (0.92 GiB for Qwen2.5-0.5B), and an fp32 copy of the gradients for the gradient norm (`get_grad_norm_direct`, 1.02 GiB, on rank 0 only). None fits the hole left by the one before it. The bucket misses a 0.92 GiB init hole by 9 MiB. In the ZeRO-2 offload run these three account for 2.87 of the 2.88 GiB gap.
- ZeRO-0 has no such per-step buffers, and its activations fit into init holes, so its gap is close to one fp32 gradient copy (1.84 GiB).

The gap depends on allocation order and on near-misses like 0.93 against 0.92 GiB, so there is no clean formula in terms of model size. That is why the fits verdict uses allocated memory. PyTorch releases unused cached segments and retries before it raises an out-of-memory error, but it can't release a segment that still holds a live block. So holes can cause an OOM: the ZeRO-0 run with accumulation 4 failed to allocate 1.16 GiB with 1.65 GiB reserved but unallocated and 0.84 GiB free on the device. Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` should shrink them, but that hasn't been measured.

### Second check: Ratchet's runs

The DeepSpeed sweep above only covers Qwen2.5. Ratchet's `train/memprobe.py` measured peak allocated memory outside DeepSpeed, on one T4: plain PyTorch, PEFT LoRA r=16, `torch.optim.AdamW`, fp16, SDPA, checkpointing with `use_reentrant=False`, random ids, torch 2.11.0 and transformers 5.22.2, on Qwen3.5-0.8B and 2B. The results are in `calibrate/results/ratchet-memprobe-2026-11-12.json`, and `scripts/compare_memprobe.py` prints the comparison. The "answer only" rows take logits for the last 65 positions and compute the loss on 64 tokens, which the script models with `Setup.loss_tokens=64`. Allocated memory, GiB:

    #  model          seq   ckpt  fla    pred   meas     err  note
    0  Qwen3.5-0.8B   512   True  True    1.62   1.83  -11.5%
    1  Qwen3.5-0.8B   4096  True  True    2.41   3.92  -38.6%
    2  Qwen3.5-0.8B   4096  True  False   2.41   3.92  -38.6%
    3  Qwen3.5-0.8B   4096  True  True       -      -       -  not compared: oom
    4  Qwen3.5-2B     4096  True  True    4.79   6.26  -23.5%
    5  Qwen3.5-2B     4096  False True       -      -       -  not compared: oom
    6  Qwen3.5-2B     4096  True  True       -   6.44       -  not compared: LoRA on all-linear
    7  Qwen3.5-2B     6144  True  True    5.43   9.10  -40.3%

The model states match: FitCheck gives 1.42 GiB against 1.43 measured after loading for 0.8B, and 3.53 against 3.51 for 2B. What is above the states does not match. Taking FitCheck's own terms (the layer inputs kept by checkpointing, the largest layer recomputed in backward, and the head for 65 positions at about 0.21 GiB by the formula in "The output head"), and subtracting the measured post-load memory from the measured peak:

| Row | FitCheck above states | Measured above states | Gap |
|---|---|---|---|
| 0.8B, seq 512 | 0.33 | 0.40 | 0.07 |
| 0.8B, seq 4096 | 1.20 | 2.49 | 1.29 |
| 2B, seq 4096 | 1.48 | 2.75 | 1.27 |
| 2B, seq 6144 | 2.11 | 5.59 | 3.47 |

All numbers are GiB. The gap is small at 512 and grows faster than linearly: from 4096 to 6144 on 2B it goes up 2.7 times, where the sequence grows 1.5 times. The two 0.8B rows at 4096 peak at the same allocated bytes with and without `flash-linear-attention`, so the package did not change the peak in that run (whether transformers used its kernels there was not checked).

Likely cause, read from source but not yet measured: on the T4 with torch 2.11 the full-attention layers run SDPA's math path, which builds s x s attention weights in fp32. The reading of PyTorch v2.11.0 behind it:

- Flash attention is rejected on sm75: `check_flash_attention_hardware_support` accepts sm80 and newer (`sdp_utils.cpp`, line 298).
- The memory-efficient kernel has no head_dim limit on CUDA, only an alignment check, which 256 passes (`check_head_dim_size_mem_efficient`, line 247). So head_dim 256 is not the reason.
- The memory-efficient kernel does not accept grouped-query attention: its dense check is called with `supports_grouped_query_attention=false` (line 901), so with `enable_gqa=True` it requires equal head counts (`sdp_utils_cpp.h`, lines 411-428) and Qwen3.5's 8 query and 2 KV heads fail it.
- transformers 5.22.2 sets `enable_gqa=True` whenever there is no mask and head_dim is at most 256 (`use_gqa_in_sdpa`, `sdpa_attention.py`, lines 30-37, 114-118). Its comment says this is meant to avoid the math kernel, which holds on GPUs where flash attention runs but not on a T4.
- The math path upcasts q, k and v to fp32 unless reduced-precision reduction is allowed (`_scaled_dot_product_attention_math`, `attention.cpp`, line 850 on), then computes `matmul`, a causal mask add and `_safe_softmax` (an fp32 softmax plus a boolean copy). Autograd keeps the softmax output, and backward adds at least one more tensor of that size.

FitCheck's own T4 runs don't show this because they used torch 2.14.1, whose memory-efficient kernel does accept grouped-query attention for dense inputs on CUDA (`supports_gqa = true` in `can_use_mem_efficient_attention`, `sdp_utils.cpp` v2.14.1). The CUDA saved-tensor check confirms the efficient kernel ran there: it counts that kernel's philox seed and offset and matched Qwen2.5 and Qwen3.5 exactly. So the gap is a torch version effect. With torch 2.11 on a GPU without flash attention, any model with grouped-query attention and no mask falls back to the math path, which FitCheck doesn't model. Upgrading to torch 2.14.1 should remove it. One run of memprobe on torch 2.14.1 would confirm it.

If that is what runs, one s x s fp32 tensor is 8 heads x s² x 4 bytes: 0.50 GiB at 4096 and 1.13 GiB at 6144. About three of them alive during one layer's recompute and backward would be 1.5 GiB and 3.4 GiB, against measured gaps of 1.27-1.29 and 3.47. That fits in size and in growth, but "about three" is my reading of the source and no run has looked at the allocator. The check is to call SDPA on the T4 with `enable_gqa=True` inside `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`, which should raise if the kernel is rejected, or to record `max_memory_allocated` around one full-attention layer. FitCheck's formulas still assume the memory-efficient kernel (see "Activations") and have not been changed.

## Choosing a setup

`fitcheck recommend` estimates every ZeRO and offload combination DeepSpeed accepts (stages 0-3, optimizer offload at stages 1-3, parameter offload at stage 3), each with and without activation checkpointing, for the same model, task, dtype and batch. That is 16 setups (`recommend.options`).

A setup fits when P / (1 - e) + C is at most the GPU's memory, where P is the predicted allocated peak, C the CUDA context, and e the largest calibration error, |predicted / measured - 1|, from `calibration.json` (3.4% now). If the estimate is off by e, the measured peak is at most P / (1 - e). The plain `fitcheck` estimate keeps its verdict without this margin.

Host RAM can only rule a setup out. The offload estimate counts the offloaded state: 16 bytes per trainable parameter, plus the weights with parameter offload. It leaves out the model as loaded on the CPU and the Python, PyTorch and CUDA runtimes. On the T4 the 0.5B offload runs predicted 7.36 GiB (8.28 GiB with parameter offload) and measured 13.76 to 17.14 GiB over both ranks, 1.9 to 2.1 times the estimate, and every 1.5B offload run (23 GiB predicted) failed on Kaggle's 31 GB of host RAM. So a setup over `--host-mem` is ruled out, and one under it is marked "host RAM unchecked". If a model has something FitCheck doesn't model, no setup is recommended and no config is written, since a "yes" could be wrong.

The setups are ranked by this order, cheapest first. It is a rule of thumb from how each feature works, not measured:

1. Offload: none, then the optimizer, then the optimizer and the parameters. Optimizer offload runs Adam on the CPU and moves every step's gradients and weights over PCIe (ZeRO-Offload, Ren et al. 2021). Parameter offload also moves the weights in for every forward and backward.
2. Checkpointing: off, then on. It runs each layer's forward again in backward, about a third more compute, since backward costs about twice the forward.
3. ZeRO stage, lowest first. Stages 1 and 2 communicate as much as plain data parallelism, 2Ψ per step, and stage 3 communicates 3Ψ (ZeRO paper, section 7).

So ZeRO-3 without checkpointing ranks above ZeRO-2 with it. Whether 1.5 times the communication over the T4's PCIe costs more than a third more compute has not been measured.

### The generated config

`recommend.ds_config` writes a config for `deepspeed.initialize` or the `deepspeed` launcher, with concrete values (no HF Trainer `"auto"`). Besides the micro-batch, gradient accumulation, dtype and ZeRO stage, it writes out the DeepSpeed v0.19.9 defaults the estimate depends on, so the estimate stays right if a default changes:

- `contiguous_gradients: true` and `reduce_bucket_size: 5e8`. The ZeRO-1 gradient and ZeRO-3 bucket terms assume both.
- `overlap_comm`, on only at stage 3, as `overlap_comm_valid` in `deepspeed/runtime/zero/config.py` sets it. With it off, ZeRO-1 reduces only at the accumulation boundary.
- `sub_group_size: 1e9` and `stage3_max_reuse_distance: 1e9` at stage 3, the step's sub-group and the gathered parameters at the start of backward.
- `data_types.grad_accum_dtype` equal to the training dtype. With fp32 accumulation, bf16 at ZeRO-1 switches to `BF16_Optimizer` (`_do_optimizer_sanity_check` in `deepspeed/runtime/engine.py`), which FitCheck doesn't model.
- An `AdamW` optimizer, which DeepSpeed builds as FusedAdam on the GPU or DeepSpeedCPUAdam with optimizer offload (`get_optimizer_configuration` in `engine.py`): the 12 bytes of optimizer state per parameter the estimate counts.
- Offload to `cpu` with `pin_memory: true`, the default (`deepspeed/runtime/zero/offload_config.py`), which the calibration runs also used. Pinned host memory is page-locked, so the offloaded state can't be swapped out: it has to fit in RAM.

`tests/test_recommend.py` loads every combination of stage, offload and dtype with DeepSpeed's own parser (`DeepSpeedConfig` in `deepspeed/runtime/config.py`, which rejects unknown keys) and checks that it reads back the values above. `calibrate/run_one.py` builds its config with the same function, changing only the initial loss scale. The runs in `calibrate/results/` were made before that, with a shorter config whose effective values are the same, since every added key is at its default.

The training script must still set what the config can't hold: the attention implementation, `gradient_checkpointing_enable()` and `model.train()`, the LoRA targets, and `logits_to_keep` for loss on the last tokens. `fitcheck recommend` prints these.

### Checked on the T4

`sweep.py --grid recommend` ran what `recommend --gpus 2 --gpu-mem 14.56 --dtype fp16` picked for three goals, each with the generated config, and for two goals also the cheaper setup it rejected (`calibrate/results/2026-11-14-recommend/`):

| Goal | Setup | Predicted GiB | Measured GiB | Result |
|---|---|---|---|---|
| 0.5B full, seq 4096 | ZeRO-1, checkpointing (picked) | 11.99 | 12.00 | trained |
| 0.5B full, seq 4096 | ZeRO-0, checkpointing (rejected) | 14.75 | - | out of memory |
| 0.5B full, seq 2048, accumulation 4 | ZeRO-1 (picked) | 11.30, now 12.23 | 12.31 | trained |
| 0.5B full, seq 2048, accumulation 4 | ZeRO-0 (rejected by the margin) | 14.06 | - | out of memory |
| 1.5B LoRA r=16, seq 4096 | ZeRO-0, checkpointing (picked) | 11.40 | 11.42 | trained |

Predicted and measured are peak allocated memory, without the CUDA context. All three picks trained, and both rejected setups ran out of memory. Without the margin, ZeRO-0 with accumulation 4 would have been called a fit. The margin saved that verdict, but only because the gap was small next to it: the 8.2% error on the ZeRO-1 run with accumulation was more than twice the margin. An allocator trace then found the missing term, ZeRO-1's reduce bucket, which brings that run to 0.7%. Gradient accumulation at the other stages is still reported as not modeled (see "ZeRO stages").

## Not modeled yet

These are known gaps. Each should show up as error in the Phase 3 calibration runs or get its own term. The last three are detected from `config.json`: the CLI names them and answers "unknown" instead of "yes", unless the part it counts already exceeds the GPU, since what is left out only adds memory.

- Short-lived buffers other than the loss backward: matmul outputs before they are consumed, gradients of activations during backward, attention kernel workspaces.
- ZeRO-3 prefetch beyond the parameters it keeps gathered, which the measured runs show no sign of at the start of backward.
- Whether CUDA's SDPA output really needs no copy before `o_proj`, as assumed above. Phase 3 measures it.
- Qwen3.5 with `flash-linear-attention` installed, and kernels from the `kernels` package.
- Padding, which brings back masks and, on CUDA, K/V copies.
- The CUDA context, a `CALIBRATE` constant fitted from measured runs as described under Calibration. Memory the allocator has reserved but not allocated is not counted.
- Mixture-of-experts layers (Gemma 4 26B-A4B, Qwen3.5 MoE, Mixtral, DeepSeek, Llama 4), detected from `num_experts`, `num_local_experts` or `n_routed_experts`, and architectures without a formula: FitCheck reports their model states only. Gemma 4 26B-A4B shares `gemma4_text` with the dense Gemma 4 models, so the expert count, not the model type, decides.
- Multi-head latent attention (DeepSeek, detected from `kv_lora_rank`): the head size shown is wrong and LoRA misses the latent K/V projections.
- Quantized checkpoints (any `quantization_config`). Packed formats such as GPTQ and AWQ undercount Ψ.

## Sources

- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2020. arXiv:1910.02054.
- Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training", 2021. arXiv:2101.06840.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. arXiv:2106.09685.
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", 2022. arXiv:2205.05198. Its per-layer counting method, applied here to the HF modeling code.
- DeepSpeed source, github.com/deepspeedai/DeepSpeed, read at v0.19.8 and v0.19.9.
- transformers source, github.com/huggingface/transformers, read at 5.19: `models/llama`, `models/gemma4`, `models/qwen3_5`, `integrations/sdpa_attention.py`, `masking_utils.py`, `loss/loss_utils.py`.
- PyTorch source, github.com/pytorch/pytorch: `tools/autograd/derivatives.yaml`, `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`.
- PyTorch CUDA notes, "Memory management", docs.pytorch.org/docs/stable/notes/cuda.html, and `c10/cuda/CUDACachingAllocator.cpp`.
- safetensors format, github.com/huggingface/safetensors.
