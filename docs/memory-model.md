# Memory model

This is what FitCheck counts, where each formula comes from, and what it leaves out. So far it covers model states only: weights, gradients and optimizer states. Activations come in Phase 2.

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

LoRA freezes each targeted Linear(in, out) and trains A (r x in) and B (out x r) beside it, adding r(in + out) parameters (Hu et al. 2021). FitCheck targets q, k, v and o in every layer. With hidden size h, query width q = heads x head_dim and KV width kv = kv_heads x head_dim:

    Ψt = r * layers * [(h + q) + 2(h + kv) + (q + h)]

The frozen base still costs bΨ in weights, partitioned only at stage 3. Gradients and optimizer states exist only for the adapters. Models that fuse q, k and v into one projection (Phi-3, for example) get slightly different counts. That case isn't handled yet.

## Not modeled yet

These are known gaps. Each one should either show up as error in the Phase 3 calibration runs or get its own term:

- Activations, logits and loss (Phase 2).
- At the optimizer step, DeepSpeed builds an fp32 copy of the gradients: all of them at stage 0 (`FP16_Optimizer.step` in `deepspeed/runtime/fp16/fused_optimizer.py`), and each rank's partition at stages 1-2 (`stage_1_and_2.py`). That is up to 4Ψt more bytes at a point where activations are already freed, so the true peak is max(backward peak, step peak). Phase 2 has to model that max.
- Communication buckets (`reduce_bucket_size`, `allgather_bucket_size`), and the parameters stage 3 gathers and prefetches for the layer it is running.
- The CUDA context, and the gap between memory the allocator has reserved and memory actually allocated. These are the `CALIBRATE` constants in `estimate.py`, left at neutral values until measured.
- Quantized checkpoints (GPTQ, AWQ), whose packed tensors undercount Ψ.

## Sources

- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2020. arXiv:1910.02054.
- Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training", 2021. arXiv:2101.06840.
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021. arXiv:2106.09685.
- DeepSpeed source, github.com/deepspeedai/DeepSpeed, read at v0.19.8.
- safetensors format, github.com/huggingface/safetensors.
