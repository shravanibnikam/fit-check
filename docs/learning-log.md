# Learning log

What each phase taught, and what I should now be able to explain in an interview.

## Phase 1: Foundation

A safetensors file starts with an 8-byte length and a JSON header listing every tensor's dtype and shape, so a model's exact parameter count costs a few KB of HTTP range requests, not a weights download. Mixed-precision Adam keeps 16 bytes per parameter: a 2-byte weight and gradient, plus a 4-byte fp32 master copy and two 4-byte moments. ZeRO stages 1, 2 and 3 partition the optimizer states, then the gradients, then the weights across data-parallel ranks, and offload moves the optimizer (and at stage 3 the weights) to CPU RAM.

I should be able to explain: why fine-tuning a 7B model in mixed precision needs about 112 GB before activations, the per-GPU memory formula for each ZeRO stage, what optimizer offload saves on the GPU and costs in host RAM, how LoRA's r(in + out) parameters change which terms shrink, and why an HTTP client must drop the auth token on a cross-host redirect.

## Phase 2: Activations

Activation memory is the set of tensors autograd saves in the forward pass for backward, and each op saves only what its required gradients need: a frozen Linear keeps nothing for its weight, RoPE keeps only cos and sin, and LoRA therefore saves less than full fine-tuning. Fused SDPA keeps no s x s scores, while eager attention keeps an fp32 softmax and its copy. At large vocabularies the output head dominates: fp32 log-softmax plus its backward buffers come to 14 bytes per vocabulary entry per token in bf16, more than all layers of a small model together. Implementation details move memory too: whether transformers builds an explicit mask (sliding windows, padding) decides whether SDPA keeps an s x s bias and whether K/V get copied to every query head.

I should be able to explain: how to count a decoder layer's saved tensors from the modeling code, why logits and loss are often the largest allocation, what activation checkpointing keeps and where its peak is, why the training peak is the larger of the start of backward and the optimizer step's fp32 gradient copy, and how `torch.autograd.graph.saved_tensors_hooks` checks such a formula byte for byte.
