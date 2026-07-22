#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~355M total params) on ~7.1B tokens (D=20N Chinchilla).
# Global batch = GRAD_ACCUM * BATCH_SIZE * WORLD_SIZE = 576 sequences/iter.
# Micro-batch is 48 (was 96) with GA=12 (was 6): identical effective batch and
# token budget, but ~half the per-step activation/logit peak so it fits on a
# single 80GB GPU. At BSZ=96 the float32 cross-entropy over the 50304-vocab
# logits (~20GB) pushed a single L20Z past 80GB and OOM'd right after step 0.
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS=${MAX_ITERS:-12030} EVAL_INTERVAL=${EVAL_INTERVAL:-1000} \
BATCH_SIZE=${BATCH_SIZE:-48} GRAD_ACCUM=${GRAD_ACCUM:-12} LEARNING_RATE=${LEARNING_RATE:-3e-4} \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
