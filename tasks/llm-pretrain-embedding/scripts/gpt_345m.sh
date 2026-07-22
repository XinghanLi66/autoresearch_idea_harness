#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~355M total params) on ~7.1B tokens (D=20N Chinchilla).
# 2-GPU DDP, BSZ=96 per GPU, GA=6 (divided by world_size internally).
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS=${MAX_ITERS:-12030} EVAL_INTERVAL=${EVAL_INTERVAL:-1000} \
BATCH_SIZE=${BATCH_SIZE:-96} GRAD_ACCUM=${GRAD_ACCUM:-6} LEARNING_RATE=${LEARNING_RATE:-3e-4} \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
