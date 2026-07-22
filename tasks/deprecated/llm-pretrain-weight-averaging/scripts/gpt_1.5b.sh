#!/bin/bash
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
# custom_pretrain.py divides GRAD_ACCUM by WORLD_SIZE under DDP.
# With 4 GPUs this sets env GRAD_ACCUM=20, so actual tokens/iter = 20 * 16 * 1024 = 327,680.
GRAD_ACCUM=$((1280 / (16 * N_GPU)))

N_LAYER=48 N_HEAD=25 N_EMBD=1600 MAX_ITERS=30517 EVAL_INTERVAL=2000 \
BATCH_SIZE=16 GRAD_ACCUM=${GRAD_ACCUM} LEARNING_RATE=2.5e-4 \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py
