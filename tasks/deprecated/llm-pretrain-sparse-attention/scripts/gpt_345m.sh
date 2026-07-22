#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~355M total params) on ~7.1B tokens (D=20N Chinchilla).
# 2-GPU DDP, BSZ=64 per GPU, GA=8.
set -e

# Wipe any stale per-run artifacts so a failed retrain cannot be silently
# evaluated by group_2 (lm_eval) using an older checkpoint.
rm -f "${OUTPUT_DIR}/ckpt_gpt-345m.pt" \
      "${OUTPUT_DIR}/model_source_gpt-345m.py" \
      "${OUTPUT_DIR}/.training_complete"
mkdir -p "${OUTPUT_DIR}"

N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS=${MAX_ITERS:-13535} EVAL_INTERVAL=${EVAL_INTERVAL:-1000} \
BATCH_SIZE=${BATCH_SIZE:-64} GRAD_ACCUM=${GRAD_ACCUM:-8} LEARNING_RATE=${LEARNING_RATE:-3e-4} \
torchrun --nproc_per_node=${N_GPU} --standalone custom_pretrain.py

# Mark training complete; lm_eval_345m.sh refuses to evaluate without this.
touch "${OUTPUT_DIR}/.training_complete"
