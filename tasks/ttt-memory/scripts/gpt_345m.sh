#!/bin/bash
# GPT-2 Medium (24L/16H/1024D, ~345M params) on ~7.1B tokens (Chinchilla-optimal).
# 2-GPU DDP. Training seq_len=1024; NIAH eval goes to 8K via model_block_size bump.
# tiktoken's default cache (/tmp/data-gym-cache) is hidden by apptainer's
# --writable-tmpfs overlay at SLURM runtime; point the loader at the copy that
# ships with the task so offline nodes don't try an HTTP fetch.
export TIKTOKEN_CACHE_DIR=/workspace/_task/tiktoken_cache
cd "${MLSBENCH_PKG_DIR:-/workspace/nanoGPT}"
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-out}"
ENV="${ENV:-gpt-345m}"
if [ -z "${OUTPUT_DIR}" ] || [ "${OUTPUT_DIR}" = "/" ]; then
    echo "ERROR: refusing to clear unsafe OUTPUT_DIR='${OUTPUT_DIR}'"
    exit 1
fi
rm -rf "${OUTPUT_DIR}"
SEED="${SEED}" OUTPUT_DIR="${OUTPUT_DIR}" ENV="${ENV}" \
N_LAYER=24 N_HEAD=16 N_EMBD=1024 \
MAX_ITERS="${MAX_ITERS:-13535}" EVAL_INTERVAL="${EVAL_INTERVAL:-1000}" \
BATCH_SIZE="${BATCH_SIZE:-32}" GRAD_ACCUM="${GRAD_ACCUM:-16}" LEARNING_RATE="${LEARNING_RATE:-3e-4}" \
torchrun --nproc_per_node="${N_GPU}" --standalone custom_pretrain.py
