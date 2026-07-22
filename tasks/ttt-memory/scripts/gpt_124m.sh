#!/bin/bash
# GPT-2 Small (12L/12H/768D, ~124M params) verification run.
# Single GPU, train_ctx parameterized via BLOCK_SIZE env (default 2048).
# Token budget defaults: 2.5B (Chinchilla-optimal for 124M).
export TIKTOKEN_CACHE_DIR=/workspace/_task/tiktoken_cache
cd "${MLSBENCH_PKG_DIR:-/workspace/nanoGPT}"
N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-out}"
ENV="${ENV:-gpt-124m}"
if [ -z "${OUTPUT_DIR}" ] || [ "${OUTPUT_DIR}" = "/" ]; then
    echo "ERROR: refusing to clear unsafe OUTPUT_DIR='${OUTPUT_DIR}'"
    exit 1
fi
rm -rf "${OUTPUT_DIR}"
# 2.5B tokens / (BS=16 * GA=8 * BLOCK_SIZE=2048) = 9537 iters
SEED="${SEED}" OUTPUT_DIR="${OUTPUT_DIR}" ENV="${ENV}" \
N_LAYER=12 N_HEAD=12 N_EMBD=768 \
BLOCK_SIZE="${BLOCK_SIZE:-2048}" MODEL_BLOCK_SIZE="${MODEL_BLOCK_SIZE:-8192}" \
MAX_ITERS="${MAX_ITERS:-9537}" EVAL_INTERVAL="${EVAL_INTERVAL:-1000}" \
BATCH_SIZE="${BATCH_SIZE:-16}" GRAD_ACCUM="${GRAD_ACCUM:-8}" LEARNING_RATE="${LEARNING_RATE:-6e-4}" \
torchrun --nproc_per_node="${N_GPU}" --standalone custom_pretrain.py
