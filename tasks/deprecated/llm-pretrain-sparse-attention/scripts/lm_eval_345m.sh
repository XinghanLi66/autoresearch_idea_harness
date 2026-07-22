#!/bin/bash
set -e

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
READONLY_HF_DATASETS_CACHE="/data/lm-eval-datasets"
export HF_DATASETS_CACHE="${OUTPUT_DIR}/hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"
if [ ! -f "${HF_DATASETS_CACHE}/.seeded" ]; then
    cp -al "${READONLY_HF_DATASETS_CACHE}/." "${HF_DATASETS_CACHE}/" 2>/dev/null || \
        cp -a "${READONLY_HF_DATASETS_CACHE}/." "${HF_DATASETS_CACHE}/"
    touch "${HF_DATASETS_CACHE}/.seeded"
fi
find "${HF_DATASETS_CACHE}" -name '*.lock' -delete

CKPT_PATH="${OUTPUT_DIR}/ckpt_gpt-345m.pt"
SOURCE_PATH="${OUTPUT_DIR}/model_source_gpt-345m.py"
TRAIN_MARKER="${OUTPUT_DIR}/.training_complete"

# gpt_345m.sh writes .training_complete only after a successful torchrun.
# Refuse to evaluate stale checkpoints from a previous experiment.
if [ ! -f "${TRAIN_MARKER}" ]; then
    echo "ERROR: ${TRAIN_MARKER} missing — gpt_345m did not finish in this run."
    echo "       Refusing to evaluate a possibly stale checkpoint."
    exit 1
fi
if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi

echo "Evaluating checkpoint: ${CKPT_PATH}"
echo "Model source: ${SOURCE_PATH}"

python nanogpt_lm_eval.py \
    --checkpoint "${CKPT_PATH}" \
    --source "${SOURCE_PATH}" \
    --tasks hellaswag,arc_easy,piqa,winogrande \
    --num_fewshot 0 \
    --batch_size 1 \
    --device cuda
