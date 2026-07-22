#!/bin/bash
set -e

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
READONLY_HF_DATASETS_CACHE="/data/lm-eval-datasets"
export HF_DATASETS_CACHE="${OUTPUT_DIR}/hf_datasets_cache"
mkdir -p "${HF_DATASETS_CACHE}"
# Idempotently sync (newly-)vendored datasets into the writable per-run cache.
# `cp -aln` hardlinks files that are MISSING and skips existing ones without
# error, so datasets added to the read-only vendor cache AFTER this per-seed
# OUTPUT_DIR was first created still get picked up. The previous `.seeded`
# marker froze the cache at whatever was vendored on the first run — if that
# was only hellaswag, arc_easy/piqa/winogrande would never appear and lm_eval
# would raise "Offline mode is enabled" ConnectionError for them.
cp -aln "${READONLY_HF_DATASETS_CACHE}/." "${HF_DATASETS_CACHE}/" 2>/dev/null || \
    cp -an "${READONLY_HF_DATASETS_CACHE}/." "${HF_DATASETS_CACHE}/"
find "${HF_DATASETS_CACHE}" -name '*.lock' -delete

CKPT_PATH="${OUTPUT_DIR}/ckpt_gpt-345m.pt"
SOURCE_PATH="${OUTPUT_DIR}/model_source_gpt-345m.py"

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
