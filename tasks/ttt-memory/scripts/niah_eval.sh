#!/bin/bash
set -e

# Standalone NIAH passkey retrieval eval with NTK-aware RoPE scaling.
# Loads the saved nanoGPT checkpoint + model source from training, runs
# the eval at three context lengths (1280, 1536, 2048) — all in the
# discriminating range just past the vanilla-RoPE extrapolation cliff.

export TIKTOKEN_CACHE_DIR=/workspace/_task/tiktoken_cache

CKPT_PATH="${OUTPUT_DIR}/ckpt_gpt-345m.pt"
SOURCE_PATH="${OUTPUT_DIR}/model_source_gpt-345m.py"

check_fresh_checkpoint() {
    python3 - "$1" <<'PY'
import os
import sys
import time

path = sys.argv[1]
age = time.time() - os.path.getmtime(path)
if age > 3600:
    raise SystemExit(
        f"ERROR: stale checkpoint: {path} is {age / 3600:.2f} hours old; "
        "refusing to reuse an old OUTPUT_DIR after failed training."
    )
PY
}

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CKPT_PATH}"
    exit 1
fi
if [ ! -f "${SOURCE_PATH}" ]; then
    echo "ERROR: Model source not found: ${SOURCE_PATH}"
    exit 1
fi
check_fresh_checkpoint "${CKPT_PATH}"

echo "Evaluating checkpoint: ${CKPT_PATH}"
echo "Model source: ${SOURCE_PATH}"

python /workspace/_task/niah_eval.py \
    --checkpoint "${CKPT_PATH}" \
    --source "${SOURCE_PATH}" \
    --ctx-lens "${NIAH_CTX_LENS:-1280,1536,2048}" \
    --n-samples "${NIAH_N_SAMPLES:-40}" \
    --seed "${SEED:-42}" \
    --device cuda
