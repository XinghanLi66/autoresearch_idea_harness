#!/bin/bash
set -e
# Evaluate the SFT'd model on downstream NLU benchmarks using lm-evaluation-harness

# Use pre-downloaded HF datasets from host filesystem (bind-mounted)
export HF_DATASETS_CACHE="/data/lm-eval-datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_PATH="${OUTPUT_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    exit 1
fi

echo "Evaluating model: ${MODEL_PATH}"

lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},trust_remote_code=True,attn_implementation=eager" \
    --tasks hellaswag,arc_challenge,piqa \
    --num_fewshot 0 \
    --batch_size auto \
    --device cuda \
    --output_path "${OUTPUT_DIR}/lm_eval_results"

echo "Evaluation complete. Results saved to ${OUTPUT_DIR}/lm_eval_results"
