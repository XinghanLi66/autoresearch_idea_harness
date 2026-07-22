#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
cd "${WORK_ROOT}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

TASK_DIR="${TASK_DIR:-${WORK_ROOT}/_task}"
PKG_DIR="${PKG_DIR:-${WORK_ROOT}/Unify-Post-Training}"
OPENR1_PATH="${OPENR1_PATH:-/root/data/openr1.parquet}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen2.5-Math-1.5B}"
SEED_VALUE="${SEED:-42}"
export PYTHONHASHSEED="${SEED_VALUE}"
# Always use /tmp inside container — shared between group_1 (train) and group_2 (relay)
# via the same apptmp bind. mlsbench's OUTPUT_DIR points to a host path not bound here.
RUN_ROOT="/tmp/mlsbench_hpt_shared"
MODEL_RUN_DIR="${RUN_ROOT}/Qwen2.5-Math-1.5B-ctx16k"
PKG_DATA_SUBDIR="data"
PKG_DATA_DIR="${PKG_DIR}/${PKG_DATA_SUBDIR}"
SPLIT_DIR="${RUN_ROOT}/shared_splits"
TRAIN_LOG="${RUN_ROOT}/shared_train.log"
FINAL_METRICS="${RUN_ROOT}/final_metrics.txt"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
TRAIN_SIZE="${TRAIN_SIZE:-144}"
TRAIN_SOURCE_SPEC="${TRAIN_SOURCE_SPEC:-amc_aime:72,cn_contest:36,olympiads:36}"
TRAIN_SOURCE_SELECTION="${TRAIN_SOURCE_SELECTION:-shortest_proxy}"
EVAL_AIME24_PATH="${EVAL_AIME24_PATH:-${TASK_DIR}/data/eval_subsets_20260426/AIME24_eval.parquet}"
EVAL_AMC23_PATH="${EVAL_AMC23_PATH:-${TASK_DIR}/data/eval_subsets_20260426/AMC23_eval.parquet}"
EVAL_MATH500_PATH="${EVAL_MATH500_PATH:-${TASK_DIR}/data/eval_subsets_20260426/MATH-500_eval.parquet}"
AIME24_SIZE="${AIME24_SIZE:-20}"
AMC23_SIZE="${AMC23_SIZE:-20}"
MATH500_SIZE="${MATH500_SIZE:-30}"
INTERNAL_AIME24_SIZE="${INTERNAL_AIME24_SIZE:-1}"
INTERNAL_AMC23_SIZE="${INTERNAL_AMC23_SIZE:-1}"
INTERNAL_MATH500_SIZE="${INTERNAL_MATH500_SIZE:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_N_GPUS="${EVAL_N_GPUS:-1}"
EXTERNAL_EVAL_SAMPLES="${EXTERNAL_EVAL_SAMPLES:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
TEST_FREQ="${TEST_FREQ:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-4}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-12288}"
LOG_PROB_USE_DYNAMIC_BSZ="${LOG_PROB_USE_DYNAMIC_BSZ:-True}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
ROLLOUT_NAME="${ROLLOUT_NAME:-vllm}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VERIFY_N="${VERIFY_N:-8}"
VAL_N="${VAL_N:-4}"
ROLLOUT_MAX_PREFIX_LEN="${ROLLOUT_MAX_PREFIX_LEN:-8192}"
SAVE_FREQ="${SAVE_FREQ:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

mkdir -p "${RUN_ROOT}"
rm -rf "${SPLIT_DIR}" "${CHECKPOINT_DIR}" "${RUN_ROOT}"/eval_* "${FINAL_METRICS}" "${TRAIN_LOG}"
mkdir -p "${SPLIT_DIR}" "${CHECKPOINT_DIR}"

if [ ! -f "${MODEL_RUN_DIR}/config.json" ]; then
  rm -rf "${MODEL_RUN_DIR}"
  mkdir -p "${MODEL_RUN_DIR}"
  cp -as "${MODEL_DIR}/." "${MODEL_RUN_DIR}/"
fi
if [ -L "${MODEL_RUN_DIR}/config.json" ]; then
  rm -f "${MODEL_RUN_DIR}/config.json"
  cp "${MODEL_DIR}/config.json" "${MODEL_RUN_DIR}/config.json"
fi

python - <<PY
import json
from pathlib import Path

config_path = Path("${MODEL_RUN_DIR}") / "config.json"
with config_path.open() as f:
    cfg = json.load(f)

cfg["max_position_embeddings"] = 16384
cfg["sliding_window"] = 16384
cfg["rope_theta"] = 40000

with config_path.open("w") as f:
    json.dump(cfg, f, indent=2, sort_keys=True)
    f.write("\\n")
PY

if [ ! -f "${PKG_DATA_DIR}/AIME24/test.parquet" ] || [ ! -f "${PKG_DATA_DIR}/AMC23/test.parquet" ] || [ ! -f "${PKG_DATA_DIR}/MATH-500/test.parquet" ]; then
  (
    cd "${PKG_DATA_DIR}"
    python preprocess.py
  )
fi

python "${TASK_DIR}/scripts/make_shared_splits.py" \
  --train-input "${OPENR1_PATH}" \
  --aime24-input "${EVAL_AIME24_PATH}" \
  --amc23-input "${EVAL_AMC23_PATH}" \
  --math500-input "${EVAL_MATH500_PATH}" \
  --out-dir "${SPLIT_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --train-source-spec "${TRAIN_SOURCE_SPEC}" \
  --train-source-selection "${TRAIN_SOURCE_SELECTION}" \
  --aime24-size "${AIME24_SIZE}" \
  --amc23-size "${AMC23_SIZE}" \
  --math500-size "${MATH500_SIZE}" \
  --internal-aime24-size "${INTERNAL_AIME24_SIZE}" \
  --internal-amc23-size "${INTERNAL_AMC23_SIZE}" \
  --internal-math500-size "${INTERNAL_MATH500_SIZE}" \
  --seed "${SEED_VALUE}"

python - <<PY
train_size = int("${TRAIN_SIZE}")
train_batch_size = int("${TRAIN_BATCH_SIZE}")
total_training_steps = int("${TOTAL_TRAINING_STEPS}")
verify_n = int("${VERIFY_N}")
actual_updates = max(total_training_steps - 1, 1)

prompt_passes = train_batch_size * actual_updates / train_size
verify_exposure = prompt_passes * verify_n

print(
    "Derived training coverage: "
    f"optimizer_updates={actual_updates} "
    f"prompt_passes_per_sample={prompt_passes:.3f} "
    f"verify_exposure_per_sample={verify_exposure:.3f}",
    flush=True,
)
PY

ray stop --force >/dev/null 2>&1 || true

cd "${PKG_DIR}/hpt/verl"

python -m verl.mix_src.main_mix_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.grpo_use_std=False \
  data.train_files="${SPLIT_DIR}/mixed_train.parquet" \
  data.val_files="[\"${SPLIT_DIR}/AIME24_eval.parquet\",\"${SPLIT_DIR}/AMC23_eval.parquet\",\"${SPLIT_DIR}/MATH-500_eval.parquet\"]" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  ++data.seed="${SEED_VALUE}" \
  actor_rollout_ref.model.path="${MODEL_RUN_DIR}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  ++actor_rollout_ref.actor.data_loader_seed="${SEED_VALUE}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.offline_loss_type=sft \
  actor_rollout_ref.actor.off_policy_normalize=False \
  actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
  actor_rollout_ref.actor.sft_loss_coef=0.3 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_remove_token_mean=True \
  actor_rollout_ref.actor.loss_remove_clip=True \
  actor_rollout_ref.ref.use_ref=False \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${LOG_PROB_USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  ++actor_rollout_ref.rollout.seed="${SEED_VALUE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.n_verify="${VERIFY_N}" \
  actor_rollout_ref.rollout.n_val="${VAL_N}" \
  actor_rollout_ref.rollout.val_temperature=0.6 \
  +actor_rollout_ref.rollout.val_top_p=0.95 \
  ++actor_rollout_ref.rollout.val_seed="${SEED_VALUE}" \
  ++actor_rollout_ref.rollout.val_kwargs.seed="${SEED_VALUE}" \
  actor_rollout_ref.rollout.max_prefix_len="${ROLLOUT_MAX_PREFIX_LEN}" \
  trainer.logger=[console] \
  trainer.project_name=mlsbench_hpt \
  trainer.experiment_name="shared_seed${SEED_VALUE}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.unify_strategy=switch \
  trainer.switch_gate=0 \
  trainer.switch_gate_off=0 \
  trainer.remove_sfted_data=False \
  trainer.max_optim_to_keep=1 \
  trainer.default_hdfs_dir=null \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  +trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  data.reward_impl_version=6 \
  2>&1 | tee "${TRAIN_LOG}"
python - <<PY | tee "${FINAL_METRICS}" | tee -a "${TRAIN_LOG}"
import re
from pathlib import Path

log_path = Path("${TRAIN_LOG}")
text = log_path.read_text()

metric_dict = None

for line in reversed(text.splitlines()):
    if "step:" not in line or "val/test_score/" not in line:
        continue
    pairs = {
        key: float(value)
        for key, value in re.findall(
            r"(val/test_score/(?:AIME24|AMC23|MATH-500)):([0-9.]+)",
            line,
        )
    }
    if len(pairs) == 3:
        metric_dict = pairs
        break

if metric_dict is None:
    extracted = {}
    for short_key in ("AIME24", "AMC23", "MATH-500"):
        values = re.findall(
            rf"val/test_score/{re.escape(short_key)}['\"]?:\s*([0-9.]+)",
            text,
        )
        if values:
            extracted[f"val/test_score/{short_key}"] = float(values[-1])
    if len(extracted) == 3:
        metric_dict = extracted

if metric_dict is None:
    raise RuntimeError(f"Could not find final validation metrics in {log_path}")

print(f"Final validation metrics: {metric_dict}")
PY
