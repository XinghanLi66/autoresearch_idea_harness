#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
cd "${WORK_ROOT}"

TASK_DIR="${TASK_DIR:-${WORK_ROOT}/_task}"
PKG_DIR="${PKG_DIR:-${WORK_ROOT}/Unify-Post-Training}"
DATA_DIR="${DATA_DIR:-/root/data}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen2.5-Math-1.5B}"
SEED_VALUE="${SEED:-42}"
export PYTHONHASHSEED="${SEED_VALUE}"
SOURCE_NAME="${ENV}"
RUN_ROOT="${OUTPUT_DIR:-/tmp/mlsbench_hpt}/${SOURCE_NAME}"

mkdir -p "${RUN_ROOT}"

python3 "${TASK_DIR}/scripts/make_source_split.py" \
  --input "${DATA_DIR}/openr1.parquet" \
  --source "${SOURCE_NAME}" \
  --train-out "${RUN_ROOT}/train.parquet" \
  --val-out "${RUN_ROOT}/val.parquet" \
  --train-size 256 \
  --val-size 64 \
  --seed "${SEED_VALUE}"

ray stop --force >/dev/null 2>&1 || true

cd "${PKG_DIR}/hpt/verl"

python3 -m verl.mix_src.main_mix_ppo \
  algorithm.adv_estimator=grpo_split \
  algorithm.grpo_use_std=False \
  data.train_files="${RUN_ROOT}/train.parquet" \
  data.val_files="${RUN_ROOT}/val.parquet" \
  data.train_batch_size=4 \
  data.val_batch_size=64 \
  data.max_prompt_length=1024 \
  data.max_response_length=1536 \
  ++data.seed="${SEED_VALUE}" \
  actor_rollout_ref.model.path="${MODEL_DIR}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  ++actor_rollout_ref.actor.data_loader_seed="${SEED_VALUE}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.offline_loss_type=off_sft \
  actor_rollout_ref.actor.off_policy_normalize=False \
  actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
  actor_rollout_ref.actor.sft_loss_coef=1.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.loss_remove_token_mean=True \
  actor_rollout_ref.actor.loss_remove_clip=True \
  actor_rollout_ref.ref.use_ref=False \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=0 \
  ++actor_rollout_ref.rollout.seed="${SEED_VALUE}" \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_verify=8 \
  actor_rollout_ref.rollout.n_val=1 \
  +actor_rollout_ref.rollout.val_top_p=1.0 \
  ++actor_rollout_ref.rollout.val_seed="${SEED_VALUE}" \
  ++actor_rollout_ref.rollout.val_kwargs.seed="${SEED_VALUE}" \
  trainer.logger=[console] \
  trainer.project_name=mlsbench_hpt \
  trainer.experiment_name="${SOURCE_NAME}_seed${SEED_VALUE}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=0 \
  trainer.test_freq=4 \
  trainer.unify_strategy=switch \
  trainer.switch_gate=0 \
  trainer.switch_gate_off=0 \
  trainer.remove_sfted_data=False \
  trainer.max_optim_to_keep=2 \
  trainer.default_hdfs_dir=null \
  trainer.total_training_steps=12 \
  trainer.default_local_dir="${RUN_ROOT}/checkpoints" \
  +trainer.val_before_train=True \
  data.reward_impl_version=6
