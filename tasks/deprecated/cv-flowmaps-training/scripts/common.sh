#!/bin/bash
set -euo pipefail

# Loss family for configs.cifar10_bench (overridden per baseline via bench_env.sh from mid_edit / baseline edit_ops).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$_SCRIPT_DIR/bench_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_SCRIPT_DIR/bench_env.sh"
fi
export FLOWMAPS_BENCH_SLURM_ID="${FLOWMAPS_BENCH_SLURM_ID:-0}"

setup_bench_env() {
  cd /workspace/flow-maps
  export PYTHONPATH="/workspace/flow-maps/py:${PYTHONPATH:-}"
  export WANDB_DISABLED=true
  export TF_CPP_MIN_LOG_LEVEL=3
  export SEED="${SEED:-42}"
  export FLOWMAPS_UNET_SIZE="${FLOWMAPS_UNET_SIZE:-medium}"
  export BATCH_SIZE="${BATCH_SIZE:-128}"
  export MAX_STEPS="${MAX_STEPS:-50000}"
  export EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
  export NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
  export NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-8}"
  export DS="${DATASET_LOCATION:-/data/tfds}"
}

prepare_cifar10_stats() {
  mkdir -p "$DS" "${OUTPUT_DIR:-./output}" "$DS/cifar10"
  local stats="$DS/cifar10/cifar_stats.npz"
  if [ -f "$stats" ] && [ ! -s "$stats" ]; then
    echo "MLS-Bench: removing empty FID stats (will recompute): $stats"
    rm -f "$stats"
  fi
  if [ ! -f "$stats" ]; then
    echo "Computing CIFAR-10 FID reference stats -> $stats"
    python py/launchers/calc_dataset_fid_stats.py \
      --dataset_location "$DS" \
      --dataset cifar10 \
      --out "$stats" \
      --batch "${FID_STATS_BATCH:-256}"
  fi
}
