#!/bin/bash
set -euo pipefail
# Large EDM2 UNet — FLOWMAPS_UNET_SIZE=large
export FLOWMAPS_UNET_SIZE="${FLOWMAPS_UNET_SIZE:-large}"
source "$(dirname "$0")/common.sh"
setup_bench_env
prepare_cifar10_stats

python py/launchers/learn.py \
  --cfg_path configs.cifar10_bench \
  --slurm_id "${FLOWMAPS_BENCH_SLURM_ID}" \
  --dataset_location "$DS" \
  --output_folder "${OUTPUT_DIR:-./output}"
