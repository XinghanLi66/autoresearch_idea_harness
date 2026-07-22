#!/bin/bash
set -euo pipefail
# Medium EDM2 UNet (default Flow Maps CIFAR-10 width)
export FLOWMAPS_UNET_SIZE="${FLOWMAPS_UNET_SIZE:-medium}"
source "$(dirname "$0")/common.sh"
setup_bench_env
prepare_cifar10_stats

python py/launchers/learn.py \
  --cfg_path configs.cifar10_bench \
  --slurm_id "${FLOWMAPS_BENCH_SLURM_ID}" \
  --dataset_location "$DS" \
  --output_folder "${OUTPUT_DIR:-./output}"
