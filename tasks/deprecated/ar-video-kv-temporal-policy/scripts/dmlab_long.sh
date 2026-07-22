#!/usr/bin/env bash
# DMLab long-horizon continuation workload, tight history budget.
# Runs from /workspace/FAR (package workdir); task mounted at /workspace/_task.
# custom_video_eval.py is placed at FAR/ by the mid_edit stage.
set -euo pipefail

python3 custom_video_eval.py \
    --workload dmlab_long_prediction \
    --budget tight_history_budget \
    --seed "${SEED:-42}"

