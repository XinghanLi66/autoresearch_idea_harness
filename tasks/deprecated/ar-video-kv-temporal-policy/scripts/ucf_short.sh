#!/usr/bin/env bash
# UCF-101 short-prediction workload, medium history budget.
# Runs from /workspace/FAR (package workdir); task mounted at /workspace/_task.
# custom_video_eval.py is placed at FAR/ by the mid_edit stage.
set -euo pipefail

python3 custom_video_eval.py \
    --workload ucf101_short_prediction \
    --budget medium_history_budget \
    --seed "${SEED:-42}"

