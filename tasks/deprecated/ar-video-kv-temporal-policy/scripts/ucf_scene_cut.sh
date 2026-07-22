#!/usr/bin/env bash
# UCF-101 short-prediction with tight memory budget (simulates scene-cut pressure).
# Runs from /workspace/FAR (package workdir); task mounted at /workspace/_task.
set -euo pipefail

python3 custom_video_eval.py \
    --workload ucf101_short_prediction \
    --budget tight_history_budget \
    --seed "${SEED:-42}"
