#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/workspace}"
cd "${WORK_ROOT}"

SEED_VALUE="${SEED:-42}"
# Match RUN_ROOT used by train_shared_hpt.sh (/tmp shared via apptmp bind).
RUN_ROOT="/tmp/mlsbench_hpt_shared"
FINAL_METRICS="${RUN_ROOT}/final_metrics.txt"
WAIT_SECS="${WAIT_SECS:-9000}"
SLEEP_SECS="${SLEEP_SECS:-15}"
ENV_NAME="${ENV:-""}"

deadline=$(( $(date +%s) + WAIT_SECS ))
while [ ! -s "${FINAL_METRICS}" ]; do
  now=$(date +%s)
  if [ "${now}" -ge "${deadline}" ]; then
    echo "ERROR: final metrics file not found after waiting ${WAIT_SECS}s: ${FINAL_METRICS}" >&2
    echo "Hint: AMC23/MATH-500 relay commands depend on the AIME24 group-1 training command completing first with the same OUTPUT_DIR." >&2
    exit 1
  fi
  sleep "${SLEEP_SECS}"
done

python - <<PY
import ast
import re
from pathlib import Path

env_name = "${ENV_NAME}"
metric_key = f"val/test_score/{env_name}"
metrics_path = Path("${FINAL_METRICS}")

lines = metrics_path.read_text(errors='ignore').splitlines()
for line in reversed(lines):
    if 'Final validation metrics' not in line:
        continue

    payload = line.split(':', 1)[-1].strip()
    if payload.startswith('(') and payload.endswith(')'):
        payload = payload[1:-1]
    if payload.startswith('"') and payload.endswith('"'):
        payload = payload[1:-1]

    try:
        start = payload.index('{')
        end = payload.rindex('}') + 1
        payload_dict = payload[start:end]
        metrics = ast.literal_eval(payload_dict)
    except Exception:
        continue

    score = metrics.get(metric_key)
    if score is not None:
        payload = {metric_key: float(score)}
        print(f"Final validation metrics: {payload}")
        break
else:
    # fallback: parse raw step log line like val/test_score/ABC:0.123
    for line in reversed(lines):
        m = re.search(rf"val/test_score/{re.escape(env_name)}:([0-9]+(?:\.[0-9]+)?)", line)
        if m:
            payload = {f"val/test_score/{env_name}": float(m.group(1))}
            print(f"Final validation metrics: {payload}")
            break
    else:
        raise RuntimeError(f"Could not find metric for env={env_name} in {metrics_path}")
PY
