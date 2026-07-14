#!/usr/bin/env bash
# Internal multi-host sweep launcher (tmux/ssh across the lab's DSW machines) —
# not needed for external reproduction (see REPRODUCE.md). Externally, run
# scripts/run_formal_sweep.py directly on your own GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DEFAULT="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$HARNESS_DEFAULT")}"
HARNESS_ROOT="$PROJECT_ROOT/autoresearch_idea_harness"
RUN_ROOT="${RUN_ROOT:-$HARNESS_ROOT/runs/formal_sweeps/v2_3_mls10_modules9}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHARD_COUNT="${SHARD_COUNT:-32}"
SESSION_PREFIX="${SESSION_PREFIX:-v23s}"

launch_local() {
  local shard="$1"
  local gpu="$2"
  local session="${SESSION_PREFIX}_${shard}"
  local log="$RUN_ROOT/shard_logs/shard_${shard}.log"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "skip existing local session $session"
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$PROJECT_ROOT' && LC_ALL=C LANG=C CUDA_VISIBLE_DEVICES='$gpu' '$PYTHON_BIN' autoresearch_idea_harness/scripts/run_formal_sweep.py --mode formal --tasks all --modules all --n-samples 20 --experts opus47,gpt55 --worker-mode claude --gpu '$gpu' --worker-timeout 7200 --result-wait-timeout 7200 --no-eval-wait-timeout 180 --max-turns 30 --max-master-advice 1 --temperature 0.7 --shard-index '$shard' --shard-count '$SHARD_COUNT' --root '$RUN_ROOT' >> '$log' 2>&1"
  echo "started local $session gpu=$gpu shard=$shard"
}

launch_remote() {
  local host="$1"
  local shard="$2"
  local gpu="$3"
  local session="${SESSION_PREFIX}_${shard}"
  local log="$RUN_ROOT/shard_logs/shard_${shard}.log"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "mkdir -p '$RUN_ROOT/shard_logs' && if command -v tmux >/dev/null 2>&1; then if tmux has-session -t '$session' 2>/dev/null; then echo 'skip existing $host $session'; else tmux new-session -d -s '$session' \"cd '$PROJECT_ROOT' && LC_ALL=C LANG=C CUDA_VISIBLE_DEVICES='$gpu' '$PYTHON_BIN' autoresearch_idea_harness/scripts/run_formal_sweep.py --mode formal --tasks all --modules all --n-samples 20 --experts opus47,gpt55 --worker-mode claude --gpu '$gpu' --worker-timeout 7200 --result-wait-timeout 7200 --no-eval-wait-timeout 180 --max-turns 30 --max-master-advice 1 --temperature 0.7 --shard-index '$shard' --shard-count '$SHARD_COUNT' --root '$RUN_ROOT' >> '$log' 2>&1\"; echo 'started $host $session gpu=$gpu shard=$shard'; fi; else pidfile='$RUN_ROOT/shard_logs/shard_${shard}.pid'; if [ -f \"\$pidfile\" ] && kill -0 \$(cat \"\$pidfile\") 2>/dev/null; then echo 'skip existing $host shard_${shard} pid='\"\$(cat \"\$pidfile\")\"; else setsid -f bash -c 'cd \"$PROJECT_ROOT\" && exec env LC_ALL=C LANG=C CUDA_VISIBLE_DEVICES=\"$gpu\" PYTHONUNBUFFERED=1 \"$PYTHON_BIN\" autoresearch_idea_harness/scripts/run_formal_sweep.py --mode formal --tasks all --modules all --n-samples 20 --experts opus47,gpt55 --worker-mode claude --gpu \"$gpu\" --worker-timeout 7200 --result-wait-timeout 7200 --no-eval-wait-timeout 180 --max-turns 30 --max-master-advice 1 --temperature 0.7 --shard-index \"$shard\" --shard-count \"$SHARD_COUNT\" --root \"$RUN_ROOT\" >> \"$log\" 2>&1' < /dev/null >/dev/null 2>&1; sleep 0.5; pgrep -f 'shard-index $shard --shard-count $SHARD_COUNT' | head -1 > \"\$pidfile\" || true; echo 'started $host shard_${shard} gpu=$gpu pid='\"\$(cat \"\$pidfile\" 2>/dev/null)\"; fi; fi"
}

mkdir -p "$RUN_ROOT/shard_logs"

for shard_num in $(seq 0 31); do
  shard="$(printf '%02d' "$shard_num")"
  if (( shard_num < 8 )); then
    launch_remote "lxh_agent_0" "$shard" "$shard_num"
  elif (( shard_num < 16 )); then
    launch_local "$shard" "$((shard_num - 8))"
  elif (( shard_num < 24 )); then
    launch_remote "lxh_agent_2" "$shard" "$((shard_num - 16))"
  else
    launch_remote "lxh_agent_3" "$shard" "$((shard_num - 24))"
  fi
done
