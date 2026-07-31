#!/bin/bash
# Launch all 7-task qwen3.6-35b-a3b parity experiments (Settings 1+2+3)
# Uses DashScope API (no local vLLM needed)
# Distributes Set3 OE evals across M0-M3 GPUs via scheduler
set -uo pipefail
cd /newcpfs/lxh/MLS-Bench

MODEL="qwen3.6-35b-a3b"
CFG_RT="configs/config.qwen-litellm.yaml"
LOGDIR="logs/_experiments"
mkdir -p "$LOGDIR"

# Tasks distributed across machines
# M0: optimization-parity, optimization-variance-reduction (scheduler GPUs 0-7)
# M1: optimization-bilevel, graph-signal-propagation   (scheduler GPUs 0-7)
# M2 (local): cv-sample-weighting, causal-treatment-effect (scheduler GPUs 0-7)
# M3: ts-anomaly-detection                             (scheduler GPUs 0-7)

echo "[$(date +%T)] Launching qwen3.6-35b-a3b parity experiments across all machines"

# --- M0 ---
echo "[$(date +%T)] Starting scheduler on M0..."
ssh lxh_agent_0 "cd /newcpfs/lxh/MLS-Bench && nohup python -m mlsbench.scheduler start --gpus 0,1,2,3,4,5,6,7 \
    --config $CFG_RT --daemon \
    > $LOGDIR/scheduler_qwen_m0.log 2>&1 &
    echo scheduler_pid=\$!" 2>/dev/null
sleep 2

for TASK in optimization-parity optimization-variance-reduction; do
    echo "[$(date +%T)] Launching $TASK on M0"
    ssh lxh_agent_0 "cd /newcpfs/lxh/MLS-Bench && bash scripts/run_parity.sh '$TASK' '$MODEL' '1 2 3'" 2>/dev/null
    echo
done

# --- M1 ---
echo "[$(date +%T)] Starting scheduler on M1..."
ssh lxh_agent_1 "cd /newcpfs/lxh/MLS-Bench && nohup python -m mlsbench.scheduler start --gpus 0,1,2,3,4,5,6,7 \
    --config $CFG_RT --daemon \
    > $LOGDIR/scheduler_qwen_m1.log 2>&1 &
    echo scheduler_pid=\$!" 2>/dev/null
sleep 2

for TASK in optimization-bilevel graph-signal-propagation; do
    echo "[$(date +%T)] Launching $TASK on M1"
    ssh lxh_agent_1 "cd /newcpfs/lxh/MLS-Bench && bash scripts/run_parity.sh '$TASK' '$MODEL' '1 2 3'" 2>/dev/null
    echo
done

# --- M2 (local) ---
echo "[$(date +%T)] Starting scheduler on M2..."
nohup python -m mlsbench.scheduler start --gpus 0,1,2,3,4,5,6,7 \
    --config "$CFG_RT" --daemon \
    > "$LOGDIR/scheduler_qwen_m2.log" 2>&1 &
echo "  scheduler_pid=$!"
sleep 2

for TASK in cv-sample-weighting causal-treatment-effect; do
    echo "[$(date +%T)] Launching $TASK on M2"
    bash scripts/run_parity.sh "$TASK" "$MODEL" "1 2 3"
    echo
done

# --- M3 ---
echo "[$(date +%T)] Starting scheduler on M3..."
ssh lxh_agent_3 "cd /newcpfs/lxh/MLS-Bench && nohup python -m mlsbench.scheduler start --gpus 0,1,2,3,4,5,6,7 \
    --config $CFG_RT --daemon \
    > $LOGDIR/scheduler_qwen_m3.log 2>&1 &
    echo scheduler_pid=\$!" 2>/dev/null
sleep 2

for TASK in ts-anomaly-detection; do
    echo "[$(date +%T)] Launching $TASK on M3"
    ssh lxh_agent_3 "cd /newcpfs/lxh/MLS-Bench && bash scripts/run_parity.sh '$TASK' '$MODEL' '1 2 3'" 2>/dev/null
    echo
done

echo
echo "[$(date +%T)] All launches complete. Monitor with:"
echo "  tail -F logs/_experiments/*qwen*.log"
