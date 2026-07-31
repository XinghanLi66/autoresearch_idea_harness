#!/bin/bash
# Relaunch Qwen Set1 (react50) for the 6 tasks that failed or had sparse results.
# Fixes applied before this relaunch:
#   1. API: now routes via litellm proxy instead of DashScope directly (no Arrearage)
#   2. Locale: LC_ALL unset in subprocess env (no infinite setlocale output on M0/M1)
#   3. Agent: consecutive-edit reminder injected after 3 edits without test()
# cv-sample-weighting is excluded — its rerun3 already produced 17 evals.
set -uo pipefail

cd "$(dirname "$0")/.."

MODEL="qwen3.6-35b-a3b"
LOGDIR="logs/_experiments"
mkdir -p "$LOGDIR"

# Task → machine mapping (mirrors launch_qwen_all.sh)
# M0: optimization-parity, optimization-variance-reduction
# M1: optimization-bilevel, graph-signal-propagation
# M2 (local): causal-treatment-effect
# M3: ts-anomaly-detection

echo "[$(date +%T)] Relaunching Qwen Set1 (react50) reruns — litellm proxy routing"

# --- M0 ---
for TASK in optimization-parity optimization-variance-reduction; do
    echo "[$(date +%T)] [$TASK] → M0 (set1 only)"
    ssh lxh_agent_0 "cd /newcpfs/lxh/MLS-Bench && bash scripts/run_parity.sh '$TASK' '$MODEL' '1'" 2>/dev/null
    echo
done

# --- M1 (optimization-bilevel only — graph-signal-propagation moved to M2) ---
# graph-signal uses mlsbench-ChebNetII conda env; M1 has no conda so torch_scatter
# fails with an ABI mismatch (torch 2.11 vs scatter built against older API).
for TASK in optimization-bilevel; do
    echo "[$(date +%T)] [$TASK] → M1 (set1 only)"
    ssh lxh_agent_1 "cd /newcpfs/lxh/MLS-Bench && bash scripts/run_parity.sh '$TASK' '$MODEL' '1'" 2>/dev/null
    echo
done

# --- M2 (local) — causal-treatment-effect + graph-signal-propagation ---
# graph-signal needs mlsbench-ChebNetII which only works on M2 (has conda)
for TASK in causal-treatment-effect graph-signal-propagation; do
    echo "[$(date +%T)] [$TASK] → M2/local (set1 only)"
    bash scripts/run_parity.sh "$TASK" "$MODEL" "1"
    echo
done


# --- M3 ---
for TASK in ts-anomaly-detection; do
    echo "[$(date +%T)] [$TASK] → M3 (set1 only)"
    ssh lxh_agent_3 "cd /newcpfs/lxh/MLS-Bench && PATH=/newcpfs/lxh/miniconda3/bin:\$PATH bash scripts/run_parity.sh '$TASK' '$MODEL' '1'" 2>/dev/null
    echo
done

echo
echo "[$(date +%T)] All Set1 reruns launched. Monitor with:"
echo "  tail -F logs/_experiments/*set1*qwen*.log"
