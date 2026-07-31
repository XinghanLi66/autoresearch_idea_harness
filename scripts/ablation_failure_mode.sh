#!/usr/bin/env bash
# MLS-Bench-Lite closed-loop eval + failure-mode ablation (native mlsbench CLI).
#
# Two-stage loop, GitHub MLS-Bench only (HF deprecated):
#   proposals from our checkpoints  ->  worker IMPLEMENTS them via `mlsbench agent --mode eng`
#   -> native scoring on each task's leaderboard.csv -> Δ-over-baseline (multi-seed).
#
# Stages:
#   gen-qs    generate proposals from base/sft/rl checkpoints on QS  (generate_mls_proposals_3arm.py)
#   frontier  frontier-proposer ceiling arm (opus-4.8 -> fable-5 -> opus-4.6, first reachable)
#   judge     opus rubric judge of the proposals                     (judge_mls_proposals.py)
#   eval      native MLS-Bench-Lite eval over the official 30 tasks  (run_mls_lite_eval.py)
#   all       frontier; judge; eval
#
# The native eval env (docker images per package, data, provider routing) is provisioned by cc002 —
# see agent-memory/coder/mls_lite_eval_env_request.md. Use `eval` with EXTRA="--dry-run" to preview.
#
# Usage: bash scripts/ablation_failure_mode.sh <stage> [ARMS]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python
EVAL=runs/researcher_cot/mls_eval
PROPS=$EVAL/proposals
LITE=docs/eval/mls_bench_lite_tasks.json
ARMS="${2:-base,sft,rl,frontier}"
STAGE="${1:-all}"
EXTRA="${EXTRA:-}"

gen_qs() {
  echo "[gen-qs] submit the QS proposal-gen trial (template: runs/qs_mls_eval/gen_cmd.sh)"
  QS_COMMAND=$(cat runs/qs_mls_eval/gen_cmd.sh)
  qs training create --name v3_qs_mls_gen_3arm \
    --image "$(grep -oE 'artifactory[^ ]+tutu_cybertron:[^ ]+' configs/default.yaml | head -1)" \
    --queue-id 532 --cloud-id 12 --cluster-id 70 --resource-package-id 234 \
    --job-type PytorchJob --worker-num 1 --priority 0 --yes -o json -q --command "$QS_COMMAND"
}

# frontier (ceiling) proposer via first reachable frontier key
frontier() {
  $PY - "$PROPS/proposals_frontier.jsonl" <<'PYEOF'
import json,os,sys,pathlib,time
sys.path.insert(0,"src")
for line in pathlib.Path(".env").read_text().splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ.setdefault(k,v.strip().strip('"').strip("'"))
from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.runway_client import RunwayClient
CANDS=[("RUNWAY_OPUS48_API_KEY","claude-opus-4-8"),("RUNWAY_FABLE5_API_KEY","claude-fable-5"),
       ("RUNWAY_OPUS46_API_KEY","claude-opus-4-6")]
cfg=load_config("configs/default.yaml"); client=None; model=None
for ke,m in CANDS:
    if not os.environ.get(ke): continue
    try:
        c=RunwayClient(cfg,key_env=ke); c.complete(endpoint="google_anthropic",model=m,
            messages=[{"role":"user","content":"ok"}],temperature=None,max_tokens=8,stream=False)
        client,model=c,m; break
    except Exception: continue
assert client, "no reachable frontier key"
pk=[json.loads(l) for l in open("runs/researcher_cot/mls_eval/packets.jsonl")]
out=open(sys.argv[1],"w")
for i,p in enumerate(pk,1):
    for a in range(5):
        try: r=client.complete(endpoint="google_anthropic",model=model,messages=p["messages"],
                temperature=None,max_tokens=1600,stream=False); break
        except Exception: time.sleep(3*(a+1))
    out.write(json.dumps({"arm":"frontier","task":p["task"],"subtask":p.get("subtask"),
        "proposal":r.text.strip(),"proposer_model":model},ensure_ascii=False)+"\n"); out.flush()
    print("frontier",i,p["task"],flush=True)
PYEOF
}

judge() {
  $PY scripts/judge_mls_proposals.py \
    --proposals $PROPS/proposals_base.jsonl $PROPS/proposals_sft.jsonl \
                $PROPS/proposals_rl.jsonl $PROPS/proposals_frontier.jsonl \
    --out $EVAL/proposal_judge.jsonl --concurrency 6
}

eval_lite() {
  $PY scripts/run_mls_lite_eval.py --lite-json "$LITE" --proposals-dir "$PROPS" \
    --arms "$ARMS" --seeds 3 --mode eng --out runs/researcher_cot/mls_lite_eval/results.json $EXTRA
}

case "$STAGE" in
  gen-qs) gen_qs ;;
  frontier) frontier ;;
  judge) judge ;;
  eval) eval_lite ;;
  all) frontier; judge; eval_lite ;;
  *) echo "unknown stage: $STAGE (gen-qs|frontier|judge|eval|all)"; exit 2 ;;
esac
