#!/usr/bin/env bash
# Replicable failure-mode ablation for the V3 researcher-CoT → MLS closed-loop eval.
#
# Isolates each locus of the observed failure (SFT/RL do not beat base on MLS pass-rate):
#   Locus 1  proposal quality     -> judge_mls_proposals.py (opus rubric) + frontier proposer
#   Locus 2a worker faithfulness   -> inspect editable_region.py vs proposal (manual/grep)
#   Locus 2b worker prompt too tight-> run_mls_eval_3arm.py --free-hparams
#   Locus 2c proposal is the lever -> frontier-proposal arm through the SAME constrained worker
#   Locus 3  training sub-cause     -> proposal-judge deltas base vs sft vs rl (+ RL reward diag)
#
# GPU-heavy proposal generation runs on QS (queue 532, GB200); the worker-MLS eval runs locally
# (MLS-Bench + Claude worker are local). Every stage is idempotent/cached.
#
# Usage: bash scripts/ablation_failure_mode.sh <stage>
#   stages: gen-qs | pull | frontier | judge | worker | worker-freehp | all
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python
EVAL=runs/researcher_cot/mls_eval
PROPS=$EVAL/proposals
SUBSET="dl_lr_schedule,cv_pooling_aggregation,cv_multitask_loss,dl_residual_connection"
STAGE="${1:-all}"

# --- Stage 1: generate proposals from all 3 checkpoints on QS (base/sft/rl) ---
gen_qs() {
  # packets must already be staged to /mnt/3fs (see prepare_v3_qs_data_stage_chunks.py --file packets.jsonl)
  # submit generate_mls_proposals_3arm.py in a QS trial (see runs/qs_mls_eval/gen_cmd.sh for the exact
  # pod command: clones V3, loads base + checkpoint-101 + GRPO adapter, writes proposals_{base,sft,rl}.jsonl)
  echo "[gen-qs] submit the QS proposal-gen trial (template: runs/qs_mls_eval/gen_cmd.sh)"
  QS_COMMAND=$(cat runs/qs_mls_eval/gen_cmd.sh)
  qs training create --name v3_qs_mls_gen_3arm --image "$(grep -oE 'artifactory[^ ]+tutu_cybertron:[^ ]+' configs/default.yaml | head -1)" \
    --queue-id 532 --cloud-id 12 --cluster-id 70 --resource-package-id 234 \
    --job-type PytorchJob --worker-num 1 --priority 0 --yes -o json -q --command "$QS_COMMAND"
}

# --- pull proposals from /mnt/3fs back to local (base64 via a tiny QS trial; see checkins) ---
pull() { echo "[pull] see runs/qs_mls_eval pull step; decode into $PROPS/proposals_{base,sft,rl}.jsonl"; }

# --- Stage 1b: frontier (ceiling) proposer via a reachable frontier API (opus-4-6 here) ---
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
# pick the first reachable frontier key/model (rotate as availability changes)
CANDS=[("RUNWAY_OPUS48_API_KEY","claude-opus-4-8"),("RUNWAY_OPUS46_API_KEY","claude-opus-4-6")]
cfg=load_config("configs/default.yaml"); client=None; model=None
for ke,m in CANDS:
    if not os.environ.get(ke): continue
    try:
        c=RunwayClient(cfg,key_env=ke); c.complete(endpoint="google_anthropic",model=m,messages=[{"role":"user","content":"ok"}],temperature=None,max_tokens=5,stream=False)
        client,model=c,m; break
    except Exception: continue
assert client, "no reachable frontier key"
pk=[json.loads(l) for l in open("runs/researcher_cot/mls_eval/packets.jsonl")]
out=open(sys.argv[1],"w")
for i,p in enumerate(pk,1):
    for a in range(5):
        try: r=client.complete(endpoint="google_anthropic",model=model,messages=p["messages"],temperature=None,max_tokens=1600,stream=False); break
        except Exception: time.sleep(3*(a+1))
    out.write(json.dumps({"arm":"frontier","task":p["task"],"subtask":p.get("subtask"),"pass_metric":p.get("pass_metric"),"proposal":r.text.strip(),"proposer_model":model},ensure_ascii=False)+"\n"); out.flush()
    print("frontier",i,p["task"],flush=True)
PYEOF
}

# --- Stage: Locus 1 proposal-quality judge (base/sft/rl/frontier) ---
judge() {
  $PY scripts/judge_mls_proposals.py \
    --proposals $PROPS/proposals_base.jsonl $PROPS/proposals_sft.jsonl \
                $PROPS/proposals_rl.jsonl $PROPS/proposals_frontier.jsonl \
    --out $EVAL/proposal_judge.jsonl --concurrency 6
}

# --- Stage: closed-loop worker eval (constrained). 3 arms x 10 tasks, or add frontier ---
worker() {
  $PY scripts/run_mls_eval_3arm.py --proposals-dir $PROPS \
    --arms base,sft,rl,frontier --max-turns 30 --worker-timeout 4800 --gpu 0 \
    --out-root $EVAL/worker_runs
}

# --- Stage: Locus 2b worker prompt-restrictiveness (free-hparams) on the subset ---
worker_freehp() {
  $PY scripts/run_mls_eval_3arm.py --proposals-dir $PROPS \
    --arms rl --tasks "$SUBSET" --max-turns 45 --worker-timeout 5400 --gpu 0 --free-hparams \
    --out-root $EVAL/worker_runs_freehp
}

case "$STAGE" in
  gen-qs) gen_qs ;;
  pull) pull ;;
  frontier) frontier ;;
  judge) judge ;;
  worker) worker ;;
  worker-freehp) worker_freehp ;;
  all) frontier; judge; worker; worker_freehp ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
