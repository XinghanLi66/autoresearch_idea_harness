#!/usr/bin/env python3
"""Generate QS command + create-job scripts for the V3 model-zoo SFT arms.

Full-FT arms (S1/S2/S3): ModelScope weights + transformers 5.5.0 + enable_thinking=False
patch + full 1-epoch train on the anchored 1626/85 set + a real HF final checkpoint.
LoRA arms (M1): same deps + peft (MoE target) + LoRA trainer.

Reuses the proven recipe validated by the s1 smoke (trial 1708978).
"""
from __future__ import annotations
import os
import stat
from pathlib import Path

ROOT = Path("/newcpfs/lxh/agentic-training/autoresearch_idea_harness")
REMOTE_BASE = "/mnt/3fs/lxh/agentic-training"
IMAGE = "artifactory.devops.xiaohongshu.com/quicksilver/images/training/media/nvidia/tutu_cybertron:ngc2510_xray_newep_3fs_v1"
TRAIN = f"{REMOTE_BASE}/data/researcher_cot/anchored_v1/train.jsonl"
VAL = f"{REMOTE_BASE}/data/researcher_cot/anchored_v1/val.jsonl"
TRAIN_SHA = "06ce0a9b40512678"
VAL_SHA = "da29cef4dfd950bd"
GIT = "https://github.com/XinghanLi66/autoresearch_idea_harness.git"

# per-arm: id, kind, modelscope tag, lr, epochs(-> max-steps -1 = 1 epoch; loop epochs via --num-epochs if supported)
FULL_ARMS = [
    dict(rid="s1_qwen3_8b_full",  tag="Qwen/Qwen3-8B",  lr="5e-06"),
    dict(rid="s2_qwen3_14b_full", tag="Qwen/Qwen3-14B", lr="3e-06"),
    dict(rid="s3_qwen3_32b_full", tag="Qwen/Qwen3-32B", lr="2e-06"),
    # cc001 idle-soak lr-sweep (priority-0 backfill) — new lr points only (dups of S1/S2/S3 skipped)
    dict(rid="zoo_qwen3_8b_lr1e5",  tag="Qwen/Qwen3-8B",  lr="1e-05"),
    dict(rid="zoo_qwen3_14b_lr6e6", tag="Qwen/Qwen3-14B", lr="6e-06"),
    dict(rid="zoo_qwen3_32b_lr4e6", tag="Qwen/Qwen3-32B", lr="4e-06"),
]
LORA_ARMS = [
    dict(rid="m1_qwen3_30b_a3b_lora", tag="Qwen/Qwen3-30B-A3B", lr="1e-04", epochs=2),
]

COMMON_HEAD = r'''#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={rid}
export REMOTE_RUN_DIR={remote_run_dir}
export HF_HOME="${{HF_HOME:-{base}/hf_cache}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-$HF_HOME/transformers}}"
export HF_HUB_ENABLE_HF_TRANSFER="${{HF_HUB_ENABLE_HF_TRANSFER:-0}}"
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{cvd}}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${{TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}}"
export NCCL_DEBUG="${{NCCL_DEBUG:-WARN}}"
export PYTORCH_ALLOC_CONF="${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
export TORCHRUN_BIN="${{TORCHRUN_BIN:-torchrun}}"

mkdir -p "$REMOTE_RUN_DIR/src" "$HF_HOME"
cd "$REMOTE_RUN_DIR"
echo "[zoo] start $(date -Is) run_id=$RUN_ID host=$(hostname)" | tee run.log
echo "[zoo] uname=$(uname -a)" | tee -a run.log

select_nccl_iface() {{
  local iface path state
  for path in /sys/class/net/*; do
    [[ -e "$path" ]] || continue
    iface=$(basename "$path")
    case "$iface" in lo|docker*|veth*|cni*|flannel*|tun*|tap*) continue ;; esac
    state=$(cat "$path/operstate" 2>/dev/null || true)
    if [[ "$state" == "up" || "$state" == "unknown" ]]; then printf '%s\n' "$iface"; return 0; fi
  done
  printf 'lo\n'
}}
export NCCL_SOCKET_IFNAME="$(select_nccl_iface)"
export GLOO_SOCKET_IFNAME="${{GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}}"
unset NCCL_ASYNC_ERROR_HANDLING || true
unset PYTORCH_CUDA_ALLOC_CONF || true
echo "[zoo] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" | tee -a run.log

# --- deps for Qwen3 (transformers 4.51+ adds Qwen3 dense+MoE while keeping the 4.x
#     TrainingArguments API the custom HF-Trainer uses; 5.x drops save_safetensors etc.){peft_note} ---
python3 -m pip install -q -U "transformers>=4.51,<5" "accelerate>=1.10.0,<2"{peft_pkg} 2>&1 | tail -4 | tee -a run.log || true
python3 -m pip uninstall -y torchao 2>&1 | tail -2 | tee -a run.log || true
python3 -c "import transformers; print('[zoo] transformers', transformers.__version__)" | tee -a run.log || true

# --- weights via ModelScope: completeness-GATED staging. A partial dir (from a killed run) makes
#     snapshot_download RESUME HANG (0% GPU forever), so: if full shard set present -> skip; else wipe + fresh pull. ---
MODEL_DIR={model_dir}
if python3 - "$MODEL_DIR" <<'PYCHK'
import sys, os, json, glob
d = sys.argv[1]
ok = os.path.exists(os.path.join(d, "config.json")) and (
    os.path.exists(os.path.join(d, "tokenizer.json")) or os.path.exists(os.path.join(d, "vocab.json")))
idx = os.path.join(d, "model.safetensors.index.json")
if ok and os.path.exists(idx):
    shards = set(json.load(open(idx))["weight_map"].values())
    have = set(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.safetensors")))
    ok = shards.issubset(have)
elif ok:
    ok = bool(glob.glob(os.path.join(d, "*.safetensors")) or glob.glob(os.path.join(d, "*.bin")))
sys.exit(0 if ok else 1)
PYCHK
then
  echo "[zoo] {tag} already fully staged -> $MODEL_DIR (skip download)" | tee -a run.log
else
  echo "[zoo] staging {tag} FRESH (wiping any partial to avoid resume-hang) -> $MODEL_DIR" | tee -a run.log
  rm -rf "$MODEL_DIR"
  python3 -m pip install -q -U modelscope 2>&1 | tail -3 | tee -a run.log || true
  python3 -c "from modelscope import snapshot_download; snapshot_download('{tag}', local_dir='$MODEL_DIR')" 2>&1 | tail -15 | tee -a run.log
fi
test -f "$MODEL_DIR/config.json"
{{ test -f "$MODEL_DIR/tokenizer.json" || test -f "$MODEL_DIR/vocab.json"; }} || {{ echo "[zoo] FATAL tokenizer files missing after download" | tee -a run.log; exit 3; }}

# --- data present + sha (guard vs stale/prejudge copy) ---
test -f {train}
test -f {val}
echo "[zoo] train_sha=$(sha256sum {train} | cut -c1-16) (expect {train_sha})  val_sha=$(sha256sum {val} | cut -c1-16) (expect {val_sha})" | tee -a run.log

rm -rf "$REMOTE_RUN_DIR/src/autoresearch_idea_harness"
git clone --depth 1 --branch V3 {git} "$REMOTE_RUN_DIR/src/autoresearch_idea_harness" 2>&1 | tee -a run.log
cd "$REMOTE_RUN_DIR/src/autoresearch_idea_harness"
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head.txt"

# --- CRITICAL: Qwen3 non-thinking (no <think> vs Mocking/Core-idea anchors) ---
sed -i 's/add_generation_prompt=False,/add_generation_prompt=False,\n            enable_thinking=False,/' {trainer}
echo "[zoo] enable_thinking patch:" | tee -a "$REMOTE_RUN_DIR/run.log"
grep -n "enable_thinking\|apply_chat_template\|add_generation_prompt" {trainer} | tee -a "$REMOTE_RUN_DIR/run.log"
python3 -m py_compile {trainer} 2>&1 | tee -a "$REMOTE_RUN_DIR/run.log"
nvidia-smi 2>&1 | tee -a "$REMOTE_RUN_DIR/run.log"
'''

FULL_BODY = r'''
$TORCHRUN_BIN --standalone --nproc_per_node=4 --master_port=29517 \
  scripts/train_v3_researcher_cot_full_fsdp.py \
  --train-jsonl @@TRAIN@@ \
  --val-jsonl @@VAL@@ \
  --output-dir "$REMOTE_RUN_DIR/output" \
  --base-model "$MODEL_DIR" \
  --num-epochs 1 \
  --max-steps -1 \
  --max-seq-length 1664 \
  --per-device-batch-size 1 \
  --grad-accum 4 \
  --lr @@LR@@ \
  --fsdp-state-dict-type FULL_STATE_DICT \
  --fsdp-transformer-layer Qwen3DecoderLayer \
  --save-strategy steps \
  --save-steps 50 \
  --eval-steps 50 \
  --save-total-limit 2 \
  --summary-name train_summary.json 2>&1 | tee -a "$REMOTE_RUN_DIR/run.log"

python3 - <<'PY' "$REMOTE_RUN_DIR/result.json" "$REMOTE_RUN_DIR/output"
import json, sys
from pathlib import Path
result_path, out = Path(sys.argv[1]), Path(sys.argv[2])
summ = out / "train_summary.json"
summary = json.loads(summ.read_text()) if summ.exists() else None
ckpts = sorted(p.name for p in out.glob("checkpoint-*") if p.is_dir())
final_ok = (out / "config.json").exists() or bool(ckpts)
res = {"status": "ok" if (summary and final_ok) else "failed_validation",
       "output_dir": str(out), "checkpoint_dirs": ckpts,
       "final_checkpoint": final_ok, "summary": summary}
result_path.write_text(json.dumps(res, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(res, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if res["status"] == "ok" else 2)
PY
echo "[zoo] done $(date -Is)" | tee -a "$REMOTE_RUN_DIR/run.log"
'''

LORA_BODY = r'''
# MoE LoRA: attention-only target — "all-linear" targets all experts (30B-A3B: 3.38B trainable/9.97%, ~37h);
# q/k/v/o_proj keeps the adapter ~0.5% for fast, high-util voice/format steering.
sed -i 's/^LORA_TARGET_MODULES = "all-linear".*/LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]/' scripts/train_v3_researcher_cot_lora.py
grep -n "^LORA_TARGET_MODULES =" scripts/train_v3_researcher_cot_lora.py | tee -a "$REMOTE_RUN_DIR/run.log"
python3 scripts/train_v3_researcher_cot_lora.py \
  --train-jsonl @@TRAIN@@ \
  --val-jsonl @@VAL@@ \
  --output-dir "$REMOTE_RUN_DIR/output" \
  --base-model "$MODEL_DIR" \
  --num-epochs @@EPOCHS@@ \
  --max-steps -1 \
  --per-device-batch-size 1 \
  --grad-accum 4 \
  --max-seq-length 1664 \
  --lr @@LR@@ \
  --no-merge \
  --skip-gen-check 2>&1 | tee -a "$REMOTE_RUN_DIR/run.log"

python3 - <<'PY' "$REMOTE_RUN_DIR/result.json" "$REMOTE_RUN_DIR/output"
import json, sys
from pathlib import Path
result_path, out = Path(sys.argv[1]), Path(sys.argv[2])
summ = out / "train_summary.json"
summary = json.loads(summ.read_text()) if summ.exists() else None
adapters = [p.name for p in out.rglob("adapter_model.safetensors")]
res = {"status": "ok" if (summary and adapters) else "failed_validation",
       "output_dir": str(out), "adapters": adapters, "summary": summary}
result_path.write_text(json.dumps(res, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(res, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if res["status"] == "ok" else 2)
PY
echo "[zoo] done $(date -Is)" | tee -a "$REMOTE_RUN_DIR/run.log"
'''

CREATE = '''#!/usr/bin/env bash
set -euo pipefail
QS_COMMAND=$(cat {cmd_path})
: "${{CONFIRM_SUBMIT_V3_QS_ZOO:?Set to 1 after reviewing the command file.}}"
[[ "${{CONFIRM_SUBMIT_V3_QS_ZOO}}" == "1" ]] || {{ echo "CONFIRM_SUBMIT_V3_QS_ZOO must equal 1" >&2; exit 2; }}
qs training create --name {rid} --image {image} --queue-id 532 --cloud-id 12 --cluster-id 70 \\
  --resource-package-id 234 --job-type PytorchJob --worker-num {workers} --priority 0 --yes -o json -q \\
  --command "$QS_COMMAND" | tee {run_dir}/submission.json
'''


def chmodx(p: Path):
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def emit(rid, tag, lr, kind, run_root, trainer, body_tpl, cvd, workers, peft=False, epochs=1):
    run_dir = ROOT / run_root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_run_dir = f"{REMOTE_BASE}/{run_root}/{rid}"
    model_dir = f"{REMOTE_BASE}/models/{tag.split('/')[-1]}"
    head = COMMON_HEAD.format(
        rid=rid, remote_run_dir=remote_run_dir, base=REMOTE_BASE, cvd=cvd, tag=tag,
        model_dir=model_dir, train=TRAIN, val=VAL, train_sha=TRAIN_SHA, val_sha=VAL_SHA,
        git=GIT, trainer=trainer,
        peft_note=" + peft (MoE target)" if peft else "",
        peft_pkg=' "peft>=0.15.0"' if peft else "",
    )
    body = (body_tpl.replace("@@TRAIN@@", TRAIN).replace("@@VAL@@", VAL)
            .replace("@@LR@@", lr).replace("@@EPOCHS@@", str(epochs)))
    cmd_path = run_dir / "qs_command.sh"
    cmd_path.write_text(head + body)
    chmodx(cmd_path)
    create_path = run_dir / "qs_create_job.sh"
    create_path.write_text(CREATE.format(cmd_path=cmd_path, rid=rid, image=IMAGE,
                                         workers=workers, run_dir=run_dir))
    chmodx(create_path)
    print(f"{rid:26s} -> {cmd_path}")


for a in FULL_ARMS:
    emit(a["rid"], a["tag"], a["lr"], "full", "runs/qs_researcher_full_sft",
         "scripts/train_v3_researcher_cot_full_fsdp.py", FULL_BODY, cvd="0,1,2,3", workers=1)
for a in LORA_ARMS:
    emit(a["rid"], a["tag"], a["lr"], "lora", "runs/qs_researcher_lora_train",
         "scripts/train_v3_researcher_cot_lora.py", LORA_BODY, cvd="0", workers=1,
         peft=True, epochs=a["epochs"])
print("done")
