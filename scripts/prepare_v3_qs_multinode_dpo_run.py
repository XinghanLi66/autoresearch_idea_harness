#!/usr/bin/env python3
"""Prepare a multi-node FSDP DPO run — produces the M2-RL (235B) checkpoint at scale.

The single-worker 235B DPO was impossible (naive-MP OOM/crawl). This shards the merged 235B base
across N QS pods via FSDP (hybrid_shard) under torchrun's PET_* rendezvous over InfiniBand, so DPO's
chosen+rejected forwards fit comfortably. Uses the merged SFT base (base 235B + M2 SFT adapter) that
prior single-node attempts already wrote to /mnt/3fs; a fresh LoRA is DPO-trained on top.

Usage:
  python scripts/prepare_v3_qs_multinode_dpo_run.py --worker-num 8 [--max-length 1024]
  bash runs/qs_multinode_dpo/<run_id>/qs_create_dry_run.sh
  CONFIRM_SUBMIT_V3_QS_MULTINODE_DPO=1 bash runs/qs_multinode_dpo/<run_id>/qs_create_job.sh
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402

MERGED_DEFAULT = "/mnt/3fs/lxh/agentic-training/runs/qs_researcher_lora_train/m2_qwen3_235b_a22b_lora/merged_sft"
PAIRS_DEFAULT = "/mnt/3fs/lxh/agentic-training/data/researcher_cot/preference/pairs_all.jsonl"


def _safe_label(v: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", v).strip("_") or "qs_multinode_dpo"


def _find_harness_repo(qs_cfg: dict[str, Any]) -> dict[str, Any]:
    for repo in qs_cfg.get("git_repos") or []:
        if repo.get("name") == "autoresearch_idea_harness":
            return dict(repo)
    raise SystemExit("v3_training.qs.git_repos must include autoresearch_idea_harness")


def _command_text(run_id: str, qs_cfg: dict[str, Any], harness_repo: dict[str, Any], p: argparse.Namespace) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_dpo/{run_id}"
    out_dir = f"{remote_run_dir}/output"
    repo_url = str(harness_repo["url"])
    repo_ref = str(harness_repo.get("ref") or "V3")
    return f"""#!/usr/bin/env bash
set -uo pipefail
export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
mkdir -p "$REMOTE_RUN_DIR" {shlex.quote(out_dir)}; cd "$REMOTE_RUN_DIR"
HOST="$(hostname)"
LOG="$REMOTE_RUN_DIR/dpo_${{HOST}}.log"
CLONE="/tmp/aih_${{RUN_ID}}"
MERGED={shlex.quote(p.merged_base)}
PAIRS={shlex.quote(p.pairs)}
export NCCL_DEBUG=WARN
export NCCL_IB_HCA="${{NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF="${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
if [ -x /workspace/get_socket_name.sh ]; then export NCCL_SOCKET_IFNAME="$(/workspace/get_socket_name.sh)"; fi
echo "[dpo-mn] host=$HOST PET_NNODES=${{PET_NNODES:-?}} PET_NODE_RANK=${{PET_NODE_RANK:-?}} MASTER_ADDR=${{MASTER_ADDR:-?}} NCCL_IB_HCA=$NCCL_IB_HCA" | tee "$LOG"
# merged 235B base must already exist (prior single-node merge wrote it); do NOT re-merge in multi-node (race).
python3 - "$MERGED" <<'PYCK' 2>&1 | tee -a "$LOG"
import sys, os, json, glob
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not (os.path.exists(os.path.join(d,"config.json")) and os.path.exists(idx)):
    print("[dpo-mn] FATAL merged base incomplete (no config/index):", d); raise SystemExit(3)
shards = set(json.load(open(idx))["weight_map"].values())
have = set(os.path.basename(x) for x in glob.glob(os.path.join(d,"*.safetensors")))
missing = shards - have
print(f"[dpo-mn] merged base shards {{len(have & shards)}}/{{len(shards)}} missing={{len(missing)}}")
raise SystemExit(0 if not missing else 3)
PYCK
[ ${{PIPESTATUS[0]}} -eq 0 ] || {{ echo "[dpo-mn] abort: merged base missing/incomplete $MERGED" | tee -a "$LOG"; exit 3; }}
test -f "$PAIRS" || {{ echo "[dpo-mn] FATAL pairs missing $PAIRS" | tee -a "$LOG"; exit 3; }}
rm -rf "$CLONE"
git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} "$CLONE" 2>&1 | tee -a "$LOG"
cd "$CLONE"
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head_${{HOST}}.txt"
python3 -m pip install -q -U "transformers>=4.51,<5" "trl>=0.12,<0.13" "peft>=0.15" "accelerate>=1.10.0,<2" 2>&1 | tail -3 | tee -a "$LOG" || true
python3 -m pip uninstall -y torchao apex 2>&1 | tail -2 | tee -a "$LOG" || true
python3 -c "import trl,transformers,peft; print('[dpo-mn] trl',trl.__version__,'tfm',transformers.__version__,'peft',peft.__version__)" | tee -a "$LOG" || true
# per-pod safety: early sharded saves so a late failure across a long run doesn't lose everything.
sed -i 's/save_strategy="epoch",/save_strategy="steps", save_steps={int(p.save_steps)}, save_total_limit=3,/' scripts/train_v3_researcher_cot_dpo.py
python3 -m py_compile scripts/train_v3_researcher_cot_dpo.py 2>&1 | tee -a "$LOG"
# torchrun reads PET_* natively; FSDP shards the 235B base (no device_map).
torchrun --nproc-per-node=4 scripts/train_v3_researcher_cot_dpo.py \
  --base-model "$MERGED" --pairs-jsonl "$PAIRS" \
  --loss-type sigmoid --beta 0.1 \
  --max-length {int(p.max_length)} --max-prompt-length {int(p.max_prompt_length)} \
  --per-device-batch {int(p.per_device_batch)} --grad-accum {int(p.grad_accum)} \
  --fsdp-sharding-strategy {shlex.quote(p.sharding_strategy)} \
  --fsdp-transformer-layer {shlex.quote(p.transformer_layer)} \
  --output-dir {shlex.quote(out_dir)} 2>&1 | tee -a "$LOG"
echo "[dpo-mn] done $(date -Is) host=$HOST" | tee -a "$LOG"
"""


def _script_text(*, run_id: str, qs_cfg: dict[str, Any], worker_num: int, command_path: Path, submit: bool) -> str:
    args = ["qs", "training", "create"]
    if not submit:
        args.append("--dry-run")
    args.extend([
        "--name", run_id, "--image", str(qs_cfg["image"]),
        "--queue-id", str(qs_cfg["queue_id"]), "--cloud-id", str(qs_cfg["cloud_id"]),
        "--cluster-id", str(qs_cfg["cluster_id"]), "--resource-package-id", str(qs_cfg["resource_package_id"]),
        "--job-type", str(qs_cfg.get("job_type", "PytorchJob")), "--worker-num", str(worker_num),
        "--priority", str(qs_cfg.get("priority", 0)),
    ])
    if qs_cfg.get("overuse", False):
        args.append("--overuse")
    args.extend(["--yes", "-o", "json", "-q"])
    prefix = f"QS_COMMAND=$(cat {shlex.quote(str(command_path))})"
    command = " ".join(shlex.quote(a) for a in args) + ' --command "$QS_COMMAND"'
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", prefix]
    if submit:
        lines.extend([
            ': "${CONFIRM_SUBMIT_V3_QS_MULTINODE_DPO:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_MULTINODE_DPO}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_MULTINODE_DPO must equal 1" >&2', "  exit 2", "fi",
            f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
        ])
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(config_path: Path, output_dir: Path, run_id: str | None, p: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = ["queue_id", "cloud_id", "cluster_id", "resource_package_id", "image", "remote_project_root", "git_repos"]
    missing = [k for k in required if not qs_cfg.get(k)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")
    if run_id is None:
        run_id = f"v3_qs_mn_dpo_{p.worker_num}w_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    harness_repo = _find_harness_repo(qs_cfg)
    command_path = run_dir / "qs_command.sh"
    command_path.write_text(_command_text(run_id, qs_cfg, harness_repo, p))
    command_path.chmod(0o755)
    (run_dir / "qs_create_dry_run.sh").write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, worker_num=p.worker_num, command_path=command_path, submit=False))
    (run_dir / "qs_create_dry_run.sh").chmod(0o755)
    (run_dir / "qs_create_job.sh").write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, worker_num=p.worker_num, command_path=command_path, submit=True))
    (run_dir / "qs_create_job.sh").chmod(0o755)
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_dpo/{run_id}"
    summary = {
        "run_id": run_id, "run_dir": str(run_dir),
        "purpose": f"M2-RL: multi-node FSDP DPO on merged 235B, {p.worker_num} workers x4 = {p.worker_num*4} GPU.",
        "worker_num": p.worker_num, "total_gpus_expected": p.worker_num * 4,
        "merged_base": p.merged_base, "sharding_strategy": p.sharding_strategy,
        "artifacts": {"command": str(command_path), "dry_run": str(run_dir / "qs_create_dry_run.sh"),
                      "submit": str(run_dir / "qs_create_job.sh"),
                      "expected_remote_output": f"{remote_run_dir}/output",
                      "expected_per_pod_logs": f"{remote_run_dir}/dpo_<hostname>.log"},
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_MULTINODE_DPO=1",
        "success_criteria": "loss/rewards healthy on all pods; DPO-LoRA adapter under output/checkpoint-* on /mnt/3fs.",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_multinode_dpo"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--worker-num", type=int, default=8, help="pods (each 4 GPU); 8 = 32 GPU")
    parser.add_argument("--merged-base", default=MERGED_DEFAULT, help="merged 235B (base+SFT adapter) dir on /mnt/3fs")
    parser.add_argument("--pairs", default=PAIRS_DEFAULT)
    parser.add_argument("--sharding-strategy", default="hybrid_shard", choices=["full_shard", "hybrid_shard"])
    parser.add_argument("--transformer-layer", default="Qwen3MoeDecoderLayer")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.config), Path(args.output_dir), args.run_id, args), indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
