#!/usr/bin/env python3
"""Prepare a multi-node FSDP training run/smoke (Phase 2/3 of the 64+ GPU effort).

Runs the existing FSDP full-SFT trainer (`scripts/train_v3_researcher_cot_full_fsdp.py`) across
`--worker-num N` QS pods (each 4 GPU) via torchrun's native PET_* rendezvous (Phase-0 finding), with
NCCL over InfiniBand (Phase-1 proven, ~128 GB/s busbw) and **HYBRID_SHARD** (shard within a node,
replicate across nodes). Code is cloned to node-local /tmp per pod (multi-pod /mnt/3fs clone race,
Phase-1 lesson); the sharded checkpoint is written to a shared /mnt/3fs output dir (SHARDED_STATE_DICT
= per-rank shard files, safe on shared fs).

Phase 2 default = 2 workers / 8 GPU, 32B dense, small --max-steps smoke. Scale via --worker-num 16
(64 GPU) for Phase 3.

Usage:
  python scripts/prepare_v3_qs_multinode_fsdp_run.py --worker-num 2 --max-steps 30
  bash runs/qs_multinode_fsdp/<run_id>/qs_create_dry_run.sh
  CONFIRM_SUBMIT_V3_QS_MULTINODE_FSDP=1 bash runs/qs_multinode_fsdp/<run_id>/qs_create_job.sh
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

MODELS_ROOT = "/mnt/3fs/lxh/agentic-training/models"
DATA_ROOT = "/mnt/3fs/lxh/agentic-training/data/researcher_cot/anchored_v1"


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_multinode_fsdp"


def _find_harness_repo(qs_cfg: dict[str, Any]) -> dict[str, Any]:
    for repo in qs_cfg.get("git_repos") or []:
        if repo.get("name") == "autoresearch_idea_harness":
            return dict(repo)
    raise SystemExit("v3_training.qs.git_repos must include autoresearch_idea_harness")


def _command_text(run_id: str, qs_cfg: dict[str, Any], harness_repo: dict[str, Any], p: argparse.Namespace) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_fsdp/{run_id}"
    out_dir = f"{remote_run_dir}/output"
    repo_url = str(harness_repo["url"])
    repo_ref = str(harness_repo.get("ref") or "V3")
    model_dir = f"{MODELS_ROOT}/{p.model}"
    return f"""#!/usr/bin/env bash
set -uo pipefail
export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
mkdir -p "$REMOTE_RUN_DIR" {shlex.quote(out_dir)}; cd "$REMOTE_RUN_DIR"
HOST="$(hostname)"
LOG="$REMOTE_RUN_DIR/fsdp_${{HOST}}.log"
CLONE="/tmp/aih_${{RUN_ID}}"   # node-local per-pod clone (avoids shared-3fs race across pods)
# --- NCCL over InfiniBand (Phase-0/1: HCAs mlx5_0/1/4/5, busbw ~128GB/s) ---
export NCCL_DEBUG=WARN
export NCCL_IB_HCA="${{NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF="${{PYTORCH_ALLOC_CONF:-expandable_segments:True}}"
if [ -x /workspace/get_socket_name.sh ]; then export NCCL_SOCKET_IFNAME="$(/workspace/get_socket_name.sh)"; fi
echo "[fsdp-mn] host=$HOST PET_NNODES=${{PET_NNODES:-?}} PET_NODE_RANK=${{PET_NODE_RANK:-?}} MASTER_ADDR=${{MASTER_ADDR:-?}} NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME:-unset}} NCCL_IB_HCA=$NCCL_IB_HCA" | tee "$LOG"
# --- staging guards (read-only shared 3fs) ---
test -f {shlex.quote(model_dir)}/config.json || {{ echo "[fsdp-mn] FATAL base model not staged: {model_dir}" | tee -a "$LOG"; exit 3; }}
test -f {shlex.quote(DATA_ROOT)}/train.jsonl || {{ echo "[fsdp-mn] FATAL train data missing" | tee -a "$LOG"; exit 3; }}
rm -rf "$CLONE"
git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} "$CLONE" 2>&1 | tee -a "$LOG"
cd "$CLONE"
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head_${{HOST}}.txt"
python3 -m py_compile scripts/train_v3_researcher_cot_full_fsdp.py 2>&1 | tee -a "$LOG"
# torchrun reads PET_NNODES/PET_NODE_RANK/PET_MASTER_ADDR/PET_MASTER_PORT natively; override nproc only.
torchrun --nproc-per-node=4 scripts/train_v3_researcher_cot_full_fsdp.py \
  --train-jsonl {shlex.quote(DATA_ROOT)}/train.jsonl \
  --val-jsonl {shlex.quote(DATA_ROOT)}/val.jsonl \
  --output-dir {shlex.quote(out_dir)} \
  --base-model {shlex.quote(model_dir)} \
  --enable-thinking false \
  --num-epochs 1 --max-steps {int(p.max_steps)} \
  --max-seq-length {int(p.max_seq_length)} \
  --per-device-batch-size {int(p.per_device_batch)} --grad-accum {int(p.grad_accum)} \
  --lr {p.lr} \
  --fsdp-sharding-strategy {shlex.quote(p.sharding_strategy)} \
  --fsdp-transformer-layer {shlex.quote(p.transformer_layer)} \
  --fsdp-state-dict-type SHARDED_STATE_DICT \
  --save-strategy steps --save-steps {int(p.save_steps)} --save-total-limit 2 \
  --eval-steps {int(p.save_steps)} \
  --summary-name train_summary.json 2>&1 | tee -a "$LOG"
echo "[fsdp-mn] done $(date -Is) host=$HOST" | tee -a "$LOG"
"""


def _script_text(*, run_id: str, qs_cfg: dict[str, Any], worker_num: int, command_path: Path, submit: bool) -> str:
    args = ["qs", "training", "create"]
    if not submit:
        args.append("--dry-run")
    args.extend([
        "--name", run_id,
        "--image", str(qs_cfg["image"]),
        "--queue-id", str(qs_cfg["queue_id"]),
        "--cloud-id", str(qs_cfg["cloud_id"]),
        "--cluster-id", str(qs_cfg["cluster_id"]),
        "--resource-package-id", str(qs_cfg["resource_package_id"]),
        "--job-type", str(qs_cfg.get("job_type", "PytorchJob")),
        "--worker-num", str(worker_num),
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
            ': "${CONFIRM_SUBMIT_V3_QS_MULTINODE_FSDP:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_MULTINODE_FSDP}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_MULTINODE_FSDP must equal 1" >&2',
            "  exit 2",
            "fi",
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
        run_id = f"v3_qs_mn_fsdp_{p.worker_num}w_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    harness_repo = _find_harness_repo(qs_cfg)
    command = _command_text(run_id, qs_cfg, harness_repo, p)
    command_path = run_dir / "qs_command.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, worker_num=p.worker_num, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)
    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, worker_num=p.worker_num, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_fsdp/{run_id}"
    summary = {
        "run_id": run_id, "run_dir": str(run_dir),
        "purpose": f"Multi-node FSDP ({p.sharding_strategy}) — {p.worker_num} workers x4 GPU = {p.worker_num*4} GPU on {p.model}.",
        "worker_num": p.worker_num, "total_gpus_expected": p.worker_num * 4,
        "model": p.model, "sharding_strategy": p.sharding_strategy, "max_steps": p.max_steps,
        "qs": {"queue_id": int(qs_cfg["queue_id"]), "resource_package_id": int(qs_cfg["resource_package_id"]),
               "image": qs_cfg["image"]},
        "artifacts": {"command": str(command_path), "dry_run": str(dry_run_path), "submit": str(submit_path),
                      "expected_remote_output": f"{remote_run_dir}/output",
                      "expected_per_pod_logs": f"{remote_run_dir}/fsdp_<hostname>.log"},
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_MULTINODE_FSDP=1",
        "success_criteria": "loss decreases; util>0 on all pods; sharded checkpoint under output/ reloads.",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_multinode_fsdp"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--worker-num", type=int, default=2, help="pods (each 4 GPU); 16 = 64 GPU")
    parser.add_argument("--model", default="Qwen3-32B", help="dir name under /mnt/3fs/.../models/")
    parser.add_argument("--sharding-strategy", default="hybrid_shard", choices=["full_shard", "hybrid_shard"])
    parser.add_argument("--transformer-layer", default="Qwen3DecoderLayer")
    parser.add_argument("--max-steps", type=int, default=30, help="smoke default; -1 for full epoch")
    parser.add_argument("--max-seq-length", type=int, default=1664)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", default="2e-6")
    parser.add_argument("--save-steps", type=int, default=15)
    args = parser.parse_args()
    summary = prepare(Path(args.config), Path(args.output_dir), args.run_id, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
