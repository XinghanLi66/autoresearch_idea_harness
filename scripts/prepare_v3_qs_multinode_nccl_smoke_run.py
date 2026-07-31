#!/usr/bin/env python3
"""Prepare the multi-node NCCL all-reduce smoke (Phase 1 of the 64+ GPU effort).

Submits a `--worker-num N` (default 2) PytorchJob that clones V3 and runs
`scripts/qs_multinode_nccl_smoke.py` under torchrun on every pod. Thanks to the Phase-0 finding that QS
injects the `PET_*` rendezvous env, the launch is just `torchrun --nproc-per-node=4 <script>` — torchrun
reads PET_NNODES / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT natively (we only override
nproc-per-node, since PET_NPROC_PER_NODE defaults to 'auto'). NCCL runs over InfiniBand (mlx5 HCAs).

Usage:
  python scripts/prepare_v3_qs_multinode_nccl_smoke_run.py [--worker-num 2]
  bash runs/qs_multinode_nccl_smoke/<run_id>/qs_create_dry_run.sh
  CONFIRM_SUBMIT_V3_QS_MULTINODE_NCCL=1 bash runs/qs_multinode_nccl_smoke/<run_id>/qs_create_job.sh
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


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_multinode_nccl_smoke"


def _find_harness_repo(qs_cfg: dict[str, Any]) -> dict[str, Any]:
    for repo in qs_cfg.get("git_repos") or []:
        if repo.get("name") == "autoresearch_idea_harness":
            return dict(repo)
    raise SystemExit("v3_training.qs.git_repos must include autoresearch_idea_harness")


def _command_text(run_id: str, qs_cfg: dict[str, Any], harness_repo: dict[str, Any]) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_nccl_smoke/{run_id}"
    repo_url = str(harness_repo["url"])
    repo_ref = str(harness_repo.get("ref") or "V3")
    # Multi-pod jobs share /mnt/3fs -> every pod cloning into the same dir races and clobbers.
    # Clone to node-local /tmp (per-pod, unique) instead; keep per-pod logs on shared 3fs.
    return f"""#!/usr/bin/env bash
set -uo pipefail
export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
mkdir -p "$REMOTE_RUN_DIR"; cd "$REMOTE_RUN_DIR"
HOST="$(hostname)"
LOG="$REMOTE_RUN_DIR/nccl_${{HOST}}.log"
CLONE="/tmp/aih_${{RUN_ID}}"   # node-local per-pod clone (avoids shared-3fs race across pods)
# --- NCCL over InfiniBand (Phase-0: HCAs mlx5_0/1/4/5 active). Image sets NCCL_SOCKET_IFNAME to a
#     literal $(...) string, so evaluate the helper ourselves. ---
export NCCL_DEBUG=INFO
export NCCL_IB_HCA="${{NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_4,mlx5_5}}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ -x /workspace/get_socket_name.sh ]; then export NCCL_SOCKET_IFNAME="$(/workspace/get_socket_name.sh)"; fi
echo "[nccl-smoke] host=$HOST PET_NNODES=${{PET_NNODES:-?}} PET_NODE_RANK=${{PET_NODE_RANK:-?}} MASTER_ADDR=${{MASTER_ADDR:-?}} MASTER_PORT=${{MASTER_PORT:-?}} PET_MASTER_PORT=${{PET_MASTER_PORT:-?}} NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME:-unset}} NCCL_IB_HCA=$NCCL_IB_HCA" | tee "$LOG"
rm -rf "$CLONE"
git clone --depth 1 --branch {shlex.quote(repo_ref)} {shlex.quote(repo_url)} "$CLONE" 2>&1 | tee -a "$LOG"
cd "$CLONE"
git rev-parse HEAD | tee "$REMOTE_RUN_DIR/git_head_${{HOST}}.txt"
python3 -m py_compile scripts/qs_multinode_nccl_smoke.py 2>&1 | tee -a "$LOG"
# torchrun reads PET_NNODES/PET_NODE_RANK/PET_MASTER_ADDR/PET_MASTER_PORT natively; override nproc only.
torchrun --nproc-per-node=4 scripts/qs_multinode_nccl_smoke.py 2>&1 | tee -a "$LOG"
echo "[nccl-smoke] done $(date -Is) host=$HOST rc" | tee -a "$LOG"
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
            ': "${CONFIRM_SUBMIT_V3_QS_MULTINODE_NCCL:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_MULTINODE_NCCL}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_MULTINODE_NCCL must equal 1" >&2',
            "  exit 2",
            "fi",
            f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
        ])
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(config_path: Path, output_dir: Path, run_id: str | None, worker_num: int) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = ["queue_id", "cloud_id", "cluster_id", "resource_package_id", "image", "remote_project_root", "git_repos"]
    missing = [k for k in required if not qs_cfg.get(k)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")

    if run_id is None:
        run_id = "v3_qs_multinode_nccl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    harness_repo = _find_harness_repo(qs_cfg)
    command = _command_text(run_id, qs_cfg, harness_repo)
    command_path = run_dir / "qs_command.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, worker_num=worker_num, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)
    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, worker_num=worker_num, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_nccl_smoke/{run_id}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "Phase 1: cross-pod NCCL all-reduce smoke over InfiniBand (rendezvous + bandwidth).",
        "worker_num": worker_num,
        "total_gpus_expected": worker_num * 4,
        "qs": {"queue_id": int(qs_cfg["queue_id"]), "cloud_id": int(qs_cfg["cloud_id"]),
               "cluster_id": int(qs_cfg["cluster_id"]), "resource_package_id": int(qs_cfg["resource_package_id"]),
               "image": qs_cfg["image"]},
        "source": {"repo": harness_repo},
        "artifacts": {"command": str(command_path), "dry_run": str(dry_run_path), "submit": str(submit_path),
                      "expected_remote_run_dir": remote_run_dir,
                      "expected_per_pod_logs": f"{remote_run_dir}/nccl_<hostname>.log"},
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_MULTINODE_NCCL=1",
        "success_criteria": "all worker_num*4 ranks report OK=True; rank-0 busbw is IB-class (hundreds GB/s).",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_multinode_nccl_smoke"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--worker-num", type=int, default=2, help="number of pods (each = 4 GPU)")
    args = parser.parse_args()
    summary = prepare(Path(args.config), Path(args.output_dir), args.run_id, args.worker_num)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
