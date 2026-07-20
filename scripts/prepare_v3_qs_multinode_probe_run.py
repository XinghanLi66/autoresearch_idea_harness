#!/usr/bin/env python3
"""Prepare a QS multi-worker (multi-node) rendezvous + fabric probe.

Phase 0 of the 64+ GPU multi-node effort. Submits a `--worker-num N` (default 2) PytorchJob whose
command runs IDENTICALLY on every pod (master-0, worker-0, ...) and dumps exactly what the QS operator
injects for distributed rendezvous (MASTER_ADDR/WORLD_SIZE/RANK/NODE_RANK/PET_*/QS_*), the network
interfaces, and the InfiniBand HCA names. Reading both pods' logs tells us how to drive torchrun
rendezvous + NCCL-over-IB in Phase 1. Read-only on the cluster (no training, no writes beyond its own
log dir); sleeps so the pods stay up for `qs exec`/log inspection.

Usage:
  python scripts/prepare_v3_qs_multinode_probe_run.py [--worker-num 2] [--sleep 240]
  bash runs/qs_multinode_probe/<run_id>/qs_create_dry_run.sh        # review request JSON
  CONFIRM_SUBMIT_V3_QS_MULTINODE_PROBE=1 bash runs/qs_multinode_probe/<run_id>/qs_create_job.sh
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
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_multinode_probe"


def _command_text(run_id: str, qs_cfg: dict[str, Any], sleep_seconds: int) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_multinode_probe/{run_id}"
    # Identical command on every pod. Per-pod log file keyed by hostname so master/worker don't clobber.
    return f"""#!/usr/bin/env bash
set -uo pipefail
export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
mkdir -p "$REMOTE_RUN_DIR"
HOST="$(hostname)"
LOG="$REMOTE_RUN_DIR/probe_${{HOST}}.log"
SLEEP_SECONDS=${{QS_PROBE_SLEEP_SECONDS:-{sleep_seconds}}}
{{
echo "==== QS MULTINODE PROBE $(date -Is) host=$HOST ===="
echo "---- [1] rendezvous-relevant env ----"
env | grep -iE 'MASTER|WORLD_SIZE|(^|_)RANK|NNODES|NODE_RANK|NPROC|LOCAL_RANK|PET_|GROUP|ROLE' | sort || true
echo "---- [2] all QS_* env ----"
env | grep -iE '^QS' | sort || true
echo "---- [3] hostname resolution (master-0 reachability) ----"
echo "self FQDN: $(hostname -f 2>/dev/null || echo n/a)"
echo "self IPs: $(hostname -I 2>/dev/null || echo n/a)"
echo "---- [4] network interfaces ----"
ls -1 /sys/class/net 2>/dev/null | tr '\\n' ' '; echo
ip -o addr show 2>/dev/null | awk '{{print $2, $3, $4}}' || true
echo "---- [5] InfiniBand / RDMA fabric ----"
ls -1 /dev/infiniband 2>/dev/null | tr '\\n' ' '; echo
if command -v ibstat >/dev/null 2>&1; then ibstat 2>/dev/null | grep -iE 'CA .|State:|Rate:|Link layer|Physical state' | head -40; else echo "no ibstat"; fi
if command -v ibv_devinfo >/dev/null 2>&1; then ibv_devinfo 2>/dev/null | grep -iE 'hca_id|link_layer|state:|active_width|active_speed' | head -40; else echo "no ibv_devinfo"; fi
echo "NCCL_IB env hints: NCCL_IB_HCA=${{NCCL_IB_HCA:-unset}} NCCL_SOCKET_IFNAME=${{NCCL_SOCKET_IFNAME:-unset}}"
echo "---- [6] GPUs ----"
nvidia-smi -L 2>/dev/null | head || echo "no nvidia-smi"
echo "---- [7] torch/nccl ----"
python3 -c "import torch;print('torch',torch.__version__,'nccl',torch.cuda.nccl.version(),'cuda_avail',torch.cuda.is_available(),'device_count',torch.cuda.device_count())" 2>&1 | tail -2 || true
echo "==== PROBE BODY DONE; sleeping $SLEEP_SECONDS for cross-pod inspection ===="
}} 2>&1 | tee "$LOG"
sleep "$SLEEP_SECONDS"
echo "[probe] done $(date -Is) host=$HOST" | tee -a "$LOG"
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
            ': "${CONFIRM_SUBMIT_V3_QS_MULTINODE_PROBE:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_MULTINODE_PROBE}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_MULTINODE_PROBE must equal 1" >&2',
            "  exit 2",
            "fi",
            f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
        ])
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(config_path: Path, output_dir: Path, run_id: str | None, worker_num: int, sleep_seconds: int) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = ["queue_id", "cloud_id", "cluster_id", "resource_package_id", "image", "remote_project_root"]
    missing = [k for k in required if not qs_cfg.get(k)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")

    if run_id is None:
        run_id = "v3_qs_multinode_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    command = _command_text(run_id, qs_cfg, sleep_seconds)
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
    remote_run_dir = f"{remote_root}/qs_multinode_probe/{run_id}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "Phase 0: discover QS multi-worker rendezvous env + InfiniBand fabric across pods.",
        "worker_num": worker_num,
        "total_gpus_expected": worker_num * 4,
        "qs": {
            "queue_id": int(qs_cfg["queue_id"]),
            "cloud_id": int(qs_cfg["cloud_id"]),
            "cluster_id": int(qs_cfg["cluster_id"]),
            "resource_package_id": int(qs_cfg["resource_package_id"]),
            "image": qs_cfg["image"],
        },
        "artifacts": {
            "command": str(command_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_run_dir": remote_run_dir,
            "expected_per_pod_logs": f"{remote_run_dir}/probe_<hostname>.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_MULTINODE_PROBE=1",
        "next": "Read both pods' logs (qs logs <trial> -c pytorch, or exec the pods) to capture rendezvous env + IB HCA names.",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_multinode_probe"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--worker-num", type=int, default=2, help="number of pods (each = 4 GPU)")
    parser.add_argument("--sleep", type=int, default=240, help="seconds to keep pods alive for inspection")
    args = parser.parse_args()
    summary = prepare(Path(args.config), Path(args.output_dir), args.run_id, args.worker_num, args.sleep)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
