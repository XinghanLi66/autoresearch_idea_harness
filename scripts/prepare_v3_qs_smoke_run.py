#!/usr/bin/env python3
"""Prepare a fail-closed QS smoke run for the V3 training migration.

The generated artifacts are intentionally submit-ready but not auto-submitted.
They verify the QS queue, GB200/ARM64 runtime, /mnt/3fs visibility, and basic
Python package availability before we port the full V3 SFT/RL launchers.
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_smoke"


def _command_text(run_id: str, qs_cfg: dict[str, Any]) -> str:
    remote_run_dir = str(qs_cfg["remote_run_root"]).rstrip("/") + f"/{run_id}"
    sleep_seconds = int(qs_cfg.get("smoke_sleep_seconds", 60))
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
SMOKE_SLEEP_SECONDS=${{QS_SMOKE_SLEEP_SECONDS:-{sleep_seconds}}}

mkdir -p "$REMOTE_RUN_DIR"
cd "$REMOTE_RUN_DIR"

echo "[qs-smoke] start $(date -Is)" | tee smoke.log
echo "[qs-smoke] run_id=$RUN_ID" | tee -a smoke.log
echo "[qs-smoke] pwd=$(pwd)" | tee -a smoke.log
echo "[qs-smoke] hostname=$(hostname)" | tee -a smoke.log
echo "[qs-smoke] uname=$(uname -a)" | tee -a smoke.log
echo "[qs-smoke] user=$(id)" | tee -a smoke.log

{{
  echo "==== env ===="
  env | sort \\
    | grep -E '^(QS_|CUDA|NCCL|MASTER|WORLD|RANK|LOCAL|PYTHON|PATH=|LD_LIBRARY_PATH=)' \\
    | grep -Ev '(TOKEN|KEY|SECRET|AUTH|CREDENTIAL)' || true
  echo "==== filesystem ===="
  df -h /mnt/3fs || true
  ls -ld /mnt /mnt/3fs /mnt/3fs/lxh /mnt/3fs/lxh/agentic-training || true
  echo "==== gpu ===="
  nvidia-smi -L
  nvidia-smi
  echo "==== python ===="
  command -v python || true
  python -V || true
  python - <<'PY'
import importlib.util
import json
import os
import platform
mods = ["torch", "transformers", "peft", "pyarrow", "pandas", "verl"]
info = {{
    "platform": platform.platform(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "cwd": os.getcwd(),
    "modules": {{m: importlib.util.find_spec(m) is not None for m in mods}},
}}
print(json.dumps(info, indent=2, sort_keys=True))
PY
}} 2>&1 | tee -a smoke.log

python - <<'PY' "$REMOTE_RUN_DIR/result.json"
import json
import os
import platform
import sys
result = {{
    "status": "smoke_command_completed",
    "run_id": os.environ.get("RUN_ID"),
    "remote_run_dir": os.getcwd(),
    "machine": platform.machine(),
    "python": platform.python_version(),
}}
with open(sys.argv[1], "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
PY

echo "[qs-smoke] sleeping $SMOKE_SLEEP_SECONDS seconds for log/exec inspection" | tee -a smoke.log
sleep "$SMOKE_SLEEP_SECONDS"
echo "[qs-smoke] done $(date -Is)" | tee -a smoke.log
"""


def _training_config(run_id: str, qs_cfg: dict[str, Any], command: str) -> dict[str, Any]:
    return {
        "kind": "Training",
        "spec": {
            "name": run_id,
            "image": qs_cfg["image"],
            "command": command,
            "jobType": qs_cfg.get("job_type", "PytorchJob"),
            "workerNum": int(qs_cfg.get("worker_num", 1)),
            "priority": int(qs_cfg.get("priority", 0)),
            "restartNum": int(qs_cfg.get("restart_num", 0)),
            "overuse": bool(qs_cfg.get("overuse", True)),
            "resources": {
                "queueId": int(qs_cfg["queue_id"]),
                "cloudId": int(qs_cfg["cloud_id"]),
                "clusterId": int(qs_cfg["cluster_id"]),
                "resourcePackageId": int(qs_cfg["resource_package_id"]),
            },
        },
    }


def _script_text(
    *,
    run_id: str,
    qs_cfg: dict[str, Any],
    command_path: Path,
    submit: bool,
) -> str:
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
        "--worker-num", str(qs_cfg.get("worker_num", 1)),
        "--priority", str(qs_cfg.get("priority", 0)),
    ])
    if qs_cfg.get("overuse", True):
        args.append("--overuse")
    args.extend(["--yes", "-o", "json", "-q"])
    prefix = f"QS_COMMAND=$(cat {shlex.quote(str(command_path))})"
    command = " ".join(shlex.quote(a) for a in args) + ' --command "$QS_COMMAND"'
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.append(prefix)
    if submit:
        lines.extend([
            ': "${CONFIRM_SUBMIT_V3_QS_SMOKE:?Set to 1 after reviewing qs_training_config.yaml and queue availability.}"',
            'if [[ "${CONFIRM_SUBMIT_V3_QS_SMOKE}" != "1" ]]; then',
            '  echo "CONFIRM_SUBMIT_V3_QS_SMOKE must equal 1" >&2',
            "  exit 2",
            "fi",
            f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
        ])
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(config_path: Path, output_dir: Path, run_id: str | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = [
        "queue_id",
        "cloud_id",
        "cluster_id",
        "resource_package_id",
        "image",
        "remote_run_root",
    ]
    missing = [key for key in required if not qs_cfg.get(key)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")

    if run_id is None:
        run_id = "v3_qs_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    command = _command_text(run_id, qs_cfg)
    command_path = run_dir / "qs_command_smoke.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    training_config = _training_config(run_id, qs_cfg, command)
    qs_config_path = run_dir / "qs_training_config.yaml"
    qs_config_path.write_text(yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True))

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)

    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(run_id=run_id, qs_cfg=qs_cfg, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "QS V3 migration smoke: verify queue 532, GB200/ARM64 runtime, /mnt/3fs, and Python deps.",
        "qs": {
            "queue_id": int(qs_cfg["queue_id"]),
            "queue_name": qs_cfg.get("queue_name"),
            "cloud_id": int(qs_cfg["cloud_id"]),
            "cloud_name": qs_cfg.get("cloud_name"),
            "cluster_id": int(qs_cfg["cluster_id"]),
            "resource_package_id": int(qs_cfg["resource_package_id"]),
            "resource_package_name": qs_cfg.get("resource_package_name"),
            "worker_num": int(qs_cfg.get("worker_num", 1)),
            "overuse": bool(qs_cfg.get("overuse", False)),
            "image": qs_cfg["image"],
            "remote_run_root": qs_cfg["remote_run_root"],
        },
        "artifacts": {
            "command": str(command_path),
            "training_config": str(qs_config_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_result": str(qs_cfg["remote_run_root"]).rstrip("/") + f"/{run_id}/result.json",
            "expected_remote_log": str(qs_cfg["remote_run_root"]).rstrip("/") + f"/{run_id}/smoke.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_SMOKE=1",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_smoke"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    summary = prepare(Path(args.config), Path(args.output_dir), args.run_id)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
