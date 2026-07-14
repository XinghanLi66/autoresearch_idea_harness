#!/usr/bin/env python3
"""Prepare a QS run that stages a QS model version onto /mnt/3fs.

The V3 training pods cannot reliably download public HuggingFace models
directly. This helper creates a short QS training job that downloads a model
from the QS model registry into a stable 3FS path, then validates the expected
HF files. The job command references QS_USER/QS_TOKEN environment variables but
does not embed their values in the generated scripts.
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


DEFAULT_MODEL_NAME = "Qwen2.5-0.5B-Instruct"
DEFAULT_VERSION = "1"
DEFAULT_SUBVERSION = "1732894032"
DEFAULT_REGION = "tencent-ap-shanghai"
DEFAULT_ZONE = "cn"


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_model_stage"


def _command_text(
    *,
    run_id: str,
    qs_cfg: dict[str, Any],
    model_name: str,
    version: str,
    subversion: str,
    region: str,
    zone: str,
    remote_model_parent: str,
    expected_total_bytes: int,
    expected_weight_bytes: int,
    weight_file_glob: str,
) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_model_stage/{run_id}"
    sleep_seconds = int(qs_cfg.get("smoke_sleep_seconds", 60))
    remote_model_parent = remote_model_parent.rstrip("/")
    model_dir = f"{remote_model_parent}/{model_name}"
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
export MODEL_NAME={shlex.quote(model_name)}
export MODEL_VERSION={shlex.quote(version)}
export MODEL_SUBVERSION={shlex.quote(subversion)}
export MODEL_REGION={shlex.quote(region)}
export MODEL_ZONE={shlex.quote(zone)}
export REMOTE_MODEL_PARENT={shlex.quote(remote_model_parent)}
export EXPECTED_MODEL_DIR={shlex.quote(model_dir)}
export EXPECTED_TOTAL_BYTES={expected_total_bytes}
export EXPECTED_WEIGHT_BYTES={expected_weight_bytes}
export WEIGHT_FILE_GLOB={shlex.quote(weight_file_glob)}
SMOKE_SLEEP_SECONDS=${{QS_SMOKE_SLEEP_SECONDS:-{sleep_seconds}}}

mkdir -p "$REMOTE_RUN_DIR" "$REMOTE_MODEL_PARENT"
cd "$REMOTE_RUN_DIR"

log() {{
  echo "[qs-model-stage] $*" | tee -a "$REMOTE_RUN_DIR/stage.log"
}}

log "start $(date -Is)"
log "run_id=$RUN_ID"
log "hostname=$(hostname)"
log "uname=$(uname -a)"
log "model=$MODEL_NAME version=$MODEL_VERSION subversion=$MODEL_SUBVERSION region=$MODEL_REGION zone=$MODEL_ZONE"
log "remote_model_parent=$REMOTE_MODEL_PARENT"
log "expected_model_dir=$EXPECTED_MODEL_DIR"

if [[ -z "${{QS_USER:-}}" ]]; then
  log "QS_USER env missing"
  exit 20
fi
if [[ -z "${{QS_TOKEN:-}}" ]]; then
  log "QS_TOKEN env missing"
  exit 21
fi
log "QS credential env vars are present"

python3 -m pip install -q requests wheel setuptools \\
  -i http://pypi.devops.xiaohongshu.com/simple/ \\
  --trusted-host pypi.devops.xiaohongshu.com 2>&1 | tee -a "$REMOTE_RUN_DIR/stage.log"

python3 -m pip install -q quicksilver-toolkit \\
  -i http://pypi.devops.xiaohongshu.com/simple/ \\
  --trusted-host pypi.devops.xiaohongshu.com \\
  --no-build-isolation 2>&1 | tee -a "$REMOTE_RUN_DIR/stage.log"

python3 - <<'PY' 2>&1 | tee -a "$REMOTE_RUN_DIR/stage.log"
import importlib.util
print("model_tools_available", importlib.util.find_spec("model_tools") is not None)
PY

normalize_layout() {{
  mkdir -p "$EXPECTED_MODEL_DIR"
  shopt -s nullglob
  for path in \\
    "$REMOTE_MODEL_PARENT"/*.bin \\
    "$REMOTE_MODEL_PARENT"/*.json \\
    "$REMOTE_MODEL_PARENT"/*.md \\
    "$REMOTE_MODEL_PARENT"/*.py \\
    "$REMOTE_MODEL_PARENT"/*.safetensors \\
    "$REMOTE_MODEL_PARENT"/*.txt \\
    "$REMOTE_MODEL_PARENT"/LICENSE \\
    "$REMOTE_MODEL_PARENT"/.[!.]*; do
    if [[ -f "$path" ]]; then
      mv -f "$path" "$EXPECTED_MODEL_DIR/$(basename "$path")"
    fi
  done
  shopt -u nullglob
  if compgen -G "$EXPECTED_MODEL_DIR/$WEIGHT_FILE_GLOB" >/dev/null; then
    log "model already in expected directory layout"
    return 0
  fi
  if compgen -G "$REMOTE_MODEL_PARENT/*.safetensors" >/dev/null; then
    log "normalizing flat model_tools download layout"
    shopt -s nullglob
    for path in \\
      "$REMOTE_MODEL_PARENT"/*.bin \\
      "$REMOTE_MODEL_PARENT"/*.json \\
      "$REMOTE_MODEL_PARENT"/*.md \\
      "$REMOTE_MODEL_PARENT"/*.py \\
      "$REMOTE_MODEL_PARENT"/*.safetensors \\
      "$REMOTE_MODEL_PARENT"/*.txt \\
      "$REMOTE_MODEL_PARENT"/LICENSE \\
      "$REMOTE_MODEL_PARENT"/.[!.]*; do
      if [[ -f "$path" ]]; then
        mv -f "$path" "$EXPECTED_MODEL_DIR/$(basename "$path")"
      fi
    done
    shopt -u nullglob
  fi
}}

if [[ -d "$EXPECTED_MODEL_DIR" ]] && ! compgen -G "$EXPECTED_MODEL_DIR/$WEIGHT_FILE_GLOB" >/dev/null; then
  log "removing incomplete expected model directory"
  rm -rf "$EXPECTED_MODEL_DIR"
fi
rm -rf "$REMOTE_MODEL_PARENT/$MODEL_NAME.tmp"

normalize_layout
if ! compgen -G "$EXPECTED_MODEL_DIR/$WEIGHT_FILE_GLOB" >/dev/null; then
  log "downloading model from QS registry"

  python3 -m model_tools download \\
    --model_name "$MODEL_NAME" \\
    --version "$MODEL_VERSION" \\
    --subversion "$MODEL_SUBVERSION" \\
    --save_path "$REMOTE_MODEL_PARENT" \\
    --region "$MODEL_REGION" \\
    --zone "$MODEL_ZONE" \\
    --num_threads 8 \\
    --user "$QS_USER" \\
    --token "$QS_TOKEN" 2>&1 \\
    | sed -E 's/(secretText )[[:alnum:]_-]+/\\1[REDACTED]/g' \\
    | tee -a "$REMOTE_RUN_DIR/stage.log"
  normalize_layout
else
  log "skipping download because expected model weights already exist"
fi

python3 - <<'PY' "$REMOTE_RUN_DIR/stage_result.json" "$EXPECTED_MODEL_DIR" "$EXPECTED_TOTAL_BYTES" "$EXPECTED_WEIGHT_BYTES" "$WEIGHT_FILE_GLOB"
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
model_dir = Path(sys.argv[2])
expected_total = int(sys.argv[3])
expected_weight = int(sys.argv[4])
weight_glob = sys.argv[5]
required = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
]
files = {{}}
missing = []
for name in required:
    path = model_dir / name
    if path.exists():
        files[name] = path.stat().st_size
    else:
        missing.append(name)
weight_files = sorted(model_dir.glob(weight_glob)) if model_dir.exists() else []
weight_sizes = {{path.name: path.stat().st_size for path in weight_files if path.is_file()}}
weight_total = sum(weight_sizes.values())
total = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()) if model_dir.exists() else 0
result = {{
    "status": "ok",
    "model_dir": str(model_dir),
    "exists": model_dir.exists(),
    "required_files": files,
    "missing_required_files": missing,
    "weight_file_glob": weight_glob,
    "weight_files": weight_sizes,
    "weight_file_count": len(weight_sizes),
    "weight_total_bytes": weight_total,
    "total_file_bytes": total,
    "expected_total_bytes": expected_total,
    "expected_weight_bytes": expected_weight,
    "weight_size_ok": weight_total >= expected_weight,
    "total_size_ok": total >= expected_total,
}}
if missing or not weight_sizes or not result["weight_size_ok"] or not result["total_size_ok"]:
    result["status"] = "failed_validation"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if result["status"] == "ok" else 2)
PY

log "sleeping $SMOKE_SLEEP_SECONDS seconds for inspection"
sleep "$SMOKE_SLEEP_SECONDS"
log "done $(date -Is)"
"""


def _script_text(
    *,
    qs_cfg: dict[str, Any],
    command_path: Path,
    submit: bool,
) -> str:
    args = ["qs", "training", "create"]
    if not submit:
        args.append("--dry-run")
    args.extend(
        [
            "--name",
            command_path.parent.name,
            "--image",
            str(qs_cfg["image"]),
            "--queue-id",
            str(qs_cfg["queue_id"]),
            "--cloud-id",
            str(qs_cfg["cloud_id"]),
            "--cluster-id",
            str(qs_cfg["cluster_id"]),
            "--resource-package-id",
            str(qs_cfg["resource_package_id"]),
            "--job-type",
            str(qs_cfg.get("job_type", "PytorchJob")),
            "--worker-num",
            str(qs_cfg.get("worker_num", 1)),
            "--priority",
            str(qs_cfg.get("priority", 0)),
        ]
    )
    if qs_cfg.get("overuse", True):
        args.append("--overuse")
    args.extend(["--yes", "-o", "json", "-q"])
    prefix = f"QS_COMMAND=$(cat {shlex.quote(str(command_path))})"
    command = " ".join(shlex.quote(a) for a in args) + ' --command "$QS_COMMAND"'
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", prefix]
    if submit:
        lines.extend(
            [
                ': "${CONFIRM_SUBMIT_V3_QS_MODEL_STAGE:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
                'if [[ "${CONFIRM_SUBMIT_V3_QS_MODEL_STAGE}" != "1" ]]; then',
                '  echo "CONFIRM_SUBMIT_V3_QS_MODEL_STAGE must equal 1" >&2',
                "  exit 2",
                "fi",
                f"{command} | tee {shlex.quote(str(command_path.parent / 'submission.json'))}",
            ]
        )
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def prepare(
    *,
    config_path: Path,
    output_dir: Path,
    run_id: str | None,
    model_name: str,
    version: str,
    subversion: str,
    region: str,
    zone: str,
    remote_model_parent: str | None,
    expected_total_bytes: int,
    expected_weight_bytes: int,
    weight_file_glob: str,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = [
        "queue_id",
        "cloud_id",
        "cluster_id",
        "resource_package_id",
        "image",
        "remote_project_root",
    ]
    missing = [key for key in required if not qs_cfg.get(key)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")

    if run_id is None:
        run_id = "v3_qs_model_stage_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    if not remote_model_parent:
        remote_model_parent = f"{remote_root}/models"
    command = _command_text(
        run_id=run_id,
        qs_cfg=qs_cfg,
        model_name=model_name,
        version=version,
        subversion=subversion,
        region=region,
        zone=zone,
        remote_model_parent=remote_model_parent,
        expected_total_bytes=expected_total_bytes,
        expected_weight_bytes=expected_weight_bytes,
        weight_file_glob=weight_file_glob,
    )
    command_path = run_dir / "qs_command_model_stage.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(qs_cfg=qs_cfg, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)

    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(qs_cfg=qs_cfg, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_run_dir = f"{remote_root}/qs_model_stage/{run_id}"
    model_dir = f"{remote_model_parent.rstrip('/')}/{model_name}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "Stage a QS model-registry HF model onto /mnt/3fs for V3 training jobs.",
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
        },
        "model": {
            "name": model_name,
            "version": version,
            "subversion": subversion,
            "region": region,
            "zone": zone,
            "expected_total_bytes": expected_total_bytes,
            "expected_weight_bytes": expected_weight_bytes,
            "weight_file_glob": weight_file_glob,
            "expected_remote_model_dir": model_dir,
        },
        "artifacts": {
            "command": str(command_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_run_dir": remote_run_dir,
            "expected_remote_result": f"{remote_run_dir}/stage_result.json",
            "expected_remote_log": f"{remote_run_dir}/stage.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_MODEL_STAGE=1",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_model_stage"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--subversion", default=DEFAULT_SUBVERSION)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument("--remote-model-parent", default=None)
    parser.add_argument("--expected-total-bytes", type=int, default=999_603_325)
    parser.add_argument("--expected-weight-bytes", type=int, default=988_097_824)
    parser.add_argument("--weight-file-glob", default="*.safetensors")
    args = parser.parse_args()
    summary = prepare(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        model_name=args.model_name,
        version=str(args.version),
        subversion=str(args.subversion),
        region=args.region,
        zone=args.zone,
        remote_model_parent=args.remote_model_parent,
        expected_total_bytes=args.expected_total_bytes,
        expected_weight_bytes=args.expected_weight_bytes,
        weight_file_glob=args.weight_file_glob,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
