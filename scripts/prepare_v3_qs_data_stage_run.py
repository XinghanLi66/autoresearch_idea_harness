#!/usr/bin/env python3
"""Prepare a QS run that stages small V3 JSONL datasets onto /mnt/3fs.

This is for tiny control/training JSONL files where OSS write access is not
available. The helper embeds gzip+base64 payloads in the QS training command,
decodes them on the pod, writes them under /mnt/3fs, and verifies row counts
and sha256 hashes. It is intentionally not for multi-MB/GB artifacts because
QS rejects large command payloads; use prepare_v3_qs_data_stage_chunks.py for
those.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import shlex
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_data_stage"


def _payload(path: Path, remote_name: str) -> dict[str, Any]:
    raw = path.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    rows = sum(1 for line in raw.splitlines() if line.strip())
    return {
        "local_path": str(path),
        "remote_name": remote_name,
        "raw_bytes": len(raw),
        "gzip_bytes": len(compressed),
        "base64_bytes": len(encoded),
        "rows": rows,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64_wrapped": "\n".join(textwrap.wrap(encoded, 76)),
    }


def _decode_block(payload: dict[str, Any], remote_data_dir: str) -> str:
    remote_name = payload["remote_name"]
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", remote_name)
    b64_path = f"$REMOTE_RUN_DIR/{stem}.gz.b64"
    gz_path = f"$REMOTE_RUN_DIR/{stem}.gz"
    remote_path = f"{remote_data_dir.rstrip('/')}/{remote_name}"
    return f"""
cat > {b64_path} <<'B64_{stem}'
{payload["base64_wrapped"]}
B64_{stem}
base64 -d {b64_path} > {gz_path}
gzip -dc {gz_path} > {shlex.quote(remote_path)}
"""


def _command_text(
    *,
    run_id: str,
    qs_cfg: dict[str, Any],
    remote_data_dir: str,
    payloads: list[dict[str, Any]],
) -> str:
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    remote_run_dir = f"{remote_root}/qs_data_stage/{run_id}"
    sleep_seconds = int(qs_cfg.get("smoke_sleep_seconds", 60))
    decode_blocks = "\n".join(_decode_block(item, remote_data_dir) for item in payloads)
    manifest = [
        {
            key: item[key]
            for key in ["remote_name", "raw_bytes", "gzip_bytes", "base64_bytes", "rows", "sha256"]
        }
        | {"remote_path": f"{remote_data_dir.rstrip('/')}/{item['remote_name']}"}
        for item in payloads
    ]
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return f"""#!/usr/bin/env bash
set -euo pipefail

export RUN_ID={shlex.quote(run_id)}
export REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
export REMOTE_DATA_DIR={shlex.quote(remote_data_dir.rstrip("/"))}
SMOKE_SLEEP_SECONDS=${{QS_SMOKE_SLEEP_SECONDS:-{sleep_seconds}}}

mkdir -p "$REMOTE_RUN_DIR" "$REMOTE_DATA_DIR"
cd "$REMOTE_RUN_DIR"

log() {{
  echo "[qs-data-stage] $*" | tee -a "$REMOTE_RUN_DIR/stage.log"
}}

log "start $(date -Is)"
log "run_id=$RUN_ID"
log "hostname=$(hostname)"
log "uname=$(uname -a)"
log "remote_data_dir=$REMOTE_DATA_DIR"

{decode_blocks}

python3 - <<'PY' "$REMOTE_RUN_DIR/stage_result.json"
import hashlib
import json
import sys
from pathlib import Path

expected = json.loads({manifest_json!r})
results = []
ok = True
for item in expected:
    path = Path(item["remote_path"])
    data = path.read_bytes() if path.exists() else b""
    row_count = sum(1 for line in data.splitlines() if line.strip())
    result = dict(item)
    result.update({{
        "exists": path.exists(),
        "observed_bytes": len(data),
        "observed_rows": row_count,
        "observed_sha256": hashlib.sha256(data).hexdigest() if data else "",
        "bytes_ok": len(data) == item["raw_bytes"],
        "rows_ok": row_count == item["rows"],
        "sha256_ok": hashlib.sha256(data).hexdigest() == item["sha256"] if data else False,
    }})
    if not (result["exists"] and result["bytes_ok"] and result["rows_ok"] and result["sha256_ok"]):
        ok = False
    results.append(result)

summary = {{"status": "ok" if ok else "failed_validation", "files": results}}
Path(sys.argv[1]).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if ok else 2)
PY

log "sleeping $SMOKE_SLEEP_SECONDS seconds for inspection"
sleep "$SMOKE_SLEEP_SECONDS"
log "done $(date -Is)"
"""


def _script_text(*, qs_cfg: dict[str, Any], command_path: Path, submit: bool) -> str:
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
                ': "${CONFIRM_SUBMIT_V3_QS_DATA_STAGE:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
                'if [[ "${CONFIRM_SUBMIT_V3_QS_DATA_STAGE}" != "1" ]]; then',
                '  echo "CONFIRM_SUBMIT_V3_QS_DATA_STAGE must equal 1" >&2',
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
    train_jsonl: Path,
    val_jsonl: Path | None,
    remote_data_dir: str | None,
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
        run_id = "v3_qs_data_stage_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    if not remote_data_dir:
        remote_data_dir = f"{remote_root}/data/researcher_cot/prejudge_v1"
    payloads = [_payload(train_jsonl, "train.jsonl")]
    if val_jsonl:
        payloads.append(_payload(val_jsonl, "val.jsonl"))

    command = _command_text(
        run_id=run_id,
        qs_cfg=qs_cfg,
        remote_data_dir=remote_data_dir,
        payloads=payloads,
    )
    command_path = run_dir / "qs_command_data_stage.sh"
    command_path.write_text(command)
    command_path.chmod(0o755)

    dry_run_path = run_dir / "qs_create_dry_run.sh"
    dry_run_path.write_text(_script_text(qs_cfg=qs_cfg, command_path=command_path, submit=False))
    dry_run_path.chmod(0o755)

    submit_path = run_dir / "qs_create_job.sh"
    submit_path.write_text(_script_text(qs_cfg=qs_cfg, command_path=command_path, submit=True))
    submit_path.chmod(0o755)

    remote_run_dir = f"{remote_root}/qs_data_stage/{run_id}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "purpose": "Stage small V3 JSONL datasets onto /mnt/3fs via gzip+base64 QS command payload.",
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
        "data": [
            {
                key: item[key]
                for key in [
                    "local_path",
                    "remote_name",
                    "raw_bytes",
                    "gzip_bytes",
                    "base64_bytes",
                    "rows",
                    "sha256",
                ]
            }
            | {"remote_path": f"{remote_data_dir.rstrip('/')}/{item['remote_name']}"}
            for item in payloads
        ],
        "artifacts": {
            "command": str(command_path),
            "dry_run": str(dry_run_path),
            "submit": str(submit_path),
            "expected_remote_run_dir": remote_run_dir,
            "expected_remote_result": f"{remote_run_dir}/stage_result.json",
            "expected_remote_log": f"{remote_run_dir}/stage.log",
        },
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_DATA_STAGE=1",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_data_stage"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--train-jsonl",
        default=str(ROOT / "runs/training_data/v3_researcher_cot_prejudge/train.jsonl"),
    )
    parser.add_argument(
        "--val-jsonl",
        default=str(ROOT / "runs/training_data/v3_researcher_cot_prejudge/val.jsonl"),
    )
    parser.add_argument("--remote-data-dir", default=None)
    args = parser.parse_args()
    summary = prepare(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        train_jsonl=Path(args.train_jsonl),
        val_jsonl=Path(args.val_jsonl) if args.val_jsonl else None,
        remote_data_dir=args.remote_data_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
