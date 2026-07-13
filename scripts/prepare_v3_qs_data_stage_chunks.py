#!/usr/bin/env python3
"""Prepare chunked QS jobs for staging V3 JSONL data onto /mnt/3fs.

Unlike `prepare_v3_qs_data_stage_run.py`, this avoids the local ARG_MAX limit
and the QS backend command-column limit by creating one QS trial per base64
chunk, plus a final verification trial. Keep chunks around 50k characters:
larger commands may pass dry-run but fail creation with "Data too long for
column 'command'".
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
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "qs_data_stage_chunks"


def _payload(path: Path, remote_name: str, chunk_chars: int) -> dict[str, Any]:
    raw = path.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    chunks = [encoded[i : i + chunk_chars] for i in range(0, len(encoded), chunk_chars)]
    return {
        "local_path": str(path),
        "remote_name": remote_name,
        "raw_bytes": len(raw),
        "gzip_bytes": len(compressed),
        "base64_bytes": len(encoded),
        "rows": sum(1 for line in raw.splitlines() if line.strip()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "chunks": chunks,
    }


def _qs_create_script(qs_cfg: dict[str, Any], command_path: Path, *, submit: bool) -> str:
    args = ["qs", "training", "create"]
    if not submit:
        args.append("--dry-run")
    args.extend(
        [
            "--name",
            command_path.stem,
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
            "--yes",
            "-o",
            "json",
            "-q",
        ]
    )
    if qs_cfg.get("overuse", True):
        args.insert(-3, "--overuse")
    prefix = f"QS_COMMAND=$(cat {shlex.quote(str(command_path))})"
    command = " ".join(shlex.quote(a) for a in args) + ' --command "$QS_COMMAND"'
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", prefix]
    if submit:
        lines.extend(
            [
                ': "${CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK:?Set to 1 after reviewing run_plan.json and dry-run output.}"',
                'if [[ "${CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK}" != "1" ]]; then',
                '  echo "CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK must equal 1" >&2',
                "  exit 2",
                "fi",
                f"{command} | tee {shlex.quote(str(command_path.with_suffix('.submission.json')))}",
            ]
        )
    else:
        lines.append(command)
    return "\n".join(lines) + "\n"


def _chunk_command(run_id: str, remote_root: str, remote_data_dir: str, item: dict[str, Any], index: int, chunk: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", item["remote_name"])
    remote_run_dir = f"{remote_root}/qs_data_stage_chunks/{run_id}"
    chunk_dir = f"{remote_run_dir}/chunks/{safe}"
    wrapped = "\n".join(textwrap.wrap(chunk, 76))
    return f"""#!/usr/bin/env bash
set -euo pipefail
REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
CHUNK_DIR={shlex.quote(chunk_dir)}
mkdir -p "$CHUNK_DIR" {shlex.quote(remote_data_dir)}
cat > "$CHUNK_DIR/part_{index:04d}.b64" <<'B64'
{wrapped}
B64
python3 - <<'PY' "$REMOTE_RUN_DIR/events.jsonl" {shlex.quote(item["remote_name"])} {index} "$CHUNK_DIR/part_{index:04d}.b64"
import json, sys, time
from pathlib import Path
event={{"time": time.time(), "event": "chunk_written", "remote_name": sys.argv[2], "index": int(sys.argv[3]), "path": sys.argv[4], "bytes": Path(sys.argv[4]).stat().st_size}}
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(event, sort_keys=True) + "\\n")
print(json.dumps(event, sort_keys=True))
PY
"""


def _final_command(run_id: str, remote_root: str, remote_data_dir: str, payloads: list[dict[str, Any]]) -> str:
    remote_run_dir = f"{remote_root}/qs_data_stage_chunks/{run_id}"
    manifest = []
    blocks = []
    for item in payloads:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", item["remote_name"])
        chunk_dir = f"{remote_run_dir}/chunks/{safe}"
        remote_path = f"{remote_data_dir.rstrip('/')}/{item['remote_name']}"
        manifest.append(
            {
                "remote_name": item["remote_name"],
                "remote_path": remote_path,
                "chunk_dir": chunk_dir,
                "chunk_count": len(item["chunks"]),
                "raw_bytes": item["raw_bytes"],
                "gzip_bytes": item["gzip_bytes"],
                "base64_bytes": item["base64_bytes"],
                "rows": item["rows"],
                "sha256": item["sha256"],
            }
        )
        blocks.append(
            f"""
python3 - <<'PY' {shlex.quote(chunk_dir)} {len(item["chunks"])}
import sys
from pathlib import Path
chunk_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
missing = [i for i in range(expected) if not (chunk_dir / f"part_{{i:04d}}.b64").exists()]
if missing:
    raise SystemExit(f"missing chunks in {{chunk_dir}}: {{missing[:10]}}")
PY
cat {shlex.quote(chunk_dir)}/part_*.b64 > "$REMOTE_RUN_DIR/{safe}.gz.b64"
base64 -d "$REMOTE_RUN_DIR/{safe}.gz.b64" > "$REMOTE_RUN_DIR/{safe}.gz"
gzip -dc "$REMOTE_RUN_DIR/{safe}.gz" > {shlex.quote(remote_path)}
"""
        )
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return f"""#!/usr/bin/env bash
set -euo pipefail
REMOTE_RUN_DIR={shlex.quote(remote_run_dir)}
REMOTE_DATA_DIR={shlex.quote(remote_data_dir.rstrip("/"))}
mkdir -p "$REMOTE_RUN_DIR" "$REMOTE_DATA_DIR"
{"".join(blocks)}
python3 - <<'PY' "$REMOTE_RUN_DIR/stage_result.json"
import hashlib, json, sys
from pathlib import Path
expected = json.loads({manifest_json!r})
results = []
ok = True
for item in expected:
    path = Path(item["remote_path"])
    data = path.read_bytes() if path.exists() else b""
    observed_sha = hashlib.sha256(data).hexdigest() if data else ""
    observed_rows = sum(1 for line in data.splitlines() if line.strip())
    result = dict(item)
    result.update({{
        "exists": path.exists(),
        "observed_bytes": len(data),
        "observed_rows": observed_rows,
        "observed_sha256": observed_sha,
        "bytes_ok": len(data) == item["raw_bytes"],
        "rows_ok": observed_rows == item["rows"],
        "sha256_ok": observed_sha == item["sha256"],
    }})
    if not (result["exists"] and result["bytes_ok"] and result["rows_ok"] and result["sha256_ok"]):
        ok = False
    results.append(result)
summary = {{"status": "ok" if ok else "failed_validation", "files": results}}
Path(sys.argv[1]).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if ok else 2)
PY
"""


def prepare(
    config_path: Path,
    output_dir: Path,
    run_id: str | None,
    train_jsonl: Path,
    val_jsonl: Path | None,
    remote_data_dir: str | None,
    chunk_chars: int,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    qs_cfg = dict(cfg.get("v3_training", {}).get("qs") or {})
    required = ["queue_id", "cloud_id", "cluster_id", "resource_package_id", "image", "remote_project_root"]
    missing = [key for key in required if not qs_cfg.get(key)]
    if missing:
        raise SystemExit(f"missing v3_training.qs config keys: {', '.join(missing)}")
    if run_id is None:
        run_id = "v3_qs_data_stage_chunks_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = _safe_label(run_id)
    run_dir = (output_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_root = str(qs_cfg["remote_project_root"]).rstrip("/")
    if not remote_data_dir:
        remote_data_dir = f"{remote_root}/data/researcher_cot/prejudge_v1"

    payloads = [_payload(train_jsonl, "train.jsonl", chunk_chars)]
    if val_jsonl:
        payloads.append(_payload(val_jsonl, "val.jsonl", chunk_chars))

    steps = []
    step_index = 0
    for item in payloads:
        for chunk_index, chunk in enumerate(item["chunks"]):
            name = f"step_{step_index:03d}_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', item['remote_name'])}_part_{chunk_index:04d}"
            command_path = run_dir / f"{name}.sh"
            command_path.write_text(_chunk_command(run_id, remote_root, remote_data_dir, item, chunk_index, chunk))
            command_path.chmod(0o755)
            dry = run_dir / f"{name}.dry_run.sh"
            submit = run_dir / f"{name}.submit.sh"
            dry.write_text(_qs_create_script(qs_cfg, command_path, submit=False))
            submit.write_text(_qs_create_script(qs_cfg, command_path, submit=True))
            dry.chmod(0o755)
            submit.chmod(0o755)
            steps.append(
                {
                    "kind": "chunk",
                    "remote_name": item["remote_name"],
                    "chunk_index": chunk_index,
                    "command": str(command_path),
                    "dry_run": str(dry),
                    "submit": str(submit),
                    "command_bytes": command_path.stat().st_size,
                }
            )
            step_index += 1

    final_name = f"step_{step_index:03d}_finalize"
    final_command = run_dir / f"{final_name}.sh"
    final_command.write_text(_final_command(run_id, remote_root, remote_data_dir, payloads))
    final_command.chmod(0o755)
    final_dry = run_dir / f"{final_name}.dry_run.sh"
    final_submit = run_dir / f"{final_name}.submit.sh"
    final_dry.write_text(_qs_create_script(qs_cfg, final_command, submit=False))
    final_submit.write_text(_qs_create_script(qs_cfg, final_command, submit=True))
    final_dry.chmod(0o755)
    final_submit.chmod(0o755)
    steps.append(
        {
            "kind": "finalize",
            "command": str(final_command),
            "dry_run": str(final_dry),
            "submit": str(final_submit),
            "command_bytes": final_command.stat().st_size,
        }
    )

    remote_run_dir = f"{remote_root}/qs_data_stage_chunks/{run_id}"
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "remote_run_dir": remote_run_dir,
        "remote_data_dir": remote_data_dir,
        "chunk_chars": chunk_chars,
        "submit_guard_env": "CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK=1",
        "data": [
            {
                key: item[key]
                for key in ["local_path", "remote_name", "raw_bytes", "gzip_bytes", "base64_bytes", "rows", "sha256"]
            }
            | {
                "chunk_count": len(item["chunks"]),
                "remote_path": f"{remote_data_dir.rstrip('/')}/{item['remote_name']}",
            }
            for item in payloads
        ],
        "steps": steps,
        "expected_remote_result": f"{remote_run_dir}/stage_result.json",
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_data_stage_chunks"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--train-jsonl", default=str(ROOT / "runs/training_data/v3_researcher_cot_prejudge/train.jsonl"))
    parser.add_argument("--val-jsonl", default=str(ROOT / "runs/training_data/v3_researcher_cot_prejudge/val.jsonl"))
    parser.add_argument("--remote-data-dir", default=None)
    parser.add_argument("--chunk-chars", type=int, default=50_000)
    args = parser.parse_args()
    summary = prepare(
        Path(args.config),
        Path(args.output_dir),
        args.run_id,
        Path(args.train_jsonl),
        Path(args.val_jsonl) if args.val_jsonl else None,
        args.remote_data_dir,
        args.chunk_chars,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
