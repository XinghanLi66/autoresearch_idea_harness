#!/usr/bin/env python3
# Internal Alibaba PAI-DLC helper — not needed for external reproduction (see REPRODUCE.md).
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_json_object(text: str) -> dict[str, Any]:
    # pai_manage output may include DSW banners and credential-provider warnings.
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(value, dict):
            if isinstance(value.get("body"), dict):
                return value
            best = value
        idx = start + max(end, 1)
    if best is None:
        raise ValueError("no JSON object found in pai_manage output")
    return best


def get_job(endpoint: str, job_id: str) -> dict[str, Any]:
    env = dict(os.environ)
    env.setdefault("ALIBABA_CLOUD_CREDENTIALS_URI", "http://localhost:7002/api/v1/credentials/0")
    cmd = [
        "python",
        os.environ.get("PAI_MANAGE", "/root/.claude/skills/pai/scripts/pai_manage.py"),
        "get-job",
        "--endpoint",
        endpoint,
        "--job-id",
        job_id,
        "--compact",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"get-job exited {result.returncode}: {combined[-2000:]}")
    payload = extract_json_object(combined)
    body = payload.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"get-job payload missing body: {payload}")
    return body


def render_report() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_v3_training_report.py")],
        cwd=str(ROOT),
        check=False,
        timeout=120,
    )


def update_submission(run_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    submission_path = run_dir / "submission.json"
    submission = load_json(submission_path)
    status = str(body.get("Status") or "")
    reason_code = body.get("ReasonCode")
    reason_message = body.get("ReasonMessage")
    pods = body.get("Pods") or []
    pod_ids = [pod.get("PodId") for pod in pods if isinstance(pod, dict) and pod.get("PodId")]
    submission["latest_status"] = {
        "checked_at_cst": now_cst(),
        "status": status,
        "sub_status": body.get("SubStatus") or "",
        "reason_code": reason_code,
        "reason_message": reason_message,
        "duration_sec": body.get("Duration"),
        "pod_ids": pod_ids,
    }
    if status in {"Succeeded", "Failed", "Stopped", "StoppedByUser"}:
        submission["result"] = status
        submission["finished_at_cst"] = now_cst()
    expected_final = submission.get("expected_final")
    if expected_final:
        final_dir = Path(str(expected_final))
        if (final_dir / "config.json").exists():
            submission["final_dir"] = str(final_dir)
    write_json(submission_path, submission)
    return submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll a V3 SFT DLC job and update submission/report artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--endpoint", default="pai-dlc.ap-southeast-1.aliyuncs.com")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-polls", type=int, default=288)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    submission = load_json(run_dir / "submission.json")
    job_id = args.job_id or submission.get("job_id")
    if not job_id:
        raise SystemExit(f"missing job id in {run_dir / 'submission.json'}")
    events_path = run_dir / "monitor_events.jsonl"

    for poll_idx in range(1, int(args.max_polls) + 1):
        try:
            body = get_job(args.endpoint, str(job_id))
            updated = update_submission(run_dir, body)
            render_report()
            event = {
                "time_cst": now_cst(),
                "poll": poll_idx,
                "job_id": job_id,
                "status": (updated.get("latest_status") or {}).get("status"),
                "reason_code": (updated.get("latest_status") or {}).get("reason_code"),
                "duration_sec": (updated.get("latest_status") or {}).get("duration_sec"),
                "pod_ids": (updated.get("latest_status") or {}).get("pod_ids") or [],
                "result": updated.get("result"),
            }
            append_jsonl(events_path, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            if event["status"] in {"Succeeded", "Failed", "Stopped", "StoppedByUser"}:
                return 0 if event["status"] == "Succeeded" else 2
        except Exception as exc:  # noqa: BLE001 - monitor must record and continue.
            event = {
                "time_cst": now_cst(),
                "poll": poll_idx,
                "job_id": job_id,
                "error": str(exc),
            }
            append_jsonl(events_path, event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if args.once:
            return 0
        time.sleep(max(10, int(args.interval_sec)))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
