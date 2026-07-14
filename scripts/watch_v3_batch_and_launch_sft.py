#!/usr/bin/env python3
# Internal Alibaba PAI-DLC helper — not needed for external reproduction (see REPRODUCE.md).
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAI = Path(os.environ.get("PAI_MANAGE", "/root/.claude/skills/pai/scripts/pai_manage.py"))
ENDPOINT = "pai-dlc.ap-southeast-1.aliyuncs.com"
CREDENTIAL_URI = "http://localhost:7002/api/v1/credentials/0"
KNOWN_POOLS = [
    ("224239", "quota1ecrg95m4n9"),
    ("137902", "quotadbz1mvpy1v5"),
    ("238626", "quota1d8xmvdw5tb"),
    ("262162", "quota1shcr2h7uae"),
]


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(errors="replace") as f:
        return sum(1 for line in f if line.strip())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_cmd(cmd: list[str], *, timeout: int = 600, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)
    return subprocess.run(cmd, cwd=str(ROOT), env=proc_env, capture_output=True, text=True, timeout=timeout)


def extract_json_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    best: Any = None
    idx = 0
    while idx < len(text):
        starts = [pos for pos in (text.find("{", idx), text.find("[", idx)) if pos >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(value, dict) and isinstance(value.get("body"), dict):
            return value
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            best = value
        idx = start + max(end, 1)
    if best is None:
        raise ValueError("no JSON object/array found in command output")
    return best


def wait_for_collate(args: argparse.Namespace, events: Path) -> tuple[Path, int]:
    sft_dir = ROOT / "runs" / "training_data" / f"v3_0_sft_qwen25_32b_{args.batch_name}"
    audit_dir = ROOT / "runs" / "training_data" / f"v3_0_targets_qwen25_32b_{args.batch_name}_audit"
    ready_path = sft_dir / ("all.jsonl" if args.train_source == "all" else "train.jsonl")
    summary_path = audit_dir / "pipeline_summary.json"
    deadline = time.time() + args.max_wait_sec
    last_rows = -1
    while time.time() < deadline:
        rows = count_jsonl(ready_path)
        if summary_path.exists() and rows >= args.min_train_rows:
            append_jsonl(events, {
                "time": now_cst(),
                "event": "collate_ready",
                "sft_dir": str(sft_dir),
                "ready_path": str(ready_path),
                "ready_rows": rows,
                "train_source": args.train_source,
            })
            return sft_dir, rows
        target_rows = count_jsonl(ROOT / "runs" / "training_data" / f"v3_0_targets_qwen25_32b_{args.batch_name}" / "tex_targets.jsonl")
        if target_rows != last_rows:
            append_jsonl(events, {
                "time": now_cst(),
                "event": "waiting_for_batch",
                "batch_name": args.batch_name,
                "target_rows": target_rows,
                "ready_rows": rows,
                "summary_exists": summary_path.exists(),
                "train_source": args.train_source,
            })
            last_rows = target_rows
        time.sleep(args.interval_sec)
    raise TimeoutError(f"timed out waiting for {args.batch_name} collate with >= {args.min_train_rows} rows")


def materialize_sft_dir(sft_dir: Path, args: argparse.Namespace, events: Path) -> tuple[Path, int]:
    if args.train_source == "train":
        return sft_dir, count_jsonl(sft_dir / "train.jsonl")
    all_rows = read_jsonl(sft_dir / "all.jsonl")
    if len(all_rows) < args.min_train_rows:
        raise RuntimeError(f"all.jsonl has {len(all_rows)} rows, expected at least {args.min_train_rows}")
    all_rows.sort(key=lambda row: (
        str(row.get("created") or ""),
        int(row.get("chronological_rank") or 0),
        str(row.get("arxiv_id") or row.get("sample_id") or ""),
    ))
    launch_dir = sft_dir.with_name(f"{sft_dir.name}_train_all")
    launch_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(launch_dir / "train.jsonl", all_rows)
    for name in ("all.jsonl", "val.jsonl", "test.jsonl", "missing_target.jsonl"):
        src = sft_dir / name
        if src.exists():
            shutil.copy2(src, launch_dir / name)
    source_summary = {}
    if (sft_dir / "summary.json").exists():
        source_summary = json.loads((sft_dir / "summary.json").read_text())
    summary = {
        **source_summary,
        "output_dir": str(launch_dir),
        "source_sft_dir": str(sft_dir),
        "train_source": "all",
        "train_row_count": len(all_rows),
        "note": "train.jsonl is materialized from all.jsonl for the 929-row CoT SFT run; original split files are preserved.",
    }
    write_json(launch_dir / "summary.json", summary)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "materialize_train_all",
        "source_sft_dir": str(sft_dir),
        "launch_sft_dir": str(launch_dir),
        "train_rows": len(all_rows),
    })
    return launch_dir, len(all_rows)


def score_sft_targets(sft_dir: Path, args: argparse.Namespace, events: Path) -> None:
    output_dir = ROOT / "runs" / "training_data" / f"{sft_dir.name}_quality"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "score_v3_sft_targets.py"),
        "--input",
        str(sft_dir / "train.jsonl"),
        "--output-dir",
        str(output_dir),
        "--allow-raw-target",
    ]
    if args.max_seq_length:
        cmd.extend(["--max-seq-length", str(args.max_seq_length)])
    result = run_cmd(cmd, timeout=1800)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "score_sft_targets",
        "returncode": result.returncode,
        "output_dir": str(output_dir),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    })
    if result.returncode != 0:
        raise RuntimeError(f"score_v3_sft_targets failed: {result.stderr[-1000:]}")


def parse_gpu(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value or "0")
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else 0


def quota_usage() -> list[dict[str, Any]]:
    env = {"ALIBABA_CLOUD_CREDENTIALS_URI": CREDENTIAL_URI}
    result = run_cmd([sys.executable, str(PAI), "quota-usage", "--json"], timeout=120, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"quota-usage failed: {(result.stderr or result.stdout)[-1500:]}")
    value = extract_json_value((result.stdout or "") + "\n" + (result.stderr or ""))
    if not isinstance(value, list):
        raise ValueError("quota-usage did not return a JSON list")
    return [row for row in value if isinstance(row, dict)]


def visible_queuing_count(workspace: str) -> int:
    request = {"WorkspaceId": workspace, "Status": "Queuing", "PageNumber": 1, "PageSize": 100}
    env = {"ALIBABA_CLOUD_CREDENTIALS_URI": CREDENTIAL_URI}
    result = run_cmd(
        [
            sys.executable,
            str(PAI),
            "list-jobs",
            "--endpoint",
            ENDPOINT,
            "--request-json",
            json.dumps(request, separators=(",", ":")),
            "--compact",
        ],
        timeout=120,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"list-jobs failed for {workspace}: {(result.stderr or result.stdout)[-1500:]}")
    value = extract_json_value((result.stdout or "") + "\n" + (result.stderr or ""))
    body = value.get("body", value) if isinstance(value, dict) else {}
    total = body.get("TotalCount", 0) if isinstance(body, dict) else 0
    return int(total or 0)


def choose_quota(required_gpus: int, events: Path) -> dict[str, Any]:
    usage = quota_usage()
    known = {quota: workspace for workspace, quota in KNOWN_POOLS}
    candidates: list[dict[str, Any]] = []
    for row in usage:
        quota = str(row.get("QuotaId") or "")
        workspace = known.get(quota)
        if not workspace:
            continue
        details = row.get("QuotaDetails") or {}
        desired = parse_gpu((details.get("DesiredMinQuota") or {}).get("GPU"))
        used = parse_gpu((details.get("UsedQuota") or {}).get("GPU"))
        free = desired - used
        waiting = visible_queuing_count(workspace)
        candidates.append({
            "workspace": workspace,
            "quota": quota,
            "quota_name": row.get("QuotaName"),
            "desired_gpus": desired,
            "used_gpus": used,
            "free_gpus": free,
            "visible_queuing": waiting,
            "launchable": free >= required_gpus and waiting == 0,
            "queue_source": "workspace_visible_list_jobs_not_global_quota_queue",
        })
    candidates.sort(key=lambda x: (x["launchable"], x["free_gpus"], -x["visible_queuing"]), reverse=True)
    append_jsonl(events, {"time": now_cst(), "event": "quota_candidates", "required_gpus": required_gpus, "candidates": candidates})
    for candidate in candidates:
        if candidate["launchable"]:
            return candidate
    raise RuntimeError(f"no launchable quota candidate for {required_gpus} GPUs: {candidates}")


def choose_quota_with_fallback(args: argparse.Namespace, events: Path) -> dict[str, Any]:
    if args.fixed_workspace_id and args.fixed_resource_id:
        waiting = visible_queuing_count(args.fixed_workspace_id)
        free = int(args.fixed_free_gpus)
        quota = {
            "workspace": args.fixed_workspace_id,
            "quota": args.fixed_resource_id,
            "quota_name": "fixed_by_user_or_workflow",
            "desired_gpus": None,
            "used_gpus": None,
            "free_gpus": free,
            "visible_queuing": waiting,
            "launchable": waiting == 0 and free >= int(args.gpus),
            "queue_source": "fixed_workspace_visible_queue_count; free_gpu_count_not_resource_api_proven",
        }
        append_jsonl(events, {"time": now_cst(), "event": "fixed_quota_selected", "quota": quota})
        if waiting != 0:
            raise RuntimeError(f"fixed quota has visible queue: {quota}")
        if free < int(args.gpus):
            raise RuntimeError(f"fixed quota free_gpus={free} < required_gpus={args.gpus}: {quota}")
        return quota
    return choose_quota(args.gpus, events)


def wait_for_quota(args: argparse.Namespace, events: Path) -> dict[str, Any]:
    deadline = time.time() + args.quota_max_wait_sec
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            quota = choose_quota_with_fallback(args, events)
            append_jsonl(events, {"time": now_cst(), "event": "quota_ready", "attempts": attempts, "quota": quota})
            return quota
        except Exception as exc:  # noqa: BLE001 - watcher should persist transient quota state.
            append_jsonl(events, {
                "time": now_cst(),
                "event": "quota_wait",
                "attempts": attempts,
                "error": repr(exc),
                "sleep_sec": args.quota_check_interval_sec,
            })
            time.sleep(args.quota_check_interval_sec)
    raise TimeoutError(f"timed out waiting for quota after {args.quota_max_wait_sec}s")


def prepare_run(sft_dir: Path, quota: dict[str, Any], args: argparse.Namespace, events: Path) -> Path:
    run_root = ROOT / "runs" / "training" / "v3_sft_qwen25_32b"
    run_dir = run_root / args.run_id
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_v3_sft_run.py"),
        "--sft-dir",
        str(sft_dir),
        "--output-dir",
        str(run_root),
        "--run-id",
        args.run_id,
        "--base-model-id",
        args.base_model_id,
        "--dlc-workspace-id",
        str(quota["workspace"]),
        "--dlc-resource-id",
        str(quota["quota"]),
        "--dlc-gpus",
        str(args.gpus),
        "--dlc-priority",
        str(args.priority),
    ]
    result = run_cmd(cmd, timeout=1800)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "prepare_sft_run",
        "returncode": result.returncode,
        "run_dir": str(run_dir),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    })
    if result.returncode != 0:
        raise RuntimeError(f"prepare_v3_sft_run failed: {result.stderr[-1500:]}")
    return run_dir


def preflight_run(run_dir: Path, quota: dict[str, Any], args: argparse.Namespace, events: Path) -> dict[str, Any]:
    output = run_dir / "launch_preflight.json"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preflight_v3_sft_launch.py"),
        "--run-dir",
        str(run_dir),
        "--min-train-rows",
        str(args.min_train_rows),
        "--confirmed-quota",
        f"{quota['workspace']}/{quota['quota']}",
        "--confirmed-free-gpus",
        str(quota["free_gpus"]),
        "--confirmed-waiting",
        str(quota["visible_queuing"]),
        "--required-gpus",
        str(args.gpus),
        "--write-json",
        str(output),
    ]
    result = run_cmd(cmd, timeout=1800)
    payload = json.loads(output.read_text()) if output.exists() else {}
    append_jsonl(events, {
        "time": now_cst(),
        "event": "preflight_sft_run",
        "returncode": result.returncode,
        "launch_ready": payload.get("launch_ready"),
        "errors": payload.get("errors"),
        "warnings": payload.get("warnings"),
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    })
    if result.returncode != 0 or not payload.get("launch_ready"):
        raise RuntimeError(f"preflight failed for {run_dir}: {payload.get('errors')}")
    return payload


def submit_run(run_dir: Path, quota: dict[str, Any], args: argparse.Namespace, events: Path) -> dict[str, Any]:
    dry = run_cmd(["bash", str(run_dir / "pai_create_job_dry_run.sh")], timeout=300)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "pai_dry_run",
        "returncode": dry.returncode,
        "stdout_tail": dry.stdout[-1500:],
        "stderr_tail": dry.stderr[-1500:],
    })
    if dry.returncode != 0:
        raise RuntimeError(f"PAI dry-run failed: {dry.stderr[-1500:]}")
    submit = run_cmd(
        ["bash", str(run_dir / "pai_create_job.sh")],
        timeout=300,
        env={"CONFIRM_FRESH_QUOTA_FOR_V3_SFT": "1"},
    )
    combined = (submit.stdout or "") + "\n" + (submit.stderr or "")
    append_jsonl(events, {
        "time": now_cst(),
        "event": "pai_submit",
        "returncode": submit.returncode,
        "stdout_tail": submit.stdout[-1500:],
        "stderr_tail": submit.stderr[-1500:],
    })
    if submit.returncode != 0:
        raise RuntimeError(f"PAI submit failed: {combined[-2000:]}")
    value = extract_json_value(combined)
    body = value.get("body", value) if isinstance(value, dict) else {}
    job_id = body.get("JobId")
    if not job_id:
        raise RuntimeError(f"PAI submit response missing JobId: {value}")
    run_plan = json.loads((run_dir / "run_plan.json").read_text()) if (run_dir / "run_plan.json").exists() else {}
    phases = run_plan.get("phases") or []
    expected_final = phases[-1].get("expected_final") if phases and isinstance(phases[-1], dict) else None
    submission = {
        "job_id": job_id,
        "job_name": args.run_id,
        "submitted_at": now_cst(),
        "workspace_id": quota["workspace"],
        "resource_id": quota["quota"],
        "priority": args.priority,
        "gpus": args.gpus,
        "confirmed_snapshot": {
            "confirmed_waiting": quota["visible_queuing"],
            "confirmed_free_gpus": quota["free_gpus"],
            "source": quota["queue_source"],
        },
        "latest_status": {"status": "Submitted", "checked_at": now_cst()},
        "result": "Submitted",
        "expected_final": str(expected_final or (run_dir / "checkpoints" / "phase_000_2025-04" / "final")),
    }
    write_json(run_dir / "submission.json", submission)
    monitor_session = re.sub(r"[^A-Za-z0-9_=-]", "_", f"{args.run_id}_monitor")[:80]
    monitor_cmd = (
        f"cd {ROOT} && python scripts/monitor_v3_sft_job.py "
        f"--run-dir {run_dir} --job-id {job_id} --interval-sec 300 --max-polls 288 "
        f"> {run_dir / 'monitor.log'} 2>&1"
    )
    tmux = run_cmd(["tmux", "new-session", "-d", "-s", monitor_session, monitor_cmd], timeout=60)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "start_monitor",
        "returncode": tmux.returncode,
        "session": monitor_session,
        "stderr_tail": tmux.stderr[-1000:],
    })
    submission["monitor_session"] = monitor_session if tmux.returncode == 0 else None
    write_json(run_dir / "submission.json", submission)
    return submission


def render_report(events: Path) -> None:
    result = run_cmd([sys.executable, str(ROOT / "scripts" / "render_v3_training_report.py")], timeout=300)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "render_report",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a V3 target batch and prepare/optionally launch the next SFT run.")
    parser.add_argument("--batch-name", default="strict_batch1000")
    parser.add_argument("--run-id", default="v3_sft_strict_batch1000_auto")
    parser.add_argument("--base-model-id", default="qwen25_32b_instruct")
    parser.add_argument("--min-train-rows", type=int, default=800)
    parser.add_argument("--train-source", choices=["train", "all"], default="train")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--priority", type=int, default=6)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--fixed-workspace-id", default=None)
    parser.add_argument("--fixed-resource-id", default=None)
    parser.add_argument(
        "--fixed-free-gpus",
        type=int,
        default=0,
        help="Manual free-GPU count for fixed quota fallback when resource quota API is unavailable.",
    )
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-wait-sec", type=int, default=12 * 3600)
    parser.add_argument("--quota-check-interval-sec", type=int, default=300)
    parser.add_argument("--quota-max-wait-sec", type=int, default=6 * 3600)
    parser.add_argument("--submit-if-launch-ready", action="store_true")
    args = parser.parse_args()

    events = ROOT / "runs" / "training" / "v3_sft_qwen25_32b" / args.run_id / "watch_events.jsonl"
    try:
        append_jsonl(events, {"time": now_cst(), "event": "watch_started", "args": vars(args)})
        source_sft_dir, rows = wait_for_collate(args, events)
        sft_dir, train_rows = materialize_sft_dir(source_sft_dir, args, events)
        score_sft_targets(sft_dir, args, events)
        quota = wait_for_quota(args, events)
        run_dir = prepare_run(sft_dir, quota, args, events)
        preflight_run(run_dir, quota, args, events)
        submission: dict[str, Any] | None = None
        if args.submit_if_launch_ready:
            submission = submit_run(run_dir, quota, args, events)
        render_report(events)
        append_jsonl(events, {
            "time": now_cst(),
            "event": "watch_completed",
            "sft_dir": str(sft_dir),
            "source_sft_dir": str(source_sft_dir),
            "ready_rows": rows,
            "train_rows": train_rows,
            "train_source": args.train_source,
            "run_dir": str(run_dir),
            "submitted": bool(submission),
            "submission": submission,
        })
        return 0
    except Exception as exc:  # noqa: BLE001 - long watcher must persist errors to disk.
        append_jsonl(events, {"time": now_cst(), "event": "watch_failed", "error": repr(exc)})
        print(repr(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
