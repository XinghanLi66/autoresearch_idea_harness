#!/usr/bin/env python3
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
PAI = Path("/root/.claude/skills/pai/scripts/pai_manage.py")
ENDPOINT = "pai-dlc.ap-southeast-1.aliyuncs.com"
CREDENTIAL_URI = "http://localhost:7002/api/v1/credentials/0"

MLS10_TASKS = [
    "dl_lr_schedule",
    "dl_activation_function",
    "cv_data_augmentation",
    "dl_weight_initialization",
    "cv_classification_loss",
    "cv_sample_weighting",
    "cv_pooling_aggregation",
    "cv_multitask_loss",
    "dl_regularization",
    "dl_residual_connection",
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
    return int((body or {}).get("TotalCount") or 0)


def model_ready(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists() or not (model_dir / "tokenizer.json").exists():
        return False
    has_sharded_weights = (model_dir / "model.safetensors.index.json").exists() and bool(list(model_dir.glob("model-*.safetensors")))
    has_single_weight = (model_dir / "model.safetensors").exists()
    return has_sharded_weights or has_single_weight


def wait_for_model(model_dir: Path, args: argparse.Namespace, events: Path) -> None:
    deadline = time.time() + args.max_wait_sec
    last_state: str | None = None
    while time.time() < deadline:
        ready = model_ready(model_dir)
        state = "ready" if ready else "waiting"
        if state != last_state:
            append_jsonl(events, {"time": now_cst(), "event": "model_wait", "model_dir": str(model_dir), "state": state})
            last_state = state
        if ready:
            return
        time.sleep(args.interval_sec)
    raise TimeoutError(f"timed out waiting for final model at {model_dir}")


def require_low_queue(workspace: str, max_visible_queue: int, events: Path, label: str) -> None:
    waiting = visible_queuing_count(workspace)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "queue_check",
        "label": label,
        "workspace": workspace,
        "visible_queuing": waiting,
        "max_visible_queue": max_visible_queue,
    })
    if waiting > max_visible_queue:
        raise RuntimeError(f"{label} queue too deep for workspace {workspace}: {waiting} > {max_visible_queue}")


def extract_job_id(text: str) -> str | None:
    value = extract_json_value(text)
    body = value.get("body", value) if isinstance(value, dict) else {}
    job_id = body.get("JobId") if isinstance(body, dict) else None
    return str(job_id) if job_id else None


def run_guarded_submit(script: Path, guard_env: str, events: Path, label: str, timeout: int = 600) -> str:
    env = {"ALIBABA_CLOUD_CREDENTIALS_URI": CREDENTIAL_URI, guard_env: "1"}
    result = run_cmd(["bash", str(script)], timeout=timeout, env=env)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "submit",
        "label": label,
        "script": str(script),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    })
    if result.returncode != 0:
        raise RuntimeError(f"{label} submit failed: {(result.stderr or result.stdout)[-2000:]}")
    job_id = extract_job_id((result.stdout or "") + "\n" + (result.stderr or ""))
    if not job_id:
        raise RuntimeError(f"{label} submit did not return a JobId")
    return job_id


def prepare_proposal_batch(model_dir: Path, args: argparse.Namespace, events: Path) -> Path:
    run_dir = ROOT / "runs" / "v3_checkpoint_proposal_batch" / args.proposal_run_name
    summary_path = run_dir / "run_plan.json"
    if summary_path.exists() and (run_dir / "pai_create_job.sh").exists():
        append_jsonl(events, {"time": now_cst(), "event": "proposal_prepare_reused", "run_dir": str(run_dir)})
        return run_dir
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_v3_proposal_batch_run.py"),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(run_dir),
        "--job-name",
        args.proposal_job_name,
        "--gpus",
        str(args.proposal_gpus),
        "--priority",
        str(args.priority),
        "--max-running-minutes",
        str(args.proposal_max_running_minutes),
        "--tasks",
        *args.tasks,
    ]
    result = run_cmd(cmd, timeout=600)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "proposal_prepare",
        "returncode": result.returncode,
        "run_dir": str(run_dir),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    })
    if result.returncode != 0:
        raise RuntimeError(f"proposal prepare failed: {(result.stderr or result.stdout)[-2000:]}")
    return run_dir


def wait_for_proposals(run_dir: Path, args: argparse.Namespace, events: Path) -> None:
    summary = run_dir / "output" / "summary.json"
    deadline = time.time() + args.max_wait_sec
    last_ok = -1
    while time.time() < deadline:
        ok = 0
        error = None
        if summary.exists():
            try:
                payload = json.loads(summary.read_text())
                ok = int(payload.get("ok_count") or 0)
                error = int(payload.get("error_count") or 0)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
        if ok != last_ok:
            append_jsonl(events, {
                "time": now_cst(),
                "event": "proposal_wait",
                "summary": str(summary),
                "ok_count": ok,
                "error": error,
            })
            last_ok = ok
        if ok >= len(args.tasks):
            return
        time.sleep(args.interval_sec)
    raise TimeoutError(f"timed out waiting for proposals in {run_dir}")


def prepare_worker_eval(proposal_run_dir: Path, args: argparse.Namespace, events: Path) -> Path:
    run_dir = ROOT / "runs" / "v3_precomputed_worker_eval" / args.worker_eval_run_name
    summary_path = run_dir / "run_plan.json"
    if summary_path.exists() and (run_dir / "pai_create_job.sh").exists():
        append_jsonl(events, {"time": now_cst(), "event": "worker_prepare_reused", "run_dir": str(run_dir)})
        return run_dir
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_v3_precomputed_worker_eval_run.py"),
        "--batch-root",
        str(proposal_run_dir / "output"),
        "--tasks",
        *args.tasks,
        "--output-dir",
        str(run_dir),
        "--job-name",
        args.worker_eval_job_name,
        "--module-id",
        args.module_id,
        "--model-id",
        args.model_id,
        "--gpus",
        str(args.worker_gpus),
        "--max-parallel",
        str(args.worker_max_parallel),
        "--priority",
        str(args.priority),
        "--workspace-id",
        args.workspace_id,
        "--resource-id",
        args.resource_id,
        "--max-running-minutes",
        str(args.worker_max_running_minutes),
        "--worker-timeout",
        str(args.worker_timeout),
        "--result-wait-timeout",
        str(args.result_wait_timeout),
        "--no-eval-wait-timeout",
        str(args.no_eval_wait_timeout),
        "--max-turns",
        str(args.max_turns),
    ]
    result = run_cmd(cmd, timeout=600)
    append_jsonl(events, {
        "time": now_cst(),
        "event": "worker_prepare",
        "returncode": result.returncode,
        "run_dir": str(run_dir),
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    })
    if result.returncode != 0:
        raise RuntimeError(f"worker prepare failed: {(result.stderr or result.stdout)[-2000:]}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="After V3 SFT finishes, launch proposal generation and worker eval.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "runs" / "training" / "v3_sft_qwen25_32b" / "v3_sft_v1sem_cot_strict929_16k" / "checkpoints" / "phase_000_2025-04" / "final",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "training" / "v3_sft_qwen25_32b" / "v3_sft_v1sem_cot_strict929_16k")
    parser.add_argument("--proposal-run-name", default="dlc_mls10_v1sem_cot_sft_16k")
    parser.add_argument("--proposal-job-name", default="v3_proposal_batch_mls10_v1sem_cot_sft_16k")
    parser.add_argument("--worker-eval-run-name", default="dlc_mls10_v1sem_cot_sft_16k")
    parser.add_argument("--worker-eval-job-name", default="v3_worker_eval_mls10_v1sem_cot_sft_16k")
    parser.add_argument("--module-id", default="qwen25_32b_v1sem_cot_sft_16k")
    parser.add_argument("--model-id", default="qwen25_32b_v1sem_cot_sft_16k")
    parser.add_argument("--workspace-id", default="262162")
    parser.add_argument("--resource-id", default="quota1shcr2h7uae")
    parser.add_argument("--priority", type=int, default=6)
    parser.add_argument("--proposal-gpus", type=int, default=2)
    parser.add_argument("--worker-gpus", type=int, default=8)
    parser.add_argument("--worker-max-parallel", type=int, default=8)
    parser.add_argument("--proposal-max-running-minutes", type=int, default=120)
    parser.add_argument("--worker-max-running-minutes", type=int, default=480)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--result-wait-timeout", type=int, default=7200)
    parser.add_argument("--no-eval-wait-timeout", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--interval-sec", type=int, default=300)
    parser.add_argument("--max-wait-sec", type=int, default=24 * 3600)
    parser.add_argument("--max-visible-queue", type=int, default=0)
    parser.add_argument("--tasks", nargs="*", default=MLS10_TASKS)
    args = parser.parse_args()

    events = args.run_dir / "post_sft_eval_events.jsonl"
    try:
        append_jsonl(events, {"time": now_cst(), "event": "post_sft_watch_started", "args": vars(args) | {"model_dir": str(args.model_dir), "run_dir": str(args.run_dir)}})
        wait_for_model(args.model_dir, args, events)
        proposal_run_dir = prepare_proposal_batch(args.model_dir, args, events)
        if not (proposal_run_dir / "submission.json").exists():
            require_low_queue(args.workspace_id, args.max_visible_queue, events, "proposal_batch")
            proposal_job_id = run_guarded_submit(
                proposal_run_dir / "pai_create_job.sh",
                "CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_BATCH",
                events,
                "proposal_batch",
            )
            write_json(proposal_run_dir / "submission.json", {"job_id": proposal_job_id, "job_name": args.proposal_job_name})
        else:
            append_jsonl(events, {"time": now_cst(), "event": "proposal_submission_reused", "path": str(proposal_run_dir / "submission.json")})
        wait_for_proposals(proposal_run_dir, args, events)
        worker_run_dir = prepare_worker_eval(proposal_run_dir, args, events)
        if not (worker_run_dir / "submission.json").exists():
            require_low_queue(args.workspace_id, args.max_visible_queue, events, "worker_eval")
            worker_job_id = run_guarded_submit(
                worker_run_dir / "pai_create_job.sh",
                "CONFIRM_FRESH_QUOTA_FOR_V3_WORKER_EVAL",
                events,
                "worker_eval",
            )
            write_json(worker_run_dir / "submission.json", {"job_id": worker_job_id, "job_name": args.worker_eval_job_name})
        else:
            append_jsonl(events, {"time": now_cst(), "event": "worker_submission_reused", "path": str(worker_run_dir / "submission.json")})
        append_jsonl(events, {"time": now_cst(), "event": "post_sft_launch_done"})
        return 0
    except Exception as exc:  # noqa: BLE001
        append_jsonl(events, {"time": now_cst(), "event": "post_sft_launch_failed", "error": str(exc), "type": type(exc).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
