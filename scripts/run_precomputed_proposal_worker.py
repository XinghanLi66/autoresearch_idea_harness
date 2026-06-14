#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.benchmarking import MLS_SUBTASKS, get_task, verify_signed_result
from autoresearch_idea_harness.formal_sweep import utc_now, write_jsonl_append
from autoresearch_idea_harness.io import load_config, stable_id, write_json
from autoresearch_idea_harness.prompts import worker_prompt


def _primary_subtask(task_name: str) -> str:
    subtasks = MLS_SUBTASKS.get(task_name)
    if not subtasks:
        raise ValueError(f"No primary subtask registered for {task_name}")
    return subtasks[0]


def _event(events_path: Path, event_name: str, **fields: Any) -> None:
    row = {"ts": utc_now(), "event": event_name, **fields}
    write_jsonl_append(events_path, row)


def _copy_worker_artifacts(sample_dir: Path, workspace: Path) -> None:
    for name in ("eval.log", "editable_region.py", "baseline.py", "submission.csv"):
        src = workspace / name
        if src.exists() and src.is_file():
            shutil.copy2(src, sample_dir / name)


def _parse_worker_result(task: Any, proposal_id: str, sample_dir: Path, elapsed_s: float) -> dict[str, Any]:
    workspace = sample_dir / "workspace"
    result_path = workspace / "result.json"
    if not result_path.exists():
        return {
            "status": "error",
            "proposal_id": proposal_id,
            "error": "result.json not written",
            "elapsed_s": round(elapsed_s, 3),
        }
    try:
        raw = json.loads(result_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "proposal_id": proposal_id,
            "error": f"invalid result.json: {exc}",
            "elapsed_s": round(elapsed_s, 3),
        }
    shutil.copy2(result_path, sample_dir / "result.json")
    _copy_worker_artifacts(sample_dir, workspace)
    metric, error = verify_signed_result(raw)
    if metric is None:
        return {
            "status": "error",
            "proposal_id": proposal_id,
            "error": error,
            "raw_result": raw,
            "elapsed_s": round(elapsed_s, 3),
        }
    parsed = {
        "status": "done",
        "proposal_id": proposal_id,
        "val_metric": metric,
        "baseline_metric": task.baseline_metric(),
        "pass_metric": task.pass_metric(),
        "improvement": round(task.improvement(metric), 6),
        "passed": task.passed(metric),
        "raw_result": raw,
        "elapsed_s": round(elapsed_s, 3),
    }
    write_json(sample_dir / "result.json", {**raw, "_parsed": parsed})
    return parsed


def _fixture_worker(task: Any, proposal_id: str, sample_dir: Path) -> dict[str, Any]:
    import hashlib
    import hmac

    metric = task.pass_metric() + (0.01 if not task.lower_is_better else -0.01)
    secret = os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret")
    payload = f"{float(metric):.6f}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    write_json(sample_dir / "workspace" / "result.json", {"val_metric": metric, "_sig": sig, "fixture": True})
    (sample_dir / "worker.log").write_text("fixture worker\n")
    (sample_dir / "workspace" / "eval.log").write_text("fixture eval\n")
    return _parse_worker_result(task, proposal_id, sample_dir, elapsed_s=0.0)


def _wait_for_result(sample_dir: Path, events_path: Path, *, result_wait_timeout: int, no_eval_wait_timeout: int) -> None:
    workspace = sample_dir / "workspace"
    result_path = workspace / "result.json"
    if result_path.exists() or result_wait_timeout <= 0:
        return
    eval_log = workspace / "eval.log"
    saw_eval = eval_log.exists()
    started = time.time()
    deadline = started + result_wait_timeout
    no_eval_deadline = started + no_eval_wait_timeout
    _event(events_path, "worker_result_wait_started", result_wait_timeout=result_wait_timeout, no_eval_wait_timeout=no_eval_wait_timeout)
    while time.time() < deadline:
        if result_path.exists():
            _event(events_path, "worker_result_wait_done", elapsed_s=round(time.time() - started, 3))
            return
        if eval_log.exists():
            saw_eval = True
        if not saw_eval and time.time() >= no_eval_deadline:
            _event(events_path, "worker_result_wait_no_eval_log", elapsed_s=round(time.time() - started, 3))
            return
        time.sleep(10)
    _event(events_path, "worker_result_wait_timeout", elapsed_s=round(time.time() - started, 3))


def _invoke_claude_worker(
    cfg: dict[str, Any],
    sample_dir: Path,
    *,
    gpu: str,
    max_turns: int,
    worker_timeout: int,
    silent_timeout: int,
    events_path: Path,
) -> None:
    workspace = sample_dir / "workspace"
    prompt_file = sample_dir / "worker_prompt.txt"
    log_file = sample_dir / "worker.log"
    claude_cmd = str((cfg.get("end_to_end") or {}).get("claude_cmd", "/newcpfs/lxh/claude-home-agent1/run_claude.sh"))
    cmd = [
        claude_cmd,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(max_turns),
        "--allowedTools",
        "Bash,Read,Edit,Write",
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": gpu,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    result_path = workspace / "result.json"
    with prompt_file.open() as stdin_f, log_file.open("a") as log_f:
        proc = subprocess.Popen(cmd, stdin=stdin_f, stdout=log_f, stderr=subprocess.STDOUT, cwd=workspace, env=env)
        started = time.time()
        while proc.poll() is None:
            elapsed = time.time() - started
            log_size = log_file.stat().st_size if log_file.exists() else 0
            if result_path.exists() and result_path.stat().st_mtime >= started - 1:
                _event(events_path, "worker_result_detected_while_process_running", elapsed_s=round(elapsed, 3))
                log_f.write("\n[runner] result.json detected; terminating worker process after result capture\n")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    _event(events_path, "worker_result_detected_kill_after_grace", elapsed_s=round(time.time() - started, 3))
                    proc.kill()
                    proc.wait()
                return
            if silent_timeout > 0 and log_size == 0 and elapsed >= silent_timeout:
                _event(events_path, "worker_silent_timeout", elapsed_s=round(elapsed, 3), silent_timeout=silent_timeout)
                proc.kill()
                proc.wait()
                log_f.write(f"\n[runner] worker produced no log output after {silent_timeout}s\n")
                return
            if elapsed >= worker_timeout:
                _event(events_path, "worker_timeout", elapsed_s=round(elapsed, 3), worker_timeout=worker_timeout)
                proc.kill()
                proc.wait()
                log_f.write(f"\n[runner] worker timeout after {worker_timeout}s\n")
                return
            time.sleep(5)
        if proc.returncode:
            _event(events_path, "worker_process_exited_nonzero", returncode=proc.returncode)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    task_name = args.task
    subtask = args.subtask or _primary_subtask(task_name)
    task = get_task(cfg, task_name, subtask)
    proposal_text = args.proposal.read_text()
    source_packet = json.loads(args.task_packet.read_text()) if args.task_packet else {}
    source_quality = json.loads(args.proposal_quality.read_text()) if args.proposal_quality and args.proposal_quality.exists() else {}

    module_id = args.module_id
    sample_id = args.sample_id or f"{task_name}__{subtask}__{module_id}__precomputed"
    sample_dir = args.output_root.resolve() / sample_id
    if sample_dir.exists() and not args.force and (sample_dir / "summary.json").exists():
        return json.loads((sample_dir / "summary.json").read_text())
    sample_dir.mkdir(parents=True, exist_ok=True)
    events_path = sample_dir / "events.jsonl"
    _event(events_path, "sample_started", task=task_name, subtask=subtask, module_id=module_id)

    task_packet = task.task_packet(cfg)
    task_packet.update({
        "run_id": sample_id,
        "sample_id": sample_id,
        "subtask": subtask,
        "module_id": module_id,
        "sweep_root": str(args.output_root.resolve()),
        "precomputed_proposal_source": str(args.proposal.resolve()),
    })
    if source_packet.get("research_question"):
        task_packet["research_question"] = source_packet["research_question"]
    if source_packet.get("condition_strategy"):
        task_packet["condition_strategy"] = source_packet["condition_strategy"]
    write_json(sample_dir / "task_packet.json", task_packet)
    (sample_dir / "proposal.txt").write_text(proposal_text)
    write_json(sample_dir / "proposal_source.json", {
        "proposal_path": str(args.proposal.resolve()),
        "task_packet_path": str(args.task_packet.resolve()) if args.task_packet else None,
        "proposal_quality_path": str(args.proposal_quality.resolve()) if args.proposal_quality else None,
        "proposal_quality": source_quality,
    })

    proposal_id = stable_id("prop", sample_id, module_id, proposal_text, length=14)
    proposal = {
        "proposal_id": proposal_id,
        "label": module_id,
        "module_id": module_id,
        "generator_id": module_id,
        "strategy": "with_research_question",
        "model_id": args.model_id,
        "text": proposal_text,
        "empty_control": False,
        "precomputed": True,
    }
    write_json(sample_dir / "proposal.json", proposal)

    workspace = sample_dir / "workspace"
    task.setup_workspace(workspace)
    prompt_text = worker_prompt(task_packet, proposal_text)
    (sample_dir / "worker_prompt.txt").write_text(prompt_text)
    (sample_dir / "master_worker_dialogue.jsonl").write_text("")

    if args.worker_mode == "skip":
        worker_result = {"status": "skipped", "proposal_id": proposal_id}
        write_json(sample_dir / "result.json", worker_result)
        _event(events_path, "worker_skipped")
    elif args.worker_mode == "fixture":
        worker_result = _fixture_worker(task, proposal_id, sample_dir)
        _event(events_path, "fixture_worker_done", passed=worker_result.get("passed"), metric=worker_result.get("val_metric"))
    else:
        _event(events_path, "worker_started", proposal_id=proposal_id, gpu=args.gpu)
        started = time.time()
        _invoke_claude_worker(
            cfg,
            sample_dir,
            gpu=args.gpu,
            max_turns=args.max_turns,
            worker_timeout=args.worker_timeout,
            silent_timeout=args.no_eval_wait_timeout,
            events_path=events_path,
        )
        _wait_for_result(
            sample_dir,
            events_path,
            result_wait_timeout=args.result_wait_timeout,
            no_eval_wait_timeout=args.no_eval_wait_timeout,
        )
        worker_result = _parse_worker_result(task, proposal_id, sample_dir, elapsed_s=time.time() - started)
        _event(events_path, "worker_done", status=worker_result.get("status"), passed=worker_result.get("passed"), metric=worker_result.get("val_metric"))
        if worker_result.get("status") != "done":
            write_json(sample_dir / "error.json", {"ts": utc_now(), **worker_result})

    summary = {
        "run_id": sample_id,
        "sample_id": sample_id,
        "task": task_name,
        "subtask": subtask,
        "task_type": task.task_type,
        "module_id": module_id,
        "proposal_id": proposal_id,
        "precomputed": True,
        "baseline_metric": task_packet.get("baseline_metric"),
        "pass_threshold": task_packet.get("pass_threshold"),
        "pass_metric": task_packet.get("pass_metric"),
        "source_quality_score": source_quality.get("score"),
        "source_quality_verdict": source_quality.get("verdict"),
        "worker_result": worker_result,
        "paths": {
            "task_packet": "task_packet.json",
            "proposal": "proposal.txt",
            "proposal_source": "proposal_source.json",
            "worker_prompt": "worker_prompt.txt",
            "worker_log": "worker.log",
            "eval_log": "eval.log",
            "result": "result.json",
        },
    }
    write_json(sample_dir / "summary.json", summary)
    write_jsonl_append(args.output_root.resolve() / "registry.jsonl", {
        "ts": utc_now(),
        "sample_id": sample_id,
        "task": task_name,
        "subtask": subtask,
        "module_id": module_id,
        "status": worker_result.get("status"),
        "passed": worker_result.get("passed"),
        "val_metric": worker_result.get("val_metric"),
        "path": sample_id,
    })
    _event(events_path, "sample_done", status=worker_result.get("status"), passed=worker_result.get("passed"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a worker evaluation from an already generated proposal.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--task-packet", type=Path, default=None)
    parser.add_argument("--proposal-quality", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "v3_precomputed_worker_eval" / "strict1000_mls_pass_tasks")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--module-id", default="v3_sft_strict1000_precomputed")
    parser.add_argument("--model-id", default="qwen25_32b_v3_sft_strict1000")
    parser.add_argument("--worker-mode", choices=["claude", "fixture", "skip"], default="claude")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--result-wait-timeout", type=int, default=7200)
    parser.add_argument("--no-eval-wait-timeout", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = run_worker(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if (summary.get("worker_result") or {}).get("status") not in {"error"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
