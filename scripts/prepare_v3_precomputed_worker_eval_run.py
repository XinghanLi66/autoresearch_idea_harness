#!/usr/bin/env python3
# Internal Alibaba PAI-DLC helper — not needed for external reproduction (see REPRODUCE.md).
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json


DEFAULT_BATCH_ROOT = ROOT / "runs" / "v3_checkpoint_proposal_batch" / "dlc_mls10_strict1000" / "output"
DEFAULT_TASKS = ["dl_lr_schedule", "dl_activation_function"]
# Real Claude Code binary used inside worker pods; set CLAUDE_REAL_BIN explicitly.
DEFAULT_CLAUDE_REAL_BIN = os.environ.get("CLAUDE_REAL_BIN", "")


def _quote_cmd(args: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _validate_claude_real_bin(path: str) -> list[str]:
    errors: list[str] = []
    raw = Path(path)
    resolved = raw.resolve(strict=False)
    if "claude-agent-proxy" in str(resolved):
        errors.append(
            "claude_real_bin must point to the real Claude Code binary, not the claude-agent-proxy wrapper"
        )
    if not raw.exists():
        errors.append(f"claude_real_bin does not exist on this filesystem: {path}")
    elif not raw.is_file():
        errors.append(f"claude_real_bin is not a file: {path}")
    elif not os.access(raw, os.X_OK):
        errors.append(f"claude_real_bin is not executable: {path}")
    return errors


def _load_summary_rows(batch_root: Path) -> list[dict[str, Any]]:
    rows = []
    path = batch_root / "summary_rows.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _task_dir_for(row: dict[str, Any], batch_root: Path) -> Path:
    raw = row.get("task_dir")
    if raw:
        return Path(str(raw))
    idx = int(row["index"])
    return batch_root / f"{idx:02d}_{row['task']}"


def _select_tasks(batch_root: Path, tasks: list[str]) -> list[dict[str, Any]]:
    selected = []
    rows = _load_summary_rows(batch_root)
    by_task = {str(row.get("task")): row for row in rows}
    for task in tasks:
        if task not in by_task:
            raise ValueError(f"Task {task!r} not found in {batch_root / 'summary_rows.jsonl'}")
        row = dict(by_task[task])
        task_dir = _task_dir_for(row, batch_root)
        for name in ("proposal.txt", "task_packet.json", "proposal_quality.json"):
            path = task_dir / name
            if not path.exists():
                raise FileNotFoundError(path)
        row["task_dir"] = str(task_dir)
        row["proposal"] = str(task_dir / "proposal.txt")
        row["task_packet"] = str(task_dir / "task_packet.json")
        row["proposal_quality"] = str(task_dir / "proposal_quality.json")
        selected.append(row)
    return selected


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    dlc = dict((cfg.get("v3_training") or {}).get("dlc") or {})
    batch_root = args.batch_root.resolve()
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_tasks(batch_root, args.tasks)
    eval_root = run_dir / "output"
    eval_root.mkdir(parents=True, exist_ok=True)

    shell_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {ROOT}",
        'POD_NAME="${K8S_POD_NAME:-${POD_NAME:-${HOSTNAME:-}}}"',
        'echo "[worker-eval] pod guard: ${POD_NAME}"',
        'if [[ "${POD_NAME}" == *aimaster* ]]; then',
        '  echo "[worker-eval] skipping aimaster pod"',
        "  exit 0",
        "fi",
        f"export CLAUDE_REAL_BIN={shlex.quote(args.claude_real_bin)}",
        f"mkdir -p {shlex.quote(str(eval_root))}",
        "pids=()",
        "names=()",
        "rc=0",
    ]
    def append_wait_batch() -> None:
        shell_lines.extend(
            [
                "for i in \"${!pids[@]}\"; do",
                "  if ! wait \"${pids[$i]}\"; then",
                "    echo \"[worker-eval] ${names[$i]} failed with rc=$?\" >&2",
                "    rc=1",
                "  fi",
                "done",
                "pids=()",
                "names=()",
            ]
        )

    max_parallel = max(1, min(args.max_parallel, args.gpus))
    for idx, row in enumerate(selected):
        gpu = idx % max(1, args.gpus)
        sample_id = f"{row['task']}__{row['subtask']}__{args.module_id}__worker_eval"
        log_path = run_dir / f"{row['task']}.worker_eval.log"
        cmd = [
            args.python_bin,
            str(ROOT / "scripts" / "run_precomputed_proposal_worker.py"),
            "--config",
            str(Path(args.config).resolve()),
            "--task",
            str(row["task"]),
            "--subtask",
            str(row["subtask"]),
            "--proposal",
            str(row["proposal"]),
            "--task-packet",
            str(row["task_packet"]),
            "--proposal-quality",
            str(row["proposal_quality"]),
            "--output-root",
            str(eval_root),
            "--sample-id",
            sample_id,
            "--module-id",
            args.module_id,
            "--model-id",
            args.model_id,
            "--worker-mode",
            args.worker_mode,
            "--gpu",
            str(gpu),
            "--worker-timeout",
            str(args.worker_timeout),
            "--result-wait-timeout",
            str(args.result_wait_timeout),
            "--no-eval-wait-timeout",
            str(args.no_eval_wait_timeout),
            "--max-turns",
            str(args.max_turns),
            "--force",
        ]
        shell_lines.extend(
            [
                f"echo '[worker-eval] launching {row['task']} on GPU {gpu}' | tee -a {shlex.quote(str(log_path))}",
                f"({_quote_cmd(cmd)}) > {shlex.quote(str(log_path))} 2>&1 &",
                "pids+=(\"$!\")",
                f"names+=({shlex.quote(str(row['task']))})",
            ]
        )
        if (idx + 1) % max_parallel == 0:
            append_wait_batch()
    append_wait_batch()
    shell_lines.extend(
        [
            f"{_quote_cmd([args.python_bin, '-c', _summary_snippet(), str(eval_root), str(run_dir / 'worker_eval_job_status.json')])}",
            "exit \"$rc\"",
            "",
        ]
    )

    command_file = run_dir / "dlc_command_skeleton.sh"
    command_file.write_text("\n".join(shell_lines))
    command_file.chmod(0o755)

    pai_config = run_dir / "pai_job_config.yaml"
    pai_config.write_text(yaml.safe_dump({"data_sources": dlc.get("data_sources") or []}, sort_keys=False))

    job_name = args.job_name or f"v3_worker_eval_strict1000_{int(time.time())}"
    create_args = [
        "python",
        os.environ.get("PAI_MANAGE", "/root/.claude/skills/pai/scripts/pai_manage.py"),
        "create-job",
        "--endpoint",
        str(dlc.get("endpoint")),
        "--name",
        job_name,
        "--command-file",
        str(command_file),
        "--workspace-id",
        str(args.workspace_id or dlc.get("workspace_id")),
        "--resource-id",
        str(args.resource_id or dlc.get("resource_id")),
        "--image",
        str(dlc.get("image")),
        "--gpus",
        str(args.gpus),
        "--cpus",
        str(args.cpus),
        "--memory",
        args.memory,
        "--shared-memory",
        args.shared_memory,
        "--max-running-minutes",
        str(args.max_running_minutes),
        "--priority",
        str(args.priority),
        "--env",
        "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(args.gpus)),
        "--env",
        "CLAUDE_REAL_BIN=" + args.claude_real_bin,
        "--enable-rdma",
        "true" if args.enable_rdma else "false",
        "--config",
        str(pai_config),
    ]
    dry_run_file = run_dir / "pai_create_job_dry_run.sh"
    dry_run_file.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "export ALIBABA_CLOUD_CREDENTIALS_URI=${ALIBABA_CLOUD_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}",
                _quote_cmd(create_args + ["--dry-run"]),
                "",
            ]
        )
    )
    dry_run_file.chmod(0o755)

    submit_file = run_dir / "pai_create_job.sh"
    submit_file.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                ': "${CONFIRM_FRESH_QUOTA_FOR_V3_WORKER_EVAL:?Set this to 1 only after a fresh quota snapshot confirms enough free GPUs.}"',
                'if [[ "${CONFIRM_FRESH_QUOTA_FOR_V3_WORKER_EVAL}" != "1" ]]; then',
                '  echo "CONFIRM_FRESH_QUOTA_FOR_V3_WORKER_EVAL must equal 1" >&2',
                "  exit 2",
                "fi",
                "export ALIBABA_CLOUD_CREDENTIALS_URI=${ALIBABA_CLOUD_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}",
                _quote_cmd(create_args),
                "",
            ]
        )
    )
    submit_file.chmod(0o755)

    errors = _validate_claude_real_bin(args.claude_real_bin)
    if not any(ds.get("mount") and ds.get("id") for ds in dlc.get("data_sources") or []):
        errors.append("DLC data_sources must include a mounted datasource with an id")
    for key in ("endpoint", "workspace_id", "resource_id", "image"):
        if key in {"workspace_id", "resource_id"} and (args.workspace_id or args.resource_id):
            continue
        if not dlc.get(key):
            errors.append(f"missing v3_training.dlc.{key}")
    if args.priority != 6:
        errors.append(f"priority must be 6 for this workflow, got {args.priority}")

    summary = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ready_to_dry_run": not errors,
        "ready_to_submit": False,
        "submit_guard_env": "CONFIRM_FRESH_QUOTA_FOR_V3_WORKER_EVAL=1",
        "errors": errors,
        "job_name": job_name,
        "batch_root": str(batch_root),
        "output_dir": str(eval_root),
        "selected": selected,
        "module_id": args.module_id,
        "model_id": args.model_id,
        "resources": {
            "workspace_id": str(args.workspace_id or dlc.get("workspace_id")),
            "resource_id": str(args.resource_id or dlc.get("resource_id")),
            "gpus": args.gpus,
            "cpus": args.cpus,
            "memory": args.memory,
            "shared_memory": args.shared_memory,
            "priority": args.priority,
            "max_running_minutes": args.max_running_minutes,
            "claude_real_bin": args.claude_real_bin,
            "max_parallel": max_parallel,
        },
        "files": {
            "command": str(command_file),
            "pai_job_config": str(pai_config),
            "dry_run": str(dry_run_file),
            "submit": str(submit_file),
        },
    }
    write_json(run_dir / "run_plan.json", summary)
    return summary


def _summary_snippet() -> str:
    return (
        "import json,sys; from pathlib import Path; "
        "root=Path(sys.argv[1]); out=Path(sys.argv[2]); rows=[]; "
        "\nfor p in sorted(root.glob('*/summary.json')):\n"
        "    try: rows.append(json.loads(p.read_text()))\n"
        "    except Exception as e: rows.append({'sample_id':p.parent.name,'error':str(e)})\n"
        "out.write_text(json.dumps({'root':str(root),'samples':rows}, indent=2, ensure_ascii=False))\n"
        "print(out.read_text())"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare DLC worker evals for precomputed V3 proposals.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "v3_precomputed_worker_eval" / "dlc_strict1000_lr_act")
    parser.add_argument("--job-name", default="v3_worker_eval_strict1000_lr_act")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--module-id", default="v3_sft_strict1000")
    parser.add_argument("--model-id", default="qwen25_32b_v3_sft_strict1000")
    parser.add_argument("--worker-mode", choices=["claude", "fixture", "skip"], default="claude")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=9999)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--memory", default="300Gi")
    parser.add_argument("--shared-memory", default="64Gi")
    parser.add_argument("--max-running-minutes", type=int, default=180)
    parser.add_argument("--priority", type=int, default=6)
    parser.add_argument("--enable-rdma", action="store_true")
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--resource-id", default=None)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    parser.add_argument("--result-wait-timeout", type=int, default=7200)
    parser.add_argument("--no-eval-wait-timeout", type=int, default=300)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--claude-real-bin", default=DEFAULT_CLAUDE_REAL_BIN)
    args = parser.parse_args()
    summary = prepare(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
