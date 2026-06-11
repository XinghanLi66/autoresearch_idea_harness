#!/usr/bin/env python3
"""Fail-closed preflight for a V3 checkpoint proposal smoke DLC run.

This script does not submit jobs and does not query PAI. It verifies local
artifacts and can optionally include a fresh user-confirmed quota snapshot.
Without a confirmed snapshot, artifacts can be ready while launch_ready remains
false.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs" / "v3_checkpoint_proposal_smoke" / "dlc_activation"
SUBMIT_GUARD_ENV = "CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_SMOKE"
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|authorization)\s*[:=]\s*[A-Za-z0-9_./+=-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(message)


def _file_mode_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def _extract_command_tokens(path: Path) -> list[str]:
    text = path.read_text()
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#") and "generate_v3_checkpoint_proposal.py" in line
    ]
    if not lines:
        return []
    command = lines[0].split(" 2>&1", 1)[0]
    try:
        return shlex.split(command)
    except ValueError:
        return []


def validate_final_model(model_dir: Path, errors: list[str]) -> None:
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        if not (model_dir / name).exists():
            add_error(errors, f"missing final model file: {model_dir / name}")
    if not list(model_dir.glob("model-*.safetensors")):
        add_error(errors, f"missing model safetensor shards in {model_dir}")


def validate_files(run_dir: Path, plan: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    files = plan.get("files") or {}
    required_files = {
        "command": files.get("command"),
        "pai_job_config": files.get("pai_job_config"),
        "dry_run": files.get("dry_run"),
        "submit": files.get("submit"),
    }
    for key, value in required_files.items():
        path = Path(str(value or ""))
        if not path.exists():
            add_error(errors, f"missing {key} file: {path}")
            continue
        if key in {"command", "dry_run", "submit"} and not _file_mode_executable(path):
            add_warning(warnings, f"{key} file is not executable: {path}")

    command_path = Path(str(required_files["command"] or ""))
    if command_path.exists():
        tokens = _extract_command_tokens(command_path)
        if not tokens:
            add_error(errors, f"command file does not invoke generate_v3_checkpoint_proposal.py: {command_path}")
        else:
            token_text = " ".join(tokens)
            if "--task" not in tokens or plan.get("task") not in tokens:
                add_error(errors, "command file does not pass the run_plan task")
            if "--subtask" not in tokens or plan.get("subtask") not in tokens:
                add_error(errors, "command file does not pass the run_plan subtask")
            if "--model-dir" not in tokens or str(plan.get("model_dir")) not in tokens:
                add_error(errors, "command file does not pass the run_plan model_dir")
            if "--output-dir" not in tokens or str(plan.get("output_dir")) not in tokens:
                add_error(errors, "command file does not pass the run_plan output_dir")
            if "generate_v3_checkpoint_proposal.py" not in token_text:
                add_error(errors, "command file has no proposal generation script token")
        text = command_path.read_text()
        if "test -s" not in text or "proposal.txt" not in text:
            add_error(errors, "command file does not verify non-empty proposal.txt")

    dry_run_path = Path(str(required_files["dry_run"] or ""))
    if dry_run_path.exists():
        dry_text = dry_run_path.read_text()
        for required in ("--dry-run", "--priority", " 6", "--config", "--command-file"):
            if required not in dry_text:
                add_error(errors, f"dry-run script missing {required!r}")

    submit_path = Path(str(required_files["submit"] or ""))
    if submit_path.exists():
        submit_text = submit_path.read_text()
        guard_pos = submit_text.find(SUBMIT_GUARD_ENV)
        create_pos = submit_text.find("pai_manage.py create-job")
        if guard_pos < 0:
            add_error(errors, f"submit script missing guard env {SUBMIT_GUARD_ENV}")
        if create_pos < 0:
            add_error(errors, "submit script does not call pai_manage.py create-job")
        if guard_pos >= 0 and create_pos >= 0 and guard_pos > create_pos:
            add_error(errors, "submit guard appears after create-job command")
        if f'"${{{SUBMIT_GUARD_ENV}}}" != "1"' not in submit_text:
            add_error(errors, "submit script does not require guard env to equal 1")

    job_config_path = Path(str(required_files["pai_job_config"] or ""))
    if job_config_path.exists():
        job_cfg = yaml.safe_load(job_config_path.read_text()) or {}
        data_sources = job_cfg.get("data_sources") or []
        if not any(ds.get("mount") == "/newcpfs" and ds.get("id") for ds in data_sources if isinstance(ds, dict)):
            add_error(errors, "pai_job_config.yaml must mount /newcpfs with a datasource id")

    output_dir = Path(str(plan.get("output_dir") or ""))
    try:
        output_dir.relative_to(run_dir)
    except ValueError:
        add_warning(warnings, f"output_dir is outside run_dir: {output_dir}")


def validate_run_plan(run_dir: Path, require_priority: int, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    plan_path = run_dir / "run_plan.json"
    if not plan_path.exists():
        add_error(errors, f"missing run_plan.json: {plan_path}")
        return {}
    plan = read_json(plan_path)
    for key in ("task", "subtask", "model_dir", "output_dir", "resources", "files"):
        if key not in plan:
            add_error(errors, f"run_plan missing {key}")
    if plan.get("ready_to_dry_run") is not True:
        add_error(errors, "run_plan ready_to_dry_run must be true")
    if plan.get("ready_to_submit") is True:
        add_error(errors, "run_plan ready_to_submit should stay false until external quota confirmation")
    if plan.get("errors"):
        add_error(errors, f"run_plan has embedded errors: {plan.get('errors')}")
    if plan.get("submit_guard_env") != f"{SUBMIT_GUARD_ENV}=1":
        add_error(errors, f"run_plan submit_guard_env must be {SUBMIT_GUARD_ENV}=1")

    resources = plan.get("resources") or {}
    if int(resources.get("priority", -1)) != require_priority:
        add_error(errors, f"priority must be {require_priority}, got {resources.get('priority')}")
    gpus = int(resources.get("gpus") or 0)
    if gpus < 1:
        add_error(errors, f"gpus must be positive, got {gpus}")
    if gpus > 8:
        add_warning(warnings, f"proposal smoke requests {gpus} GPUs")

    model_dir = Path(str(plan.get("model_dir") or ""))
    validate_final_model(model_dir, errors)
    validate_files(run_dir, plan, errors, warnings)
    return plan


def scan_for_secrets(run_dir: Path) -> list[str]:
    findings = []
    skip_suffixes = {".parquet", ".arrow", ".pt", ".bin", ".safetensors", ".png", ".jpg", ".jpeg"}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() in skip_suffixes:
            continue
        if path.name == ".env":
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        rel = path.relative_to(run_dir)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"possible secret in {rel}")
                break
    return findings


def quota_confirmation(args: argparse.Namespace, plan: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if not args.confirmed_quota:
        return None, ["no confirmed quota snapshot provided; launch_ready=false"]
    if "/" not in args.confirmed_quota:
        return None, ["--confirmed-quota must be WORKSPACE/QUOTA"]
    workspace, quota = args.confirmed_quota.split("/", 1)
    resources = plan.get("resources") or {}
    errors = []
    if str(resources.get("workspace_id")) != workspace:
        errors.append(f"confirmed workspace {workspace} does not match plan {resources.get('workspace_id')}")
    if str(resources.get("resource_id")) != quota:
        errors.append(f"confirmed quota {quota} does not match plan {resources.get('resource_id')}")
    if args.confirmed_waiting != 0:
        errors.append(f"confirmed quota has waiting={args.confirmed_waiting}")
    required_gpus = int(resources.get("gpus") or 0)
    if args.confirmed_free_gpus < required_gpus:
        errors.append(f"confirmed free_gpus={args.confirmed_free_gpus} < required {required_gpus}")
    return {
        "workspace_id": workspace,
        "resource_id": quota,
        "confirmed_waiting": args.confirmed_waiting,
        "confirmed_free_gpus": args.confirmed_free_gpus,
        "required_gpus": required_gpus,
        "source": "user_confirmed_snapshot",
    }, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--require-priority", type=int, default=6)
    parser.add_argument("--confirmed-quota", help="Fresh user-confirmed quota as WORKSPACE/QUOTA.")
    parser.add_argument("--confirmed-free-gpus", type=int, default=0)
    parser.add_argument("--confirmed-waiting", type=int, default=0)
    parser.add_argument("--require-launch-ready", action="store_true")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    plan = validate_run_plan(run_dir, args.require_priority, errors, warnings)
    secret_findings = scan_for_secrets(run_dir)
    errors.extend(secret_findings)
    quota, quota_errors = quota_confirmation(args, plan) if plan else (None, ["missing run plan"])

    artifact_ready = not errors
    launch_ready = artifact_ready and not quota_errors and quota is not None
    report = {
        "run_dir": str(run_dir),
        "artifact_ready": artifact_ready,
        "launch_ready": launch_ready,
        "errors": errors,
        "warnings": warnings,
        "quota_errors": quota_errors,
        "quota": quota,
        "run_plan": {
            "job_name": plan.get("job_name") if plan else None,
            "task": plan.get("task") if plan else None,
            "subtask": plan.get("subtask") if plan else None,
            "model_dir": plan.get("model_dir") if plan else None,
            "output_dir": plan.get("output_dir") if plan else None,
            "resources": plan.get("resources") if plan else None,
            "ready_to_dry_run": plan.get("ready_to_dry_run") if plan else None,
            "ready_to_submit": plan.get("ready_to_submit") if plan else None,
            "submit_guard_env": plan.get("submit_guard_env") if plan else None,
        },
    }
    output_path = Path(args.write_json) if args.write_json else run_dir / "proposal_smoke_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_launch_ready:
        return 0 if launch_ready else 2
    return 0 if artifact_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
