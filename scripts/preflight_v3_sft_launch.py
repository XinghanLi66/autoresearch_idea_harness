#!/usr/bin/env python3
# Internal Alibaba PAI-DLC helper — not needed for external reproduction (see REPRODUCE.md).
"""Fail-closed preflight for launching a V3 SFT DLC job.

This script does not submit jobs. It verifies that the prepared chronological
SFT run is internally consistent and, unless explicitly skipped, that a
quota-selection result is authoritative and matches the run plan.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CHECK_QUOTA = ROOT / "scripts" / "check_dlc_quota.py"
DEFAULT_RUN_DIR = ROOT / "runs" / "training" / "v3_sft_qwen25_32b" / "v3_sft_200_smoke"
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|credential|authorization)\s*[:=]\s*[A-Za-z0-9_./+=-]{16,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not an object")
            rows.append(value)
    return rows


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def add_warning(warnings: list[str], message: str) -> None:
    warnings.append(message)


def _valid_date(value: str, *, field: str) -> date:
    try:
        return date.fromisoformat(value)
    except Exception as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}") from exc


def validate_messages(row: dict[str, Any], *, source: str) -> list[str]:
    errors = []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return [f"{source}: messages must have at least system/user/assistant"]
    roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
    if roles[:3] != ["system", "user", "assistant"]:
        errors.append(f"{source}: first message roles are {roles[:3]}, expected system/user/assistant")
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"{source}: message {idx} is not an object")
            continue
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"{source}: message {idx} has empty content")
    return errors


def validate_run_plan(run_dir: Path, min_train_rows: int, require_priority: int) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    plan_path = run_dir / "run_plan.json"
    if not plan_path.exists():
        return {}, [f"missing run_plan.json: {plan_path}"], warnings
    plan = read_json(plan_path)

    if plan.get("train_row_count", 0) < min_train_rows:
        add_error(errors, f"train_row_count={plan.get('train_row_count')} < required {min_train_rows}")
    if plan.get("phase_count", 0) < 1:
        add_error(errors, "phase_count must be >= 1")

    release = str(plan.get("base_model_release_date") or "")
    try:
        release_date = _valid_date(release, field="base_model_release_date")
    except ValueError as exc:
        add_error(errors, str(exc))
        release_date = None

    metadata = plan.get("base_model_metadata") or {}
    size_b = metadata.get("size_b")
    if not isinstance(size_b, (int, float)) or not (30 <= float(size_b) <= 40):
        add_error(errors, f"base model size_b must be 30-40B, got {size_b!r}")
    if metadata.get("smoke_only"):
        add_error(errors, "base model metadata is marked smoke_only")
    model_path = Path(str(plan.get("base_model_path") or ""))
    if not (model_path / "config.json").exists():
        add_error(errors, f"base model config.json missing: {model_path / 'config.json'}")

    phases = plan.get("phases") or []
    created_values: list[str] = []
    row_total = 0
    for phase in phases:
        phase_name = phase.get("phase_name", "?")
        dataset_jsonl = Path(str(phase.get("dataset_jsonl") or ""))
        dataset_parquet = Path(str(phase.get("dataset_parquet") or ""))
        if not dataset_jsonl.exists():
            add_error(errors, f"{phase_name}: dataset_jsonl missing: {dataset_jsonl}")
            continue
        rows = read_jsonl(dataset_jsonl)
        row_total += len(rows)
        if phase.get("row_count") != len(rows):
            add_error(errors, f"{phase_name}: row_count={phase.get('row_count')} but JSONL has {len(rows)} rows")
        for idx, row in enumerate(rows):
            source = f"{dataset_jsonl}:{idx + 1}"
            for message_error in validate_messages(row, source=source):
                add_error(errors, message_error)
            if row.get("condition_strategy") != "with_research_question":
                add_error(errors, f"{source}: condition_strategy is not with_research_question")
            quality = ((row.get("target_cache") or {}).get("target_quality") or {})
            if quality.get("has_all_required_tags") is not True:
                add_error(errors, f"{source}: target cache lacks all required tags")
            leakage = row.get("leakage_guard") or {}
            if leakage.get("eligible_after_base_release") is not True:
                add_error(errors, f"{source}: leakage guard says sample is not eligible after base release")
            created = str(row.get("created") or "")
            try:
                created_date = _valid_date(created, field=f"{source}.created")
            except ValueError as exc:
                add_error(errors, str(exc))
                continue
            if release_date and created_date < release_date:
                add_error(errors, f"{source}: created {created} predates base release {release}")
            created_values.append(created)
        if not dataset_parquet.exists():
            add_error(errors, f"{phase_name}: dataset_parquet missing: {dataset_parquet}")
        else:
            try:
                import pandas as pd

                frame = pd.read_parquet(dataset_parquet)
                if len(frame) != len(rows):
                    add_error(errors, f"{phase_name}: parquet rows={len(frame)} but JSONL rows={len(rows)}")
                if "messages" not in frame.columns:
                    add_error(errors, f"{phase_name}: parquet missing messages column")
            except Exception as exc:  # noqa: BLE001 - surface as preflight failure
                add_error(errors, f"{phase_name}: failed to read parquet {dataset_parquet}: {exc}")

        config_path = Path(str(phase.get("config_path") or ""))
        if not config_path.exists():
            add_error(errors, f"{phase_name}: sft config missing: {config_path}")
        else:
            phase_cfg = yaml.safe_load(config_path.read_text()) or {}
            sft_cfg = phase_cfg.get("sft") or {}
            if int(sft_cfg.get("max_seq_length", 0)) < 8192:
                add_warning(warnings, f"{phase_name}: max_seq_length={sft_cfg.get('max_seq_length')} < 8192")
            if phase_cfg.get("model_name_or_path") != phase.get("input_model"):
                add_error(errors, f"{phase_name}: config model_name_or_path does not match run_plan input_model")

        command = str(phase.get("command") or "")
        if "train/sft.py" not in command:
            add_error(errors, f"{phase_name}: command does not invoke train/sft.py")
        torchrun_bin = _extract_torchrun_bin(command)
        if not torchrun_bin:
            add_error(errors, f"{phase_name}: command does not contain torchrun")
        elif torchrun_bin.startswith("/"):
            torchrun_path = Path(torchrun_bin)
            if not torchrun_path.exists():
                add_error(errors, f"{phase_name}: torchrun binary missing: {torchrun_path}")
            else:
                python_path = torchrun_path.parent / "python"
                if not python_path.exists():
                    add_warning(warnings, f"{phase_name}: cannot find python beside torchrun: {python_path}")
                else:
                    import_check = subprocess.run(
                        [
                            str(python_path),
                            "-c",
                            (
                                "import importlib.util, sys; "
                                "mods=['verl','torch','transformers','peft','pyarrow','pandas']; "
                                "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
                                "sys.exit(1 if missing else 0)"
                            ),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if import_check.returncode != 0:
                        add_error(
                            errors,
                            f"{phase_name}: training env cannot import required packages: {import_check.stderr[:500]}",
                        )

    if row_total != plan.get("train_row_count"):
        add_error(errors, f"phase row total {row_total} != train_row_count {plan.get('train_row_count')}")
    if created_values != sorted(created_values):
        add_error(errors, "training rows are not globally chronological by created date")

    dlc = plan.get("dlc") or {}
    if int(dlc.get("priority", -1)) != require_priority:
        add_error(errors, f"dlc priority must be {require_priority}, got {dlc.get('priority')}")
    if int(dlc.get("gpus", 0)) != 8:
        add_warning(warnings, f"dlc gpus={dlc.get('gpus')}, expected 8 for smoke 32B run")
    data_sources = dlc.get("data_sources") or []
    if not any(ds.get("mount") and ds.get("id") for ds in data_sources if isinstance(ds, dict)):
        add_error(errors, "dlc data_sources must include a mounted datasource with an id")
    command_file = run_dir / "dlc_command_skeleton.sh"
    if not command_file.exists():
        add_error(errors, f"missing command file: {command_file}")
    dry_run = run_dir / "pai_create_job_dry_run.sh"
    if not dry_run.exists():
        add_error(errors, f"missing dry-run create script: {dry_run}")
    else:
        dry_text = dry_run.read_text()
        for required in [
            "--priority",
            str(require_priority),
            "--config",
            str(run_dir / "pai_job_config.yaml"),
            "--command-file",
            str(command_file),
        ]:
            if required not in dry_text:
                add_error(errors, f"dry-run script missing {required!r}")
    job_config = run_dir / "pai_job_config.yaml"
    if not job_config.exists():
        add_error(errors, f"missing PAI config with data sources: {job_config}")
    else:
        job_cfg = yaml.safe_load(job_config.read_text()) or {}
        if not any(ds.get("mount") for ds in job_cfg.get("data_sources", []) if isinstance(ds, dict)):
            add_error(errors, "pai_job_config.yaml has no mounted datasource")

    return plan, errors, warnings


def _extract_torchrun_bin(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for token in tokens:
        if token == "torchrun" or token.endswith("/torchrun"):
            return token
    return None


def user_confirmed_quota(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.confirmed_quota:
        return None
    if "/" not in args.confirmed_quota:
        raise ValueError("--confirmed-quota must be WORKSPACE/QUOTA")
    workspace, quota = args.confirmed_quota.split("/", 1)
    waiting = int(args.confirmed_waiting)
    free_gpus = int(args.confirmed_free_gpus)
    best = {
        "workspace": workspace,
        "quota": quota,
        "label": "user_confirmed",
        "global_queue_source": "--confirmed-quota",
        "global_waiting": waiting,
        "free_gpus": free_gpus,
    }
    return {
        "queue_source": "user_confirmed_candidate",
        "can_prove_global_queue": False,
        "selection_enabled": waiting == 0 and free_gpus >= int(args.required_gpus),
        "limitations": [
            "This is a user-confirmed candidate quota, not an API-proven full global snapshot.",
            "Launch is allowed only because the chosen pool is explicitly confirmed to have no queue and enough free GPUs.",
        ],
        "best": best if waiting == 0 and free_gpus >= int(args.required_gpus) else None,
        "results": [best],
        "errors": [] if waiting == 0 and free_gpus >= int(args.required_gpus) else [
            f"confirmed quota not launchable: waiting={waiting}, free_gpus={free_gpus}, required_gpus={args.required_gpus}"
        ],
    }


def run_quota_checker(args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    confirmed = user_confirmed_quota(args)
    if confirmed is not None:
        return confirmed, [] if confirmed.get("best") else list(confirmed.get("errors") or [])
    if args.skip_quota:
        return None, ["quota check skipped by --skip-quota; launch_ready will be false"]
    cmd = [
        sys.executable,
        str(CHECK_QUOTA),
        "--json",
    ]
    for entry in args.manual_queue or []:
        cmd.extend(["--manual-queue", entry])
    if args.queue_snapshot:
        cmd.extend(["--queue-snapshot", args.queue_snapshot])
    if args.quota_mock_file:
        cmd.extend(["--mock-file", args.quota_mock_file])
    result = subprocess.run(cmd, capture_output=True, text=True)
    payload_text = result.stdout.strip()
    try:
        payload = json.loads(payload_text[payload_text.find("{"): payload_text.rfind("}") + 1])
    except Exception as exc:
        return None, [f"quota checker did not return parseable JSON: {exc}; stderr={result.stderr[:500]}"]
    errors = []
    if result.returncode != 0:
        errors.append(f"quota checker exited {result.returncode}")
    if not payload.get("can_prove_global_queue"):
        errors.append("quota checker cannot prove global queue; provide full manual snapshot")
    if not payload.get("best"):
        errors.append("quota checker returned no best pool")
    return payload, errors


def validate_quota_match(plan: dict[str, Any], quota: dict[str, Any] | None) -> list[str]:
    if quota is None:
        return ["quota was skipped"]
    best = quota.get("best") or {}
    dlc = plan.get("dlc") or {}
    errors = []
    if str(best.get("workspace")) != str(dlc.get("workspace_id")):
        errors.append(
            "quota best workspace does not match run_plan dlc.workspace_id: "
            f"best={best.get('workspace')} plan={dlc.get('workspace_id')}"
        )
    if str(best.get("quota")) != str(dlc.get("resource_id")):
        errors.append(
            "quota best resource does not match run_plan dlc.resource_id: "
            f"best={best.get('quota')} plan={dlc.get('resource_id')}"
        )
    required_gpus = int(dlc.get("gpus") or 0)
    if best.get("free_gpus") is not None and int(best["free_gpus"]) < required_gpus:
        errors.append(f"confirmed quota free_gpus={best['free_gpus']} < run_plan gpus={required_gpus}")
    if best.get("global_waiting") not in (None, 0):
        errors.append(f"confirmed quota has waiting={best.get('global_waiting')}")
    return errors


def scan_for_secrets(run_dir: Path) -> list[str]:
    findings = []
    skip_suffixes = {".parquet", ".arrow", ".pt", ".bin", ".safetensors", ".png", ".jpg", ".jpeg"}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() in skip_suffixes:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--require-priority", type=int, default=6)
    parser.add_argument("--manual-queue", action="append", default=[])
    parser.add_argument("--queue-snapshot")
    parser.add_argument("--quota-mock-file", help="Offline test fixture passed through to check_dlc_quota.py.")
    parser.add_argument("--confirmed-quota", help="Explicit user-confirmed launch quota as WORKSPACE/QUOTA.")
    parser.add_argument("--confirmed-free-gpus", type=int, default=0)
    parser.add_argument("--confirmed-waiting", type=int, default=0)
    parser.add_argument("--required-gpus", type=int, default=8)
    parser.add_argument("--skip-quota", action="store_true", help="Validate artifacts only; launch_ready will be false.")
    parser.add_argument("--write-json", default=None, help="Write preflight report JSON to this path.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    plan, errors, warnings = validate_run_plan(run_dir, args.min_train_rows, args.require_priority)
    quota, quota_errors = run_quota_checker(args)
    errors.extend(quota_errors)
    if plan:
        errors.extend(validate_quota_match(plan, quota))
    secret_findings = scan_for_secrets(run_dir)
    errors.extend(secret_findings)

    report = {
        "run_dir": str(run_dir),
        "launch_ready": not errors and quota is not None,
        "artifact_ready": not [e for e in errors if not e.startswith("quota ") and "quota checker" not in e],
        "errors": errors,
        "warnings": warnings,
        "run_plan": {
            "run_id": plan.get("run_id") if plan else None,
            "train_row_count": plan.get("train_row_count") if plan else None,
            "phase_count": plan.get("phase_count") if plan else None,
            "base_model_path": plan.get("base_model_path") if plan else None,
            "base_model_release_date": plan.get("base_model_release_date") if plan else None,
            "dlc": plan.get("dlc") if plan else None,
        },
        "quota": quota,
    }
    output_path = Path(args.write_json) if args.write_json else run_dir / "launch_preflight.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["launch_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
