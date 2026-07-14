from __future__ import annotations

import json
import os
import re
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .io import iter_jsonl, project_root, write_json, write_jsonl

# Internal Alibaba PAI-DLC helper — not needed for external reproduction
# (see REPRODUCE.md). External users run the generated run_sft_curriculum.sh
# directly with torchrun; the PAI submission skeleton below is only emitted
# when the internal DLC config section is filled in.
PAI_MANAGE = os.environ.get("PAI_MANAGE", "/root/.claude/skills/pai/scripts/pai_manage.py")


def read_sft_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            continue
        if not row.get("arxiv_id") or not row.get("created"):
            continue
        rows.append(row)
    rows.sort(key=lambda r: (r.get("created") or "", r.get("arxiv_id") or ""))
    return rows


def write_messages_parquet(input_jsonl: Path, output_parquet: Path, limit: int | None = None) -> dict[str, Any]:
    import pandas as pd

    rows = []
    for row in iter_jsonl(input_jsonl):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue
        rows.append({
            "messages": messages,
            "sample_id": row.get("sample_id"),
            "arxiv_id": row.get("arxiv_id"),
            "created": row.get("created"),
            "split": row.get("split"),
            "quality_score": (row.get("quality") or {}).get("score"),
        })
        if limit is not None and len(rows) >= limit:
            break
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_parquet, index=False)
    return {
        "input_jsonl": str(input_jsonl),
        "output_parquet": str(output_parquet),
        "row_count": len(rows),
    }


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "phase"


def _month_from_created(created: str | None) -> str:
    return (created or "unknown")[:7]


def _phase_rows(rows: list[dict[str, Any]], phase_by: str, max_samples_per_phase: int | None) -> list[tuple[str, list[dict[str, Any]]]]:
    if phase_by == "none":
        return [("all", rows)]
    if phase_by == "month":
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            buckets.setdefault(_month_from_created(row.get("created")), []).append(row)
        return [(month, buckets[month]) for month in sorted(buckets)]
    if phase_by == "fixed":
        size = max(1, int(max_samples_per_phase or 1000))
        phases = []
        for i in range(0, len(rows), size):
            phases.append((f"{i // size:04d}", rows[i:i + size]))
        return phases
    raise ValueError(f"unknown phase_by={phase_by!r}")


def _proposal_rl_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg.get("proposal_rl_root") or project_root().parent / "proposal_rl")


def _phase_config(
    cfg: dict[str, Any],
    phase_dir: Path,
    dataset_jsonl: Path,
    model_path: str,
    output_dir: Path,
    train_cfg: dict[str, Any],
) -> dict[str, Any]:
    proposal_root = _proposal_rl_root(cfg)
    return {
        "runs_dir": str(phase_dir),
        "arxiv_root": cfg.get("arxiv_root"),
        "model_name_or_path": model_path,
        "prompt_builder": {
            "strategy": "with_research_question",
            "max_refs": 5,
            "shuffle_seed": 42,
        },
        "sft": {
            "finetune_mode": train_cfg.get("finetune_mode", "lora"),
            "output_dir": str(output_dir),
            "dataset_file": str(dataset_jsonl),
            "num_train_epochs": train_cfg.get("num_train_epochs", 1),
            "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 16),
            "learning_rate": float(train_cfg.get("learning_rate", 1.0e-5)),
            "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.03)),
            "max_seq_length": int(train_cfg.get("max_seq_length", 8192)),
            "save_steps": int(train_cfg.get("save_steps", 200)),
            "lora_r": int(train_cfg.get("lora_r", 64)),
            "lora_alpha": int(train_cfg.get("lora_alpha", 128)),
        },
        "_v3": {
            "trainer": "proposal_rl.train.sft",
            "proposal_rl_root": str(proposal_root),
            "prebuilt_parquet": str(dataset_jsonl.with_suffix(".parquet")),
            "note": (
                "proposal_rl/train/sft.py will use the prebuilt parquet because it "
                "shares the dataset JSONL stem and is newer than the JSONL."
            ),
        },
    }


def prepare_v3_sft_run(
    cfg: dict[str, Any],
    sft_dir: Path,
    output_dir: Path,
    base_model_path: str,
    base_model_release_date: str,
    run_id: str,
    base_model_metadata: dict[str, Any] | None = None,
    phase_by: str = "month",
    max_samples_per_phase: int | None = None,
    limit: int | None = None,
    train_overrides: dict[str, Any] | None = None,
    dlc_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_cfg = dict(cfg.get("v3_training", {}).get("sft", {}))
    train_cfg.update(train_overrides or {})
    dlc_cfg = dict(cfg.get("v3_training", {}).get("dlc", {}))
    dlc_cfg.update({k: v for k, v in (dlc_overrides or {}).items() if v is not None})
    sft_dir = sft_dir.resolve()
    run_dir = (output_dir / run_id).resolve()
    train_file = sft_dir / "train.jsonl"
    val_file = sft_dir / "val.jsonl"
    test_file = sft_dir / "test.jsonl"
    train_rows = read_sft_rows(train_file)
    if limit is not None:
        train_rows = train_rows[:limit]
    phases = _phase_rows(train_rows, phase_by=phase_by, max_samples_per_phase=max_samples_per_phase)

    phase_specs: list[dict[str, Any]] = []
    current_model = base_model_path
    for idx, (label, rows) in enumerate(phases):
        phase_name = f"phase_{idx:03d}_{_safe_label(label)}"
        phase_dir = run_dir / "curriculum" / phase_name
        dataset_jsonl = phase_dir / "train.jsonl"
        output_model_dir = run_dir / "checkpoints" / phase_name
        write_jsonl(dataset_jsonl, rows)
        parquet_summary = write_messages_parquet(dataset_jsonl, dataset_jsonl.with_suffix(".parquet"))
        config = _phase_config(
            cfg,
            phase_dir=phase_dir,
            dataset_jsonl=dataset_jsonl,
            model_path=current_model,
            output_dir=output_model_dir,
            train_cfg=train_cfg,
        )
        config_path = phase_dir / "sft_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        torchrun_bin = train_cfg.get("torchrun_bin", "torchrun")
        cmd = (
            f"cd {_proposal_rl_root(cfg)} && "
            f"{torchrun_bin} --nproc_per_node=${{NGPU:-8}} train/sft.py --config {config_path}"
        )
        phase_specs.append({
            "phase_index": idx,
            "phase_name": phase_name,
            "label": label,
            "row_count": len(rows),
            "created_range": {
                "first": rows[0].get("created") if rows else None,
                "last": rows[-1].get("created") if rows else None,
            },
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_parquet": parquet_summary["output_parquet"],
            "config_path": str(config_path),
            "input_model": current_model,
            "output_dir": str(output_model_dir),
            "expected_final": str(output_model_dir / "final"),
            "command": cmd,
        })
        current_model = str(output_model_dir / "final")

    val_summary = None
    test_summary = None
    if val_file.exists():
        val_summary = write_messages_parquet(val_file, run_dir / "eval" / "val.parquet")
    if test_file.exists():
        test_summary = write_messages_parquet(test_file, run_dir / "eval" / "test.parquet")

    launcher = run_dir / "run_sft_curriculum.sh"
    launcher.write_text(_launcher_text(phase_specs, run_dir))
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "sft_dir": str(sft_dir),
        "base_model_path": base_model_path,
        "base_model_release_date": base_model_release_date,
        "base_model_metadata": base_model_metadata or {},
        "phase_by": phase_by,
        "phase_count": len(phase_specs),
        "train_row_count": sum(p["row_count"] for p in phase_specs),
        "val": val_summary,
        "test": test_summary,
        "train_created_range": {
            "first": train_rows[0].get("created") if train_rows else None,
            "last": train_rows[-1].get("created") if train_rows else None,
        },
        "by_month": dict(Counter(_month_from_created(r.get("created")) for r in train_rows)),
        "training_policy": {
            "curriculum": "sequential chronological phases; each phase starts from previous phase final checkpoint",
            "objective": "SFT on with_research_question -> proposal target",
            "rl_followup": (
                "After SFT passes smoke evaluation, run V2.3-style MLS proposal sweep and "
                "use expert/worker outcomes for preference or GRPO training."
            ),
            "ability_preservation": (
                "Default to low-LR LoRA for first smoke; move to larger/full 32B "
                "continued tuning only after regression checks on master advice behavior."
            ),
        },
        "dlc": dlc_cfg,
        "phases": phase_specs,
        "launcher": str(launcher),
    }
    write_json(run_dir / "run_plan.json", summary)
    _write_dlc_skeleton(run_dir, summary)
    return summary


def _launcher_text(phases: list[dict[str, Any]], run_dir: Path) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}",
        f"mkdir -p {run_dir}/logs",
        "",
    ]
    for phase in phases:
        log_path = run_dir / "logs" / f"{phase['phase_name']}.log"
        lines.extend([
            f"echo '[v3-sft] starting {phase['phase_name']} at '$(date)",
            f"{phase['command']} 2>&1 | tee {log_path}",
            f"test -f {phase['expected_final']}/config.json",
            "",
        ])
    lines.append("echo '[v3-sft] all phases complete at '$(date)")
    return "\n".join(lines) + "\n"


def _write_dlc_skeleton(run_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# V3 SFT DLC submission skeleton (INTERNAL: Alibaba PAI-DLC only)
#
# This is the command that should run inside the DLC worker pod.
# Submit it through `pai_create_job_dry_run.sh` after checking quota availability.
# External users: ignore this file and run the launcher directly (REPRODUCE.md).

set -euo pipefail
cd {project_root()}
bash {summary['launcher']}
"""
    path = run_dir / "dlc_command_skeleton.sh"
    path.write_text(text)
    dlc_config_path = _write_pai_job_config(run_dir, summary)
    dry_run = _pai_create_job_text(run_dir, summary, dry_run=True)
    if dry_run:
        dry_run_path = run_dir / "pai_create_job_dry_run.sh"
        dry_run_path.write_text(dry_run)
        dry_run_path.chmod(0o755)
    submit = _pai_create_job_text(run_dir, summary, dry_run=False)
    if submit:
        submit_path = run_dir / "pai_create_job.sh"
        submit_path.write_text(submit)
        submit_path.chmod(0o755)


def _write_pai_job_config(run_dir: Path, summary: dict[str, Any]) -> Path | None:
    dlc = summary.get("dlc") or {}
    data_sources = dlc.get("data_sources") or []
    if not data_sources:
        return None
    path = run_dir / "pai_job_config.yaml"
    path.write_text(yaml.safe_dump({"data_sources": data_sources}, sort_keys=False))
    return path


def _pai_create_job_text(run_dir: Path, summary: dict[str, Any], *, dry_run: bool) -> str:
    dlc = summary.get("dlc") or {}
    required = ["endpoint", "workspace_id", "resource_id", "image"]
    if any(not dlc.get(key) for key in required):
        return ""
    config_path = run_dir / "pai_job_config.yaml"
    args = [
        "python",
        PAI_MANAGE,
        "create-job",
        "--endpoint", str(dlc["endpoint"]),
        "--name", str(summary["run_id"]),
        "--command-file", str(run_dir / "dlc_command_skeleton.sh"),
        "--workspace-id", str(dlc["workspace_id"]),
        "--resource-id", str(dlc["resource_id"]),
        "--image", str(dlc["image"]),
        "--gpus", str(dlc.get("gpus", 8)),
        "--cpus", str(dlc.get("cpus", 100)),
        "--memory", str(dlc.get("memory", "1000Gi")),
        "--shared-memory", str(dlc.get("shared_memory", "1000Gi")),
        "--max-running-minutes", str(dlc.get("max_running_minutes", 720)),
        "--priority", str(dlc.get("priority", 7)),
        "--env", f"NGPU={dlc.get('gpus', 8)}",
        "--enable-rdma", "true" if dlc.get("enable_rdma", True) else "false",
    ]
    if config_path.exists():
        args.extend(["--config", str(config_path)])
    if dry_run:
        args.append("--dry-run")
    quoted = " ".join(shlex.quote(arg) for arg in args)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Uses the local DSW credential provider; do not write credentials to disk.",
        "export ALIBABA_CLOUD_CREDENTIALS_URI=${ALIBABA_CLOUD_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}",
    ]
    if not dry_run:
        lines.extend([
            "# Run only after a fresh quota snapshot confirms no queue and enough free GPUs.",
            ': "${CONFIRM_FRESH_QUOTA_FOR_V3_SFT:?Set this to 1 only after a fresh quota snapshot confirms no queue and enough free GPUs.}"',
            'if [[ "${CONFIRM_FRESH_QUOTA_FOR_V3_SFT}" != "1" ]]; then',
            '  echo "CONFIRM_FRESH_QUOTA_FOR_V3_SFT must equal 1" >&2',
            "  exit 2",
            "fi",
        ])
    lines.extend([quoted, ""])
    return "\n".join(lines)
