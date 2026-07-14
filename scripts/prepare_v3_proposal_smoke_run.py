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


DEFAULT_MODEL_DIR = (
    ROOT
    / "runs"
    / "training"
    / "v3_sft_qwen25_32b"
    / "v3_sft_200_smoke"
    / "checkpoints"
    / "phase_000_2025-04"
    / "final"
)


def _quote_cmd(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _write_pai_job_config(run_dir: Path, data_sources: list[dict[str, Any]]) -> Path:
    path = run_dir / "pai_job_config.yaml"
    path.write_text(yaml.safe_dump({"data_sources": data_sources}, sort_keys=False))
    return path


def _validate_final_model(model_dir: Path) -> list[str]:
    errors = []
    required = ["config.json", "model.safetensors.index.json", "tokenizer.json"]
    for name in required:
        if not (model_dir / name).exists():
            errors.append(f"missing final model file: {model_dir / name}")
    if not list(model_dir.glob("model-*.safetensors")):
        errors.append(f"missing model safetensor shards in {model_dir}")
    return errors


def prepare_run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    dlc = dict((cfg.get("v3_training") or {}).get("dlc") or {})
    data_sources = list(dlc.get("data_sources") or [])
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir.resolve()
    output_dir = run_dir / "output"
    command = [
        args.python_bin,
        str(ROOT / "scripts" / "generate_v3_checkpoint_proposal.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--model-dir",
        str(model_dir),
        "--task",
        args.task,
        "--subtask",
        args.subtask,
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    if args.attn_implementation:
        command.extend(["--attn-implementation", args.attn_implementation])
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    command_file = run_dir / "dlc_command_skeleton.sh"
    command_file.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {ROOT}",
                "mkdir -p " + shlex.quote(str(output_dir)),
                _quote_cmd(command) + f" 2>&1 | tee {shlex.quote(str(run_dir / 'generate.log'))}",
                f"test -s {shlex.quote(str(output_dir / 'proposal.txt'))}",
                "",
            ]
        )
    )
    command_file.chmod(0o755)

    pai_config = _write_pai_job_config(run_dir, data_sources)
    job_name = args.job_name or f"v3_proposal_smoke_{int(time.time())}"
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
        str(dlc.get("workspace_id")),
        "--resource-id",
        str(dlc.get("resource_id")),
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
        f"CUDA_VISIBLE_DEVICES=0,{','.join(str(i) for i in range(1, args.gpus))}" if args.gpus > 1 else "CUDA_VISIBLE_DEVICES=0",
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
                "# Run only after a fresh quota snapshot confirms no queue and enough free GPUs.",
                ': "${CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_SMOKE:?Set this to 1 only after a fresh quota snapshot confirms no queue and enough free GPUs.}"',
                'if [[ "${CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_SMOKE}" != "1" ]]; then',
                '  echo "CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_SMOKE must equal 1" >&2',
                "  exit 2",
                "fi",
                "export ALIBABA_CLOUD_CREDENTIALS_URI=${ALIBABA_CLOUD_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}",
                _quote_cmd(create_args),
                "",
            ]
        )
    )
    submit_file.chmod(0o755)

    errors = _validate_final_model(model_dir)
    if not any(ds.get("mount") and ds.get("id") for ds in data_sources if isinstance(ds, dict)):
        errors.append("DLC data_sources must include a mounted datasource with an id")
    for key in ("endpoint", "workspace_id", "resource_id", "image"):
        if not dlc.get(key):
            errors.append(f"missing v3_training.dlc.{key}")
    if args.priority != 6:
        errors.append(f"priority must be 6 for this workflow, got {args.priority}")

    summary = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ready_to_dry_run": not errors,
        "ready_to_submit": False,
        "submit_blocker": "requires fresh user/console quota snapshot before running pai_create_job.sh",
        "submit_guard_env": "CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_SMOKE=1",
        "errors": errors,
        "job_name": job_name,
        "task": args.task,
        "subtask": args.subtask,
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "resources": {
            "workspace_id": dlc.get("workspace_id"),
            "resource_id": dlc.get("resource_id"),
            "gpus": args.gpus,
            "cpus": args.cpus,
            "memory": args.memory,
            "shared_memory": args.shared_memory,
            "priority": args.priority,
            "max_running_minutes": args.max_running_minutes,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a safe DLC smoke run for V3 checkpoint proposal generation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--task", default="dl_activation_function")
    parser.add_argument("--subtask", default="resnet20-cifar10")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "v3_checkpoint_proposal_smoke" / "dlc_activation")
    parser.add_argument("--job-name", default="")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--memory", default="300Gi")
    parser.add_argument("--shared-memory", default="64Gi")
    parser.add_argument("--max-running-minutes", type=int, default=60)
    parser.add_argument("--priority", type=int, default=6)
    parser.add_argument("--enable-rdma", action="store_true")
    args = parser.parse_args()
    summary = prepare_run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
