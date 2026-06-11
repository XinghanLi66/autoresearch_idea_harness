#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    / "v3_sft_strict_batch1000_auto"
    / "checkpoints"
    / "phase_000_2025-04"
    / "final"
)


def _quote_cmd(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _validate_final_model(model_dir: Path) -> list[str]:
    errors = []
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        if not (model_dir / name).exists():
            errors.append(f"missing final model file: {model_dir / name}")
    if not list(model_dir.glob("model-*.safetensors")):
        errors.append(f"missing model safetensor shards in {model_dir}")
    return errors


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    dlc = dict((cfg.get("v3_training") or {}).get("dlc") or {})
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.resolve()
    batch_output = run_dir / "output"

    command = [
        args.python_bin,
        str(ROOT / "scripts" / "generate_v3_checkpoint_proposal_batch.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(batch_output),
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
    if args.tasks:
        command.append("--tasks")
        command.extend(args.tasks)
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    command_file = run_dir / "dlc_command_skeleton.sh"
    command_file.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "cd /newcpfs/lxh/agentic-training/autoresearch_idea_harness",
                "mkdir -p " + shlex.quote(str(batch_output)),
                _quote_cmd(command) + f" 2>&1 | tee {shlex.quote(str(run_dir / 'batch_generate.log'))}",
                f"test -s {shlex.quote(str(batch_output / 'summary.json'))}",
                "",
            ]
        )
    )
    command_file.chmod(0o755)

    pai_config = run_dir / "pai_job_config.yaml"
    pai_config.write_text(yaml.safe_dump({"data_sources": dlc.get("data_sources") or []}, sort_keys=False))

    job_name = args.job_name or f"v3_proposal_batch_{int(time.time())}"
    create_args = [
        "python",
        "/root/.claude/skills/pai/scripts/pai_manage.py",
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
        "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(args.gpus)),
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
                ': "${CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_BATCH:?Set this to 1 only after a fresh quota snapshot confirms no queue and enough free GPUs.}"',
                'if [[ "${CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_BATCH}" != "1" ]]; then',
                '  echo "CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_BATCH must equal 1" >&2',
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
    if not any(ds.get("mount") == "/newcpfs" and ds.get("id") for ds in dlc.get("data_sources") or []):
        errors.append("DLC data_sources must include /newcpfs with an id")
    for key in ("endpoint", "workspace_id", "resource_id", "image"):
        if not dlc.get(key):
            errors.append(f"missing v3_training.dlc.{key}")
    if args.priority != 6:
        errors.append(f"priority must be 6 for this workflow, got {args.priority}")

    summary = {
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ready_to_dry_run": not errors,
        "ready_to_submit": False,
        "submit_guard_env": "CONFIRM_FRESH_QUOTA_FOR_V3_PROPOSAL_BATCH=1",
        "errors": errors,
        "job_name": job_name,
        "model_dir": str(model_dir),
        "output_dir": str(batch_output),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a guarded DLC run for V3 checkpoint proposal batch generation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "v3_checkpoint_proposal_batch" / "dlc_mls10_strict1000")
    parser.add_argument("--job-name", default="v3_proposal_batch_mls10_strict1000")
    parser.add_argument("--python-bin", default="/newcpfs/lxh/miniconda3/envs/loongflow_ml/bin/python")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--memory", default="300Gi")
    parser.add_argument("--shared-memory", default="64Gi")
    parser.add_argument("--max-running-minutes", type=int, default=120)
    parser.add_argument("--priority", type=int, default=6)
    parser.add_argument("--enable-rdma", action="store_true")
    args = parser.parse_args()
    summary = prepare(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
