#!/usr/bin/env python3
# Internal Alibaba PAI-DLC helper — not needed for external reproduction (see REPRODUCE.md).
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config, write_json

from build_v3_rl_parquet import build_v3_rl_parquet


DEFAULT_SFT_DIR = (
    ROOT
    / "runs"
    / "training_data"
    / "v3_0_sft_qwen25_32b_strict_batch1000"
)
DEFAULT_MODEL = (
    ROOT
    / "runs"
    / "training"
    / "v3_sft_qwen25_32b"
    / "v3_sft_strict_batch1000_auto"
    / "checkpoints"
    / "phase_000_2025-04"
    / "final"
)


def _require_model(path: Path) -> None:
    missing = []
    for rel in ["config.json", "model.safetensors.index.json", "tokenizer.json"]:
        if not (path / rel).exists():
            missing.append(rel)
    if not list(path.glob("model-*.safetensors")):
        missing.append("model-*.safetensors")
    if missing:
        raise FileNotFoundError(f"model path is incomplete: {path}; missing {missing}")


def _write_dlc_files(run_dir: Path, summary: dict[str, Any]) -> None:
    dlc = summary["dlc"]
    command = f"""# V3 PRS RL DLC command
set -euo pipefail
cd {ROOT}
{summary['launcher']}
"""
    command_path = run_dir / "dlc_command_skeleton.sh"
    command_path.write_text(command)

    config_path = run_dir / "pai_job_config.yaml"
    config_path.write_text(yaml.safe_dump({"data_sources": dlc.get("data_sources") or []}, sort_keys=False))

    base_args = [
        "python",
        os.environ.get("PAI_MANAGE", "/root/.claude/skills/pai/scripts/pai_manage.py"),
        "create-job",
        "--endpoint", str(dlc["endpoint"]),
        "--name", str(summary["run_id"]),
        "--command-file", str(command_path),
        "--workspace-id", str(dlc["workspace_id"]),
        "--resource-id", str(dlc["resource_id"]),
        "--image", str(dlc["image"]),
        "--gpus", str(dlc.get("gpus", 8)),
        "--cpus", str(dlc.get("cpus", 100)),
        "--memory", str(dlc.get("memory", "1000Gi")),
        "--shared-memory", str(dlc.get("shared_memory", "1000Gi")),
        "--max-running-minutes", str(dlc.get("max_running_minutes", 720)),
        "--priority", str(dlc.get("priority", 6)),
        "--env", f"NGPU={dlc.get('gpus', 8)}",
        "--enable-rdma", "true" if dlc.get("enable_rdma", True) else "false",
        "--config", str(config_path),
    ]
    for dry_run in (True, False):
        args = list(base_args)
        if dry_run:
            args.append("--dry-run")
        lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "export ALIBABA_CLOUD_CREDENTIALS_URI=${ALIBABA_CLOUD_CREDENTIALS_URI:-http://localhost:7002/api/v1/credentials/0}",
        ]
        if not dry_run:
            lines.extend([
                ': "${CONFIRM_FRESH_QUOTA_FOR_V3_RL:?Set this to 1 after a fresh quota check confirms enough free GPUs and acceptable queue.}"',
                'if [[ "${CONFIRM_FRESH_QUOTA_FOR_V3_RL}" != "1" ]]; then',
                '  echo "CONFIRM_FRESH_QUOTA_FOR_V3_RL must equal 1" >&2',
                "  exit 2",
                "fi",
            ])
        lines.append(" ".join(shlex.quote(a) for a in args))
        path = run_dir / ("pai_create_job_dry_run.sh" if dry_run else "pai_create_job.sh")
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    model_path = Path(args.model_path or DEFAULT_MODEL).resolve()
    _require_model(model_path)
    sft_dir = Path(args.sft_dir or DEFAULT_SFT_DIR).resolve()
    train_jsonl = sft_dir / "train.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)

    run_dir = Path(args.output_dir).resolve() / args.run_id
    dataset_dir = run_dir / "dataset"
    checkpoint_dir = run_dir / "checkpoints" / "prs"
    parquet = dataset_dir / f"train_prs_limit{args.limit}.parquet" if args.limit else dataset_dir / "train_prs.parquet"
    parquet_summary = build_v3_rl_parquet(
        input_jsonl=train_jsonl,
        output_parquet=parquet,
        tokenizer_path=model_path,
        reward_type="prs",
        limit=args.limit,
    )

    python_bin = args.python_bin
    rl_cmd = [
        python_bin,
        str(ROOT / "scripts" / "run_v3_rl.py"),
        "--run-id", args.run_id,
        "--train-file", str(parquet),
        "--model-path", str(model_path),
        "--output-dir", str(checkpoint_dir),
        "--reward-type", "prs",
        "--n-gpus", str(args.gpus),
        "--rollout-tensor-parallel-size", str(args.rollout_tensor_parallel_size),
        "--total-training-steps", str(args.total_training_steps),
        "--num-generations", str(args.num_generations),
        "--learning-rate", str(args.learning_rate),
        "--kl-coeff", str(args.kl_coeff),
        "--max-prompt-length", str(args.max_prompt_length),
        "--max-response-length", str(args.max_response_length),
        "--ppo-max-token-len", str(args.ppo_max_token_len),
        "--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
        "--save-steps", str(args.save_steps),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
    ]
    launcher = " ".join(shlex.quote(x) for x in rl_cmd) + f" 2>&1 | tee {shlex.quote(str(run_dir / 'rl.log'))}"
    dlc = dict((cfg.get("v3_training") or {}).get("dlc") or {})
    for key, value in {
        "workspace_id": args.dlc_workspace_id,
        "resource_id": args.dlc_resource_id,
        "gpus": args.gpus,
        "priority": args.priority,
        "max_running_minutes": args.max_running_minutes,
    }.items():
        if value is not None:
            dlc[key] = value
    summary = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "objective": "V3 PRS GRPO on strict synthesized proposal targets",
        "reward": {
            "type": "prs",
            "formula": "0.8 * cosine(generated_proposal, strict_target_proposal) + 0.2 * V3_XML_format",
            "ground_truth": "target proposal from strict TeX synthesis, not user prompt",
        },
        "sft_dir": str(sft_dir),
        "train_jsonl": str(train_jsonl),
        "parquet": parquet_summary,
        "init_model": str(model_path),
        "checkpoint_dir": str(checkpoint_dir),
        "expected_final": str(checkpoint_dir / "final"),
        "hyperparameters": {
            "total_training_steps": args.total_training_steps,
            "num_generations": args.num_generations,
            "learning_rate": args.learning_rate,
            "kl_coeff": args.kl_coeff,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "ppo_max_token_len": args.ppo_max_token_len,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "rollout_tensor_parallel_size": args.rollout_tensor_parallel_size,
        },
        "leakage_policy": parquet_summary["leakage_policy"],
        "launcher": launcher,
        "dlc": dlc,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_plan.json", summary)
    _write_dlc_files(run_dir, summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare a V3 PRS RL run directory and guarded DLC scripts.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--run-id", default="v3_rl_prs_strict1000_sftinit")
    p.add_argument("--sft-dir", default=str(DEFAULT_SFT_DIR))
    p.add_argument("--model-path", default=str(DEFAULT_MODEL))
    p.add_argument("--output-dir", default=str(ROOT / "runs" / "training" / "v3_rl_qwen25_32b"))
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--total-training-steps", type=int, default=80)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2.0e-6)
    p.add_argument("--kl-coeff", type=float, default=0.02)
    p.add_argument("--max-prompt-length", type=int, default=4096)
    p.add_argument("--max-response-length", type=int, default=1800)
    p.add_argument("--ppo-max-token-len", type=int, default=8192)
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    p.add_argument("--save-steps", type=int, default=40)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--rollout-tensor-parallel-size", type=int, default=8)
    p.add_argument("--priority", type=int, default=6)
    p.add_argument("--max-running-minutes", type=int, default=720)
    p.add_argument("--dlc-workspace-id", default=None)
    p.add_argument("--dlc-resource-id", default=None)
    args = p.parse_args()
    summary = prepare(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
