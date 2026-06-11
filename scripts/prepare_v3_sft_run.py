#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.model_registry import resolve_base_model
from autoresearch_idea_harness.training_runs import prepare_v3_sft_run


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare a chronological V3 SFT run directory.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument(
        "--sft-dir",
        default=str(ROOT / "runs" / "training_data" / "v3_0_sft"),
        help="Directory containing collated train/val/test JSONL.",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "training" / "v3_sft"),
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--base-model-id", default=None, help="Registered base model id in config base_models.registry.")
    p.add_argument("--base-model-path", default=None)
    p.add_argument("--base-model-release-date", default=None)
    p.add_argument("--phase-by", default="month", choices=["month", "fixed", "none"])
    p.add_argument("--max-samples-per-phase", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--finetune-mode", default=None, choices=["lora", "full"])
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--num-train-epochs", type=int, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--dlc-workspace-id", default=None)
    p.add_argument("--dlc-resource-id", default=None)
    p.add_argument("--dlc-gpus", type=int, default=None)
    p.add_argument("--dlc-priority", type=int, default=None)
    p.add_argument("--allow-missing-model", action="store_true")
    p.add_argument("--allow-non-32b", action="store_true", help="Only for smoke/debug runs.")
    p.add_argument("--allow-smoke-model", action="store_true", help="Allow registry entries marked smoke_only.")
    args = p.parse_args()

    overrides = {}
    for arg_name, cfg_name in [
        ("finetune_mode", "finetune_mode"),
        ("learning_rate", "learning_rate"),
        ("num_train_epochs", "num_train_epochs"),
        ("per_device_train_batch_size", "per_device_train_batch_size"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
        ("max_seq_length", "max_seq_length"),
        ("lora_r", "lora_r"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            overrides[cfg_name] = value

    cfg = load_config(args.config)
    dlc_overrides = {
        "workspace_id": args.dlc_workspace_id,
        "resource_id": args.dlc_resource_id,
        "gpus": args.dlc_gpus,
        "priority": args.dlc_priority,
    }
    base_model = resolve_base_model(
        cfg,
        model_id=args.base_model_id,
        model_path=args.base_model_path,
        release_date=args.base_model_release_date,
        require_32b=not args.allow_non_32b,
        allow_missing_model=args.allow_missing_model,
        allow_smoke_model=args.allow_smoke_model,
    )
    run_id = args.run_id or f"v3_sft_{base_model['release_date'].replace('-', '')}"
    summary = prepare_v3_sft_run(
        cfg,
        sft_dir=Path(args.sft_dir),
        output_dir=Path(args.output_dir),
        base_model_path=base_model["path"],
        base_model_release_date=base_model["release_date"],
        base_model_metadata=base_model,
        run_id=run_id,
        phase_by=args.phase_by,
        max_samples_per_phase=args.max_samples_per_phase,
        limit=args.limit,
        train_overrides=overrides,
        dlc_overrides=dlc_overrides,
    )
    print(summary)


if __name__ == "__main__":
    main()
