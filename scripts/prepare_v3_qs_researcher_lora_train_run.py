#!/usr/bin/env python3
"""Prepare a QS controlled V3 researcher-CoT LoRA training run.

This is the production-facing wrapper around
`prepare_v3_qs_researcher_lora_smoke_run.py`. It keeps the proven QS command
generation path but writes plans and remote outputs under
`qs_researcher_lora_train` instead of the earlier smoke namespace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_v3_qs_researcher_lora_smoke_run import prepare  # noqa: E402


DEFAULT_BASE_MODEL = "/mnt/3fs/lxh/agentic-training/models/Qwen2.5-7B-Instruct"
DEFAULT_REMOTE_TRAIN = "/mnt/3fs/lxh/agentic-training/data/researcher_cot/prejudge_v1/train.jsonl"
DEFAULT_REMOTE_VAL = "/mnt/3fs/lxh/agentic-training/data/researcher_cot/prejudge_v1/val.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "qs_researcher_lora_train"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--remote-train-jsonl", default=DEFAULT_REMOTE_TRAIN)
    parser.add_argument("--remote-val-jsonl", default=DEFAULT_REMOTE_VAL)
    parser.add_argument("--limit-rows", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    summary = prepare(
        Path(args.config),
        Path(args.output_dir),
        args.run_id,
        args.base_model,
        None,
        args.remote_train_jsonl,
        args.remote_val_jsonl,
        args.limit_rows,
        args.max_steps,
        args.max_seq_length,
        remote_run_family="qs_researcher_lora_train",
        purpose=(
            "QS V3 controlled researcher-CoT LoRA training run: staged model, "
            "staged train/val JSONL, no merge, no generation check."
        ),
        log_prefix="qs-lora-train",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
