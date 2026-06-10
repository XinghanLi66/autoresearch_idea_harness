#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.training_manifest import collate_v3_sft_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Collate V3 manifest + target cache into SFT JSONL.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument(
        "--manifest-dir",
        default=str(ROOT / "runs" / "training_data" / "v3_0_manifest"),
    )
    p.add_argument(
        "--target-cache",
        action="append",
        required=True,
        help="JSONL containing target_impl_proposal or target_proposal. May be repeated.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--include-cot", action="store_true")
    p.add_argument("--min-quality-score", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    collate_cfg = cfg.get("sft_collate", {})
    summary = collate_v3_sft_dataset(
        manifest_dir=Path(args.manifest_dir),
        target_cache_paths=[Path(p) for p in args.target_cache],
        output_dir=Path(args.output_dir) if args.output_dir else None,
        include_cot=bool(args.include_cot or collate_cfg.get("include_cot", False)),
        min_quality_score=args.min_quality_score,
    )
    print(summary)


if __name__ == "__main__":
    main()
