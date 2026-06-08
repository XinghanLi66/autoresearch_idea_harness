#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.market import make_llm_forecasts


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LLM expert prediction-market forecasts.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--expert-batches", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--experts", default=None, help="Comma-separated expert IDs from config.")
    p.add_argument("--limit-packets", type=int, default=None)
    p.add_argument("--skip-missing", action="store_true", help="Skip experts whose key env var is absent.")
    p.add_argument("--skip-errors", action="store_true", help="Record failed expert calls and continue.")
    args = p.parse_args()

    cfg = load_config(args.config)
    runs = Path(cfg["runs_dir"])
    expert_batches = Path(args.expert_batches) if args.expert_batches else runs / "proposal_batches" / "expert.jsonl"
    output = Path(args.output) if args.output else runs / "expert_forecasts" / "llm.jsonl"
    experts = [x.strip() for x in args.experts.split(",") if x.strip()] if args.experts else None

    summary = make_llm_forecasts(
        cfg,
        expert_batches_path=expert_batches,
        output_path=output,
        expert_ids=experts,
        limit_packets=args.limit_packets,
        skip_missing=args.skip_missing,
        skip_errors=args.skip_errors,
    )
    print(summary)


if __name__ == "__main__":
    main()
