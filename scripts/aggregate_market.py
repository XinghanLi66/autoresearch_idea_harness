#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.market import aggregate_forecasts, make_fixture_forecasts


def _brief(summary: dict, output_dir: Path) -> dict:
    return {
        "forecast_count": summary.get("forecast_count"),
        "proposal_count": summary.get("proposal_count"),
        "packet_count": summary.get("packet_count"),
        "by_generator": summary.get("by_generator"),
        "summary_path": str(output_dir / "summary.json"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate expert prediction-market forecasts.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--private-batches", default=None)
    p.add_argument("--forecasts", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--make-fixture", action="store_true")
    p.add_argument("--experts", type=int, default=3)
    args = p.parse_args()

    cfg = load_config(args.config)
    runs = Path(cfg["runs_dir"])
    private_path = Path(args.private_batches) if args.private_batches else runs / "proposal_batches" / "private.jsonl"
    forecasts_path = Path(args.forecasts) if args.forecasts else runs / "expert_forecasts" / "fixture.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else runs / "eval_summary"

    if args.make_fixture:
        n = make_fixture_forecasts(private_path, forecasts_path, experts=args.experts)
        print({"fixture_forecasts": n, "path": str(forecasts_path)})
    summary = aggregate_forecasts(private_path, forecasts_path, output_dir)
    print(_brief(summary, output_dir))


if __name__ == "__main__":
    main()
