#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.report import render_markdown_report


def main() -> None:
    p = argparse.ArgumentParser(description="Render a markdown preview report.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--output", default=None)
    p.add_argument("--packets", default=None)
    p.add_argument("--expert-batches", default=None)
    p.add_argument("--private-batches", default=None)
    p.add_argument("--summary", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    runs = Path(cfg["runs_dir"])
    output = Path(args.output) if args.output else runs / "reports" / "market_preview.md"
    render_markdown_report(
        packets_path=Path(args.packets) if args.packets else runs / "evidence_packets" / "v1" / "all.jsonl",
        expert_batches_path=Path(args.expert_batches) if args.expert_batches else runs / "proposal_batches" / "expert.jsonl",
        private_batches_path=Path(args.private_batches) if args.private_batches else runs / "proposal_batches" / "private.jsonl",
        summary_path=Path(args.summary) if args.summary else runs / "eval_summary" / "summary.json",
        output_path=output,
        limit=args.limit,
    )
    print({"report": str(output)})


if __name__ == "__main__":
    main()
