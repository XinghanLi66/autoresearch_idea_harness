#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.generators import generate_batches
from autoresearch_idea_harness.io import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Generate private and expert-facing proposal batches.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--packets", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--generators", default=None, help="Comma-separated generator IDs")
    args = p.parse_args()

    cfg = load_config(args.config)
    names = [x.strip() for x in args.generators.split(",") if x.strip()] if args.generators else None
    summary = generate_batches(
        cfg,
        packets_path=Path(args.packets) if args.packets else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limit=args.limit,
        generator_names=names,
    )
    print(summary)


if __name__ == "__main__":
    main()
