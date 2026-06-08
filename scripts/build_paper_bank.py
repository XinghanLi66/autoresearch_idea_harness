#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.paper_bank import build_paper_bank


def main() -> None:
    p = argparse.ArgumentParser(description="Build V2 paper bank and reference evidence cache.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    summary = build_paper_bank(
        cfg,
        limit=args.limit,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(summary)


if __name__ == "__main__":
    main()
