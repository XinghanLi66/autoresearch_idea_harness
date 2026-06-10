#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.training_runs import write_messages_parquet


def main() -> None:
    p = argparse.ArgumentParser(description="Convert V3 message-format SFT JSONL to Parquet.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    summary = write_messages_parquet(Path(args.input), Path(args.output), limit=args.limit)
    print(summary)


if __name__ == "__main__":
    main()
