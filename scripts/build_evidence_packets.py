#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.evidence_packets import build_evidence_packets
from autoresearch_idea_harness.io import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Build V2 evidence packets.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--paper-bank-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--allow-metadata-only-refs",
        action="store_true",
        help="Keep packets even when selected references have no TeX snippets.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    limit = args.limit
    if limit is None:
        limit = int(cfg.get("evidence_packets", {}).get("default_limit", 100))
    summary = build_evidence_packets(
        cfg,
        limit=limit,
        paper_bank_dir=Path(args.paper_bank_dir) if args.paper_bank_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        allow_metadata_only_refs=args.allow_metadata_only_refs,
    )
    print(summary)


if __name__ == "__main__":
    main()
