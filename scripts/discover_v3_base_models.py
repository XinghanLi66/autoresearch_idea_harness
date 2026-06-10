#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.model_discovery import discover_local_base_models, write_discovery_report


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only local discovery for V3 30B-40B base-model candidates.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--root", action="append", default=None, help="Additional/override search root. Can be repeated.")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--include-all", action="store_true", help="Store every detected config instead of a preview.")
    p.add_argument("--output", default=str(ROOT / "runs" / "reports" / "v3_base_model_candidates.json"))
    p.add_argument("--markdown-output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    configured_roots = (((cfg.get("base_models") or {}).get("search_roots")) or [])
    roots = args.root or configured_roots
    if not roots:
        raise SystemExit("No roots configured. Add base_models.search_roots or pass --root.")
    output = Path(args.output)
    markdown = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
    report = discover_local_base_models(roots, max_depth=args.max_depth, include_all=args.include_all)
    write_discovery_report(report, output_json=output, output_md=markdown)
    print({
        "output": str(output),
        "markdown_output": str(markdown),
        "counts": report.get("counts"),
        "missing_roots": report.get("missing_roots"),
    })


if __name__ == "__main__":
    main()
