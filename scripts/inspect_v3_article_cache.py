#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.article_cache_inspector import (  # noqa: E402
    inspect_article_cache,
    render_article_cache_markdown,
)
from autoresearch_idea_harness.io import load_config, write_json  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect all V3/V1 prompt caches and TeX snippets for one arXiv paper.")
    p.add_argument("arxiv_id")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--no-prompts", action="store_true", help="Show prompt metadata without full prompt text.")
    p.add_argument("--output")
    args = p.parse_args()

    cfg = load_config(args.config)
    report = inspect_article_cache(cfg, args.arxiv_id)
    if args.format == "json":
        text = json.dumps(report, indent=2, ensure_ascii=False)
    else:
        text = render_article_cache_markdown(report, include_prompts=not args.no_prompts)

    if args.output:
        out = Path(args.output)
        if args.format == "json":
            write_json(out, report)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
