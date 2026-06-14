#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TEX_KINDS = [
    "abstract",
    "problem",
    "method",
    "implementation",
    "algorithm_or_system",
    "training_or_data_recipe",
    "evaluation",
    "results",
    "risks_and_limitations",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _backfill_article(path: Path, *, dry_run: bool) -> dict[str, Any]:
    article = _load_json(path)
    raw = article.setdefault("raw_tex_snippets", {})
    curated = article.get("tex_snippets") or {}
    added: dict[str, int] = {}

    for kind in TEX_KINDS:
        if raw.get(kind):
            continue
        snippets = curated.get(kind) or []
        if not snippets:
            continue
        backfilled = []
        for snippet in snippets:
            text = str(snippet.get("text") or "").strip()
            if not text:
                continue
            backfilled.append({
                "kind": kind,
                "heading": snippet.get("heading") or kind,
                "source": "backfilled_from_curated_tex_snippets",
                "original_source": snippet.get("source"),
                "chars_raw": len(text),
                "text": text,
                "quality": snippet.get("quality") or {},
            })
        if backfilled:
            raw[kind] = backfilled
            added[kind] = len(backfilled)

    if added and not dry_run:
        _write_json(path, article)
    return {
        "arxiv_id": article.get("arxiv_id") or path.parent.name,
        "article_cache": str(path),
        "added": added,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing raw_tex_snippets kinds from audited curated TeX snippets."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.glob("articles/*/article_cache.json")):
        row = _backfill_article(path, dry_run=args.dry_run)
        if row["added"]:
            rows.append(row)

    added_by_kind: dict[str, int] = {}
    for row in rows:
        for kind, count in row["added"].items():
            added_by_kind[kind] = added_by_kind.get(kind, 0) + count
    summary = {
        "root": str(args.root),
        "dry_run": args.dry_run,
        "articles_changed": len(rows),
        "added_by_kind": added_by_kind,
        "changed_articles": rows[:50],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
