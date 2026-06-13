#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, write_json  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _link_article_dir(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise SystemExit(f"Missing article directory: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise SystemExit(f"Article output already exists with different target: {dst}")
    dst.symlink_to(src.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent V3 quality-cache shard outputs.")
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pattern", default="shard_*")
    parser.add_argument("--allow-missing-summary", action="store_true")
    args = parser.parse_args()

    shards = sorted(p for p in args.shards_root.glob(args.pattern) if p.is_dir())
    if not shards:
        raise SystemExit(f"No shard directories matched {args.shards_root}/{args.pattern}")
    output = args.output or (args.shards_root / "merged")
    output.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    compact_by_key: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()
    quality_errors: Counter[str] = Counter()
    article_dirs_linked = 0

    for shard in shards:
        summary = _read_json(shard / "summary.json")
        if not summary and not args.allow_missing_summary:
            raise SystemExit(f"Missing summary.json in {shard}")
        if summary:
            summaries.append({"shard": str(shard), **summary})

        for row in iter_jsonl(shard / "index.jsonl"):
            aid = str(row.get("arxiv_id") or "")
            if aid in seen_ids:
                duplicate_ids.append(aid)
            seen_ids.add(aid)
            index_rows.append(row)
            quality_errors.update(row.get("quality_errors") or [])
            _link_article_dir(shard / "articles" / _safe_name(aid), output / "articles" / _safe_name(aid))
            article_dirs_linked += 1

        for row in iter_jsonl(shard / "ref_abstract_cache.jsonl"):
            ref_rows.append(row)

        for row in iter_jsonl(shard / "top_k_5_index.jsonl"):
            top_rows.append(row)

        for row in iter_jsonl(shard / "ref_compact_by_key.jsonl"):
            key = row.get("cache_key")
            if key and key not in compact_by_key:
                compact_by_key[str(key)] = row

    missing_ref_rows = [
        row for row in ref_rows
        if not str(row.get("abstract") or "").strip() or not str(row.get("compact_abstract") or "").strip()
    ]
    summary = {
        "schema_version": "v3_quality_cache_merged_v1",
        "shards_root": str(args.shards_root),
        "output": str(output),
        "shards": [str(p) for p in shards],
        "article_count": len(index_rows),
        "quality_passed": sum(1 for row in index_rows if row.get("quality_passed")),
        "quality_failed": sum(1 for row in index_rows if not row.get("quality_passed")),
        "duplicate_arxiv_ids": duplicate_ids,
        "ref_cache_rows": len(ref_rows),
        "ref_cache_rows_with_missing_abstract": len(missing_ref_rows),
        "top_k_index_rows": len(top_rows),
        "compact_cache_keys": len(compact_by_key),
        "quality_error_counts": dict(quality_errors),
        "article_dirs_linked": article_dirs_linked,
        "shard_summaries": summaries,
    }

    _write_jsonl(output / "index.jsonl", index_rows)
    _write_jsonl(output / "ref_abstract_cache.jsonl", ref_rows)
    _write_jsonl(output / "top_k_5_index.jsonl", top_rows)
    _write_jsonl(output / "ref_compact_by_key.jsonl", list(compact_by_key.values()))
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if duplicate_ids:
        raise SystemExit("Duplicate arXiv ids found in shards")
    if missing_ref_rows:
        raise SystemExit("Some ref cache rows still have empty abstract or compact_abstract")


if __name__ == "__main__":
    main()
