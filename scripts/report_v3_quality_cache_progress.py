#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, write_json


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


def _iter_jsonl_safe(path: Path):
    if not path.exists():
        return []
    return list(iter_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Report V3 quality cache shard progress.")
    parser.add_argument("--root", type=Path, default=Path("runs/v3_quality_cache/strict929_v1_sharded"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    total = {
        "processed": 0,
        "quality_passed": 0,
        "quality_failed": 0,
        "errors": 0,
        "ref_cache_rows": 0,
        "top_k_index_rows": 0,
        "excluded_missing_abstract_count": 0,
    }
    for shard in sorted(args.root.glob("shard_*")):
        index_rows = _iter_jsonl_safe(shard / "index.jsonl")
        summary = {}
        if (shard / "summary.json").exists():
            summary = json.loads((shard / "summary.json").read_text())
        errors = _iter_jsonl_safe(shard / "errors.jsonl")
        row = {
            "shard": shard.name,
            "processed": len(index_rows),
            "quality_passed": sum(1 for item in index_rows if item.get("quality_passed")),
            "quality_failed": sum(1 for item in index_rows if item.get("quality_passed") is False),
            "errors": len(errors),
            "ref_cache_rows": _count_jsonl(shard / "ref_abstract_cache.jsonl"),
            "top_k_index_rows": _count_jsonl(shard / "top_k_5_index.jsonl"),
            "excluded_missing_abstract_count": sum(item.get("excluded_missing_abstract_count", 0) for item in index_rows),
            "finished": bool(summary),
            "last_arxiv_id": index_rows[-1].get("arxiv_id") if index_rows else None,
        }
        rows.append(row)
        for key in total:
            total[key] += row[key]
    report = {
        "root": str(args.root),
        "shards": rows,
        "total": total,
        "finished_shards": sum(1 for row in rows if row["finished"]),
        "num_shards": len(rows),
        "all_finished": bool(rows) and all(row["finished"] for row in rows),
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
