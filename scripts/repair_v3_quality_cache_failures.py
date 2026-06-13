#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, write_json  # noqa: E402


def _load_build_module():
    path = ROOT / "scripts" / "build_v3_quality_cache.py"
    spec = importlib.util.spec_from_file_location("build_v3_quality_cache", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_name(value: str) -> str:
    return value.replace("/", "_")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fallback_related_work(article: dict[str, Any], build: Any) -> dict[str, Any]:
    top_rw = (((article.get("prompts") or {}).get("top_k_related_work") or {}).get("related_work") or {})
    top_text = top_rw.get("text") or ""
    top_quality = top_rw.get("quality") or {}
    if top_text and top_quality.get("complete") and not top_quality.get("has_ellipsis"):
        fallback = dict(top_rw)
        fallback["source"] = "fallback_from_top_k_related_work_repair"
        return fallback
    return build._deterministic_related_work(article.get("refs") or [])


def _repair_article(article_dir: Path, build: Any) -> dict[str, Any]:
    cache_path = article_dir / "article_cache.json"
    article = json.loads(cache_path.read_text())
    aid = str(article.get("arxiv_id") or article_dir.name)
    before = article.get("quality_audit") or {}
    prompts = article.setdefault("prompts", {})
    related_entry = prompts.setdefault("related_work", {})
    related_work = related_entry.get("related_work") or {}
    quality = related_work.get("quality") or {}
    repaired_fields: list[str] = []

    if (
        "related_work_incomplete" in (before.get("errors") or [])
        or not related_work.get("text")
        or quality.get("has_ellipsis")
        or not quality.get("complete")
    ):
        fallback = _fallback_related_work(article, build)
        related_entry["related_work"] = fallback
        related_entry["prompt"] = build._related_prompt(fallback.get("text") or "")
        related_entry["source"] = fallback.get("source")
        repaired_fields.append("related_work")

    article["quality_audit"] = build._article_quality(article)
    write_json(cache_path, article)
    write_json(article_dir / "quality_audit.json", article["quality_audit"])
    if repaired_fields:
        prompt_dir = article_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "related_work.txt").write_text(prompts["related_work"]["prompt"])
    return {
        "arxiv_id": aid,
        "article_dir": str(article_dir),
        "before": before,
        "after": article["quality_audit"],
        "repaired_fields": repaired_fields,
    }


def _repair_shard(shard: Path, build: Any, target_ids: set[str] | None) -> list[dict[str, Any]]:
    index_path = shard / "index.jsonl"
    rows = list(iter_jsonl(index_path))
    repaired: list[dict[str, Any]] = []
    row_by_id = {str(row.get("arxiv_id") or ""): row for row in rows}

    candidate_ids = [
        aid for aid, row in row_by_id.items()
        if (target_ids is None or aid in target_ids) and row.get("quality_passed") is False
    ]
    if target_ids is not None:
        candidate_ids.extend(sorted(target_ids - set(candidate_ids) - {""}))
    seen: set[str] = set()
    candidate_ids = [aid for aid in candidate_ids if not (aid in seen or seen.add(aid))]

    for aid in candidate_ids:
        article_dir = shard / "articles" / _safe_name(aid)
        if not (article_dir / "article_cache.json").exists():
            continue
        result = _repair_article(article_dir, build)
        repaired.append(result)
        article = json.loads((article_dir / "article_cache.json").read_text())
        if aid in row_by_id:
            row = row_by_id[aid]
            row["quality_passed"] = bool(article["quality_audit"]["passed"])
            row["quality_errors"] = article["quality_audit"]["errors"]
            row["prompt_strategies"] = sorted(article.get("prompts") or {})

    if repaired:
        _write_jsonl(index_path, rows)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair known V3 quality-cache article failures in place.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arxiv-id", action="append")
    args = parser.parse_args()

    build = _load_build_module()
    target_ids = set(args.arxiv_id) if args.arxiv_id else None
    all_repairs: list[dict[str, Any]] = []
    for shard in sorted(args.root.glob("shard_*")):
        repairs = _repair_shard(shard, build, target_ids)
        all_repairs.extend({"shard": shard.name, **row} for row in repairs)

    log_path = args.root / "repair_log.jsonl"
    with log_path.open("a") as f:
        for row in all_repairs:
            row = {"repaired_at": int(time.time()), **row}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "root": str(args.root),
        "requested_arxiv_ids": sorted(target_ids) if target_ids else None,
        "repaired": len(all_repairs),
        "repairs": all_repairs,
        "repair_log": str(log_path),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if any((row.get("after") or {}).get("passed") is not True for row in all_repairs):
        raise SystemExit("Some repaired articles still fail quality audit")


if __name__ == "__main__":
    main()
