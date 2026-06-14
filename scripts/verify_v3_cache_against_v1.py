#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_ROOT = ROOT / "runs" / "v3_quality_cache" / "strict929_v4_sharded" / "merged"
DEFAULT_V1_CACHE = ROOT.parent / "proposal_rl" / "runs" / "dataset" / "prompt_cache"


REF_TITLE_RE = re.compile(r"^\[(\d+)\]\s+(.+?)(?:\s+\((?:\d{4}|n\.d\.)\))?\s*$")


def _load_kv_jsonl(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        if key is None:
            continue
        value = row.get("value")
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") or text.startswith("{"):
                try:
                    value = json.loads(text)
                except Exception:
                    value = row.get("value")
        out[str(key)] = value
    return out


def _collapse(value: Any) -> str:
    return " ".join(str(value or "").split())


def _ref_key(ref: dict[str, Any]) -> str:
    return str(ref.get("ref_key") or (f"arxiv:{ref.get('arxiv_id')}" if ref.get("arxiv_id") else ref.get("title") or ""))


def _v1_full_refs(refs: list[dict[str, Any]], max_refs: int, seed: int) -> list[dict[str, Any]]:
    selected = list(refs)
    random.Random(seed).shuffle(selected)
    return selected[:max_refs]


def _extract_ref_titles(prompt: str) -> list[str]:
    titles: list[str] = []
    for line in prompt.splitlines():
        match = REF_TITLE_RE.match(line.strip())
        if match:
            titles.append(_collapse(match.group(2)))
    return titles


def _titles(refs: list[dict[str, Any]]) -> list[str]:
    return [_collapse(ref.get("title")) for ref in refs]


def _word_count(text: str) -> int:
    return len(_collapse(text).split())


def _check_article(path: Path, caches: dict[str, dict[str, Any]], max_refs: int, seed: int) -> dict[str, Any]:
    article = json.loads(path.read_text())
    aid = str(article.get("arxiv_id") or path.parent.name)
    refs = list(article.get("refs") or [])
    prompts = article.get("prompts") or {}
    problems: list[str] = []
    allowed_deviations: list[str] = []

    full_refs = _v1_full_refs(refs, max_refs=max_refs, seed=seed)
    full_prompt = ((prompts.get("full_refs") or {}).get("prompt") or "")
    full_titles = _extract_ref_titles(full_prompt)
    if full_titles != _titles(full_refs):
        problems.append("full_refs_order_or_titles_mismatch")
    if (prompts.get("full_refs") or {}).get("n_refs") != len(full_refs):
        problems.append("full_refs_count_mismatch")

    cached_indices = caches["top_k_indices"].get(aid)
    expected_indices: list[int] = []
    if isinstance(cached_indices, list):
        expected_indices = [int(i) for i in cached_indices if isinstance(i, int) or str(i).isdigit()]
    expected_indices_raw = [i for i in expected_indices if 0 <= i < len(refs)][:5]
    expected_indices = []
    for i in expected_indices_raw:
        if i not in expected_indices:
            expected_indices.append(i)
    if expected_indices_raw != expected_indices:
        allowed_deviations.append("deduped_duplicate_v1_top_k_indices")
    top_meta = article.get("top_ref_selection") or {}
    top_indices = [int(i) for i in top_meta.get("indices") or []]
    if expected_indices and top_indices != expected_indices:
        problems.append("top_k_indices_mismatch")
    top_refs = [refs[i] for i in top_indices if 0 <= i < len(refs)]
    top_prompt = ((prompts.get("top_k_refs") or {}).get("prompt") or "")
    top_titles = _extract_ref_titles(top_prompt)
    if top_titles != _titles(top_refs):
        problems.append("top_k_refs_titles_mismatch")

    related = caches["related_work_annotated"].get(aid) or caches["related_work"].get(aid) or ""
    related_prompt = ((prompts.get("related_work") or {}).get("prompt") or "")
    if not related:
        problems.append("missing_v1_related_work_cache")
    elif _collapse(related) not in _collapse(related_prompt):
        problems.append("related_work_v1_block_not_embedded")
    elif "**References:**" not in related_prompt:
        problems.append("related_work_missing_references_title_list")

    top_related = caches["top_k_related_work"].get(aid) or ""
    top_related_prompt = ((prompts.get("top_k_related_work") or {}).get("prompt") or "")
    if not top_related:
        problems.append("missing_v1_top_k_related_work_cache")
    elif _collapse(top_related) not in _collapse(top_related_prompt):
        problems.append("top_k_related_work_v1_block_not_embedded")
    elif "**References:**" not in top_related_prompt:
        problems.append("top_k_related_work_missing_references_title_list")

    rq_entry = article.get("research_question") or ((prompts.get("with_research_question") or {}).get("research_question") or {})
    rq_text = str(rq_entry.get("text") or "")
    rq_words = _word_count(rq_text)
    rq_source = str(rq_entry.get("source") or "")
    with_rq_prompt = ((prompts.get("with_research_question") or {}).get("prompt") or "")
    with_rq_titles = _extract_ref_titles(with_rq_prompt)
    if with_rq_titles != _titles(full_refs):
        problems.append("with_research_question_ref_set_not_full_refs")
    if not (120 <= rq_words <= 200):
        problems.append("research_question_not_v3_long")
    if "legacy_v1_research_question" in rq_source:
        problems.append("research_question_unexpected_v1_short_source")

    return {
        "arxiv_id": aid,
        "ref_count": len(refs),
        "full_refs_count": len(full_refs),
        "top_refs_count": len(top_refs),
        "top_indices": top_indices,
        "rq_words": rq_words,
        "rq_source": rq_source,
        "problems": problems,
        "allowed_deviations": allowed_deviations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify V3 strict cache against intended V1 prompt-conditioning semantics.")
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--v1-cache", type=Path, default=DEFAULT_V1_CACHE)
    parser.add_argument("--max-refs", type=int, default=40)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    caches = {
        "top_k_indices": _load_kv_jsonl(args.v1_cache / "top_k_5_index.jsonl"),
        "related_work": _load_kv_jsonl(args.v1_cache / "related_work.jsonl"),
        "related_work_annotated": _load_kv_jsonl(args.v1_cache / "related_work_annotated.jsonl"),
        "top_k_related_work": _load_kv_jsonl(args.v1_cache / "top_k_5_related_work.jsonl"),
    }
    paths = sorted((args.v3_root / "articles").glob("*/article_cache.json"))
    if args.limit is not None:
        paths = paths[: args.limit]
    rows = [_check_article(path, caches, max_refs=args.max_refs, seed=args.shuffle_seed) for path in paths]
    problem_counts: dict[str, int] = {}
    deviation_counts: dict[str, int] = {}
    for row in rows:
        for problem in row["problems"]:
            problem_counts[problem] = problem_counts.get(problem, 0) + 1
        for deviation in row["allowed_deviations"]:
            deviation_counts[deviation] = deviation_counts.get(deviation, 0) + 1
    summary = {
        "schema_version": "v3_cache_against_v1_semantics_v1",
        "v3_root": str(args.v3_root),
        "v1_cache": str(args.v1_cache),
        "articles_checked": len(rows),
        "articles_with_problems": sum(1 for row in rows if row["problems"]),
        "articles_with_allowed_deviations": sum(1 for row in rows if row["allowed_deviations"]),
        "problem_counts": dict(sorted(problem_counts.items())),
        "allowed_deviation_counts": dict(sorted(deviation_counts.items())),
        "question_sources": {
            source: sum(1 for row in rows if row["rq_source"] == source)
            for source in sorted({row["rq_source"] for row in rows})
        },
        "passed": not problem_counts,
    }
    result = {"summary": summary, "problem_rows": [row for row in rows if row["problems"]]}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
