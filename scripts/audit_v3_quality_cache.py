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
from autoresearch_idea_harness.training_manifest import TARGET_XML_TAGS  # noqa: E402


PROMPT_STRATEGIES = {
    "full_refs",
    "top_k_refs",
    "related_work",
    "top_k_related_work",
    "with_research_question",
}
CACHE_PROMPT_STRATEGIES = {*PROMPT_STRATEGIES, "abstract"}
TEX_KINDS = {
    "abstract",
    "problem",
    "method",
    "implementation",
    "algorithm_or_system",
    "training_or_data_recipe",
    "evaluation",
    "results",
    "risks_and_limitations",
}
TARGET_TEX_KINDS = {tag for tag in TARGET_XML_TAGS if tag != "title"}


def _words(text: str | None) -> int:
    return len((text or "").split())


def _has_bad_ellipsis(text: str) -> bool:
    text = " ".join((text or "").split())
    if "…" in text:
        return True
    if "..." not in text:
        return False
    if text.endswith("..."):
        return True
    sequence_ellipsis = re.compile(r"[A-Za-z0-9_{}\\^/()+\\-]+,\s*\.\.\.,\s*[A-Za-z0-9_{}\\^/()+\\-]+")
    cleaned = sequence_ellipsis.sub("", text)
    quoted_example_ellipsis = re.compile(r"""["'“‘][^"'“”‘’]{0,160}\.\.\.[^"'“”‘’]{0,80}["'”’]""")
    cleaned = quoted_example_ellipsis.sub("", cleaned)
    return "..." in cleaned


def _complete(text: str | None) -> bool:
    text = " ".join((text or "").split())
    return bool(text) and bool(re.search(r"""[.!?。！？)"'\]]$""", text)) and not _has_bad_ellipsis(text)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _audit_article(article_dir: Path, index_row: dict[str, Any] | None) -> dict[str, Any]:
    aid = article_dir.name
    errors: list[str] = []
    required = [
        "article_cache.json",
        "refs.json",
        "quality_audit.json",
        "tex_snippets.json",
        "target_tex_snippets.json",
        "top_k_5_index.json",
    ]
    for name in required:
        if not (article_dir / name).exists():
            errors.append(f"missing_file:{name}")
    for strategy in CACHE_PROMPT_STRATEGIES:
        if not (article_dir / "prompts" / f"{strategy}.txt").exists():
            errors.append(f"missing_prompt:{strategy}")
    if errors:
        return {"arxiv_id": aid, "article_dir": str(article_dir), "passed": False, "errors": errors}

    article = _load_json(article_dir / "article_cache.json")
    refs = _load_json(article_dir / "refs.json")
    top = _load_json(article_dir / "top_k_5_index.json")
    audit = _load_json(article_dir / "quality_audit.json")

    if index_row and str(index_row.get("arxiv_id")) != str(article.get("arxiv_id")):
        errors.append("index_article_id_mismatch")
    if audit.get("passed") is not True:
        errors.append(f"article_quality_failed:{','.join(audit.get('errors') or [])}")
    if article.get("excluded_missing_abstract_refs"):
        errors.append("excluded_missing_abstract_refs")

    for idx, ref in enumerate(refs):
        if not str(ref.get("abstract") or "").strip():
            errors.append(f"ref_empty_abstract:{idx}")
        compact = ref.get("compact_abstract") or ""
        if not compact.strip():
            errors.append(f"ref_empty_compact:{idx}")
        elif _words(compact) > 200 or not _complete(compact):
            errors.append(f"ref_bad_compact:{idx}")

    q = article.get("research_question") or {}
    qtext = q.get("text") if isinstance(q, dict) else q
    if not (120 <= _words(qtext) <= 200) or not _complete(qtext):
        errors.append("bad_research_question")

    article_indices = (article.get("top_ref_selection") or {}).get("indices") or []
    if top.get("value") != article_indices:
        errors.append("top_k_index_mismatch")

    snippets = article.get("tex_snippets") or {}
    for kind in TEX_KINDS:
        values = snippets.get(kind) or []
        if not values:
            errors.append(f"missing_tex:{kind}")
        for i, snippet in enumerate(values):
            text = snippet.get("text") or ""
            if not _complete(text):
                errors.append(f"bad_tex:{kind}:{i}")
            if "\\begin" in text or "\\end" in text:
                errors.append(f"raw_tex:{kind}:{i}")

    target_snippets = article.get("target_tex_snippets") or {}
    for kind in TARGET_TEX_KINDS:
        values = target_snippets.get(kind) or []
        if not values:
            errors.append(f"missing_target_tex:{kind}")
        for i, snippet in enumerate(values):
            text = snippet.get("text") or ""
            if not _complete(text):
                errors.append(f"bad_target_tex:{kind}:{i}")
            if "\\begin" in text or "\\end" in text:
                errors.append(f"raw_target_tex:{kind}:{i}")

    return {"arxiv_id": article.get("arxiv_id") or aid, "article_dir": str(article_dir), "passed": not errors, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V3 quality cache outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root
    index_rows = {str(row.get("arxiv_id")): row for row in iter_jsonl(root / "index.jsonl")}
    article_dirs = sorted((root / "articles").glob("*")) if (root / "articles").exists() else []
    results = [_audit_article(article_dir, index_rows.get(article_dir.name)) for article_dir in article_dirs if article_dir.is_dir()]

    ref_rows = list(iter_jsonl(root / "ref_abstract_cache.jsonl"))
    empty_ref_cache = [
        row for row in ref_rows
        if not str(row.get("abstract") or "").strip() or not str(row.get("compact_abstract") or "").strip()
    ]
    top_rows = list(iter_jsonl(root / "top_k_5_index.jsonl"))
    error_counts: Counter[str] = Counter()
    for result in results:
        error_counts.update(result.get("errors") or [])
    summary = {
        "schema_version": "v3_quality_cache_audit_v1",
        "root": str(root),
        "index_rows": len(index_rows),
        "article_dirs": len(article_dirs),
        "articles_passed": sum(1 for row in results if row["passed"]),
        "articles_failed": sum(1 for row in results if not row["passed"]),
        "ref_cache_rows": len(ref_rows),
        "ref_cache_empty_rows": len(empty_ref_cache),
        "top_k_index_rows": len(top_rows),
        "error_counts": dict(error_counts),
        "passed": (
            len(index_rows) == len(article_dirs)
            and all(row["passed"] for row in results)
            and not empty_ref_cache
            and len(top_rows) == len(index_rows)
        ),
    }
    output = args.output or (root / "audit_summary.json")
    write_json(output, {"summary": summary, "articles": results, "empty_ref_cache_rows": empty_ref_cache[:50]})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
