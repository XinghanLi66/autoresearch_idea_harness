#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.prompt_properties import KeyValueCache  # noqa: E402
from autoresearch_idea_harness.training_manifest import PROPOSAL_FORMAT  # noqa: E402


DEFAULT_CACHE_ROOT = ROOT / "runs" / "v3_quality_cache" / "strict929_v4_sharded" / "merged"
DEFAULT_V1_PROMPT_CACHE = ROOT.parent / "proposal_rl" / "runs" / "dataset" / "prompt_cache"
PROMPT_STRATEGIES = [
    "abstract",
    "full_refs",
    "top_k_refs",
    "related_work",
    "top_k_related_work",
    "with_research_question",
]


def _collapse(value: Any) -> str:
    return " ".join(str(value or "").split())


def _quality_text(text: str) -> dict[str, Any]:
    text = _collapse(text)
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "tokens": len(words),
        "complete": bool(text) and text[-1:] in '.!?"\')]}。！？',
    }


def _question_quality_ok(question: dict[str, Any]) -> bool:
    quality = question.get("quality") or _quality_text(question.get("text") or "")
    words = quality.get("words") or 0
    return bool(question.get("text")) and 120 <= words <= 200 and bool(quality.get("complete"))


def _recover_question_from_logs(article_dir: Path) -> dict[str, Any] | None:
    rq_dir = article_dir / "llm_calls" / "research_question"
    if not rq_dir.exists():
        return None
    candidates = sorted(rq_dir.glob("attempt_*/response.json"))
    best: dict[str, Any] | None = None
    for path in candidates:
        try:
            row = json.loads(path.read_text())
        except Exception:
            continue
        text = _collapse(row.get("text"))
        question = {
            "text": text,
            "source": "restored_v3_long_research_question_from_llm_log",
            "quality": _quality_text(text),
            "restored_from": str(path),
        }
        if _question_quality_ok(question):
            return question
        if text and best is None:
            best = question
    return best


def _ref_entry(idx: int, ref: dict[str, Any], *, use_compact: bool = True) -> str:
    abstract = ref.get("compact_abstract") if use_compact else ref.get("abstract")
    if isinstance(abstract, dict):
        abstract = abstract.get("text")
    abstract = _collapse(abstract)
    return (
        f"[{idx}] {ref.get('title') or 'Unknown'} ({ref.get('year') or 'n.d.'})\n"
        f"Abstract: {abstract or '(no abstract available)'}"
    )


def _ref_block(refs: list[dict[str, Any]], *, use_compact: bool = True) -> str:
    return "\n\n".join(_ref_entry(i + 1, ref, use_compact=use_compact) for i, ref in enumerate(refs))


def _title_list(refs: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{i + 1}] {_collapse(ref.get('title')) or 'Unknown'}" for i, ref in enumerate(refs))


def _standard_prompt(refs: list[dict[str, Any]]) -> str:
    return (
        f"Below are {len(refs)} papers from a researcher's reading list. "
        "Based on these references, propose a novel research direction.\n\n"
        f"{_ref_block(refs, use_compact=True)}\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _question_prompt(refs: list[dict[str, Any]], question: str) -> str:
    return (
        f"Below are {len(refs)} papers from a researcher's reading list. "
        "Based on these references, propose a novel research direction.\n\n"
        f"{_ref_block(refs, use_compact=True)}\n\n"
        "The researcher has identified the following open question as their primary motivation:\n"
        f"\"{question}\"\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _related_prompt(related_work: str, refs: list[dict[str, Any]], *, focused: bool) -> str:
    block = related_work or "(related_work cache missing)"
    if "**References:**" not in block:
        block = f"{block}\n\n**References:**\n{_title_list(refs) or '(reference list missing)'}"
    area = "focused area" if focused else "area"
    return (
        f"A researcher has been studying the following {area} of the literature:\n\n"
        f"{block}\n\n"
        "Based on this background, propose a novel research direction.\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _load_json_cache(path: Path) -> dict[str, Any]:
    cache = KeyValueCache(path)
    return cache.data


def _select_top_refs(article: dict[str, Any], refs: list[dict[str, Any]], top_indices_cache: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aid = str(article.get("arxiv_id") or "")
    cached = top_indices_cache.get(aid)
    indices: list[int] = []
    source = "legacy_v1_top_k_5_index"
    if isinstance(cached, str):
        try:
            cached = json.loads(cached)
        except Exception:
            cached = None
    if isinstance(cached, list):
        for idx in cached:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < len(refs) and i not in indices:
                indices.append(i)
            if len(indices) >= 5:
                break
    if not indices:
        source = "article_top_ref_selection_or_first5"
        indices = [int(i) for i in ((article.get("top_ref_selection") or {}).get("indices") or []) if str(i).isdigit()]
        indices = [i for i in indices if 0 <= i < len(refs)][:5]
    if not indices:
        indices = list(range(min(5, len(refs))))
    return [refs[i] for i in indices], {"source": source, "top_k": 5, "indices": indices}


def _ordered_full_refs(refs: list[dict[str, Any]], *, max_refs: int, seed: int) -> list[dict[str, Any]]:
    # Matches V1 FullRefsBuilder when pinned_count is absent: shuffle all refs with seed 42, then truncate.
    selected = list(refs)
    random.Random(seed).shuffle(selected)
    return selected[:max_refs]


def _rebuild_article(
    path: Path,
    *,
    caches: dict[str, dict[str, Any]],
    max_refs: int,
    shuffle_seed: int,
    dry_run: bool,
) -> dict[str, Any]:
    article = json.loads(path.read_text())
    aid = str(article.get("arxiv_id") or path.parent.name)
    refs = list(article.get("refs") or [])
    full_refs = _ordered_full_refs(refs, max_refs=max_refs, seed=shuffle_seed)
    top_refs, top_meta = _select_top_refs(article, refs, caches["top_k_indices"])

    related_all = caches["related_work_annotated"].get(aid) or caches["related_work"].get(aid) or ""
    related_top = caches["top_k_related_work"].get(aid) or ""
    existing_question = article.get("research_question") or {}
    if _question_quality_ok(existing_question):
        question = {
            **existing_question,
            "source": existing_question.get("source") or "existing_v3_long_research_question",
        }
    else:
        question = _recover_question_from_logs(path.parent) or {}
    if not _question_quality_ok(question):
        legacy_question = caches["research_question"].get(aid) or ""
        question = {
            "text": legacy_question,
            "source": "legacy_v1_research_question_full_refs_short_fallback" if legacy_question else "missing_research_question",
            "quality": _quality_text(legacy_question),
        }
    question_text = question.get("text") or ""
    target_compact = ((article.get("target_abstract") or {}).get("compact") or {})
    target_compact_text = target_compact.get("text") or ""

    prompts = {
        "abstract": {
            "prompt": (
                "Below is the target paper abstract. Based only on this abstract, "
                "rewrite the work as one concrete research proposal.\n\n"
                f"Title: {article.get('title') or 'Unknown'}\n\n"
                f"Abstract: {target_compact_text}\n\n"
                f"{PROPOSAL_FORMAT}"
            ),
            "n_refs": 0,
            "source": "target_compact_abstract",
            "compact_abstract": target_compact,
        },
        "full_refs": {
            "prompt": _standard_prompt(full_refs),
            "n_refs": len(full_refs),
            "source": "v1_full_refs_shuffle_seed_42_v3_compact_abstracts",
        },
        "top_k_refs": {
            "prompt": _standard_prompt(top_refs),
            "n_refs": len(top_refs),
            "source": top_meta["source"],
        },
        "related_work": {
            "prompt": _related_prompt(str(related_all), full_refs, focused=False),
            "n_refs": len(full_refs),
            "source": "legacy_v1_related_work_annotated" if related_all else "missing_related_work_cache",
            "related_work": {
                "text": str(related_all),
                "source": "legacy_v1_related_work_annotated" if related_all else "missing",
                "quality": _quality_text(str(related_all)),
            },
        },
        "top_k_related_work": {
            "prompt": _related_prompt(str(related_top), top_refs, focused=True),
            "n_refs": len(top_refs),
            "source": "legacy_v1_top_k_5_related_work" if related_top else "missing_top_k_related_work_cache",
            "related_work": {
                "text": str(related_top),
                "source": "legacy_v1_top_k_5_related_work" if related_top else "missing",
                "quality": _quality_text(str(related_top)),
            },
        },
        "with_research_question": {
            "prompt": _question_prompt(full_refs, question_text),
            "n_refs": len(full_refs),
            "source": question["source"],
            "research_question": question,
        },
    }

    old = article.get("prompts") or {}
    changed = [strategy for strategy in PROMPT_STRATEGIES if (old.get(strategy) or {}).get("prompt") != prompts[strategy]["prompt"]]
    if changed and not dry_run:
        article["prompts"] = prompts
        article["top_refs"] = top_refs
        article["top_ref_selection"] = top_meta
        article["research_question"] = question
        article["prompt_rebuild"] = {
            "version": "v1_conditioning_semantics_v3_xml_targets",
            "rebuilt_at": int(time.time()),
            "changed_strategies": changed,
            "v1_prompt_cache": str(DEFAULT_V1_PROMPT_CACHE),
            "notes": [
                "full_refs and with_research_question use the V1 full-ref conditioning set",
                "top_k strategies use V1 top_k_5_index when available",
                "related_work strategies use V1 annotated narrative plus title list caches",
                "proposal output schema remains the V3 XML target schema",
            ],
        }
        path.write_text(json.dumps(article, indent=2, ensure_ascii=False) + "\n")
        prompt_dir = path.parent / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for strategy, entry in prompts.items():
            (prompt_dir / f"{strategy}.txt").write_text(entry["prompt"])

    return {
        "arxiv_id": aid,
        "changed": changed,
        "ref_count": len(refs),
        "full_refs_n": len(full_refs),
        "top_refs_n": len(top_refs),
        "has_v1_related_work": bool(related_all),
        "has_v1_top_k_related_work": bool(related_top),
        "has_v1_research_question": bool(caches["research_question"].get(aid)),
        "question_source": question.get("source"),
        "question_words": (question.get("quality") or {}).get("words"),
        "question_quality_ok": _question_quality_ok(question),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild strict V3 article prompts with V1 conditioning strategy semantics.")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--v1-prompt-cache", default=str(DEFAULT_V1_PROMPT_CACHE))
    parser.add_argument("--article-id", action="append", default=[])
    parser.add_argument("--max-refs", type=int, default=40)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    v1_root = Path(args.v1_prompt_cache)
    caches = {
        "top_k_indices": _load_json_cache(v1_root / "top_k_5_index.jsonl"),
        "related_work": _load_json_cache(v1_root / "related_work.jsonl"),
        "related_work_annotated": _load_json_cache(v1_root / "related_work_annotated.jsonl"),
        "top_k_related_work": _load_json_cache(v1_root / "top_k_5_related_work.jsonl"),
        "research_question": _load_json_cache(v1_root / "research_question.jsonl"),
    }

    root = Path(args.cache_root)
    article_root = root / "articles"
    if args.article_id:
        paths = [article_root / article_id / "article_cache.json" for article_id in args.article_id]
    else:
        paths = sorted(article_root.glob("*/article_cache.json"))

    rows = []
    for path in paths:
        if not path.exists():
            rows.append({"path": str(path), "error": "missing_article_cache"})
            continue
        rows.append(
            _rebuild_article(
                path,
                caches=caches,
                max_refs=args.max_refs,
                shuffle_seed=args.shuffle_seed,
                dry_run=args.dry_run,
            )
        )

    summary = {
        "cache_root": str(root),
        "v1_prompt_cache": str(v1_root),
        "dry_run": args.dry_run,
        "articles_seen": len(rows),
        "articles_changed": sum(1 for row in rows if row.get("changed")),
        "strategy_changes": {
            strategy: sum(1 for row in rows if strategy in (row.get("changed") or []))
            for strategy in PROMPT_STRATEGIES
        },
        "missing_v1_related_work": sum(1 for row in rows if not row.get("has_v1_related_work")),
        "missing_v1_top_k_related_work": sum(1 for row in rows if not row.get("has_v1_top_k_related_work")),
        "missing_v1_research_question": sum(1 for row in rows if not row.get("has_v1_research_question")),
        "bad_question_quality": sum(1 for row in rows if not row.get("question_quality_ok")),
        "question_sources": {
            source: sum(1 for row in rows if row.get("question_source") == source)
            for source in sorted({str(row.get("question_source")) for row in rows})
        },
        "errors": [row for row in rows if row.get("error")],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.dry_run:
        (root / "prompt_rebuild_v1_semantics_audit.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
