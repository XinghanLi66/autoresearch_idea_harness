from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .io import ensure_src_paths, iter_jsonl, load_dataset_records, short_text
from .prompt_properties import (
    PROMPT_STRATEGIES,
    build_prompt_variants,
    compact_research_question,
    load_prompt_property_caches,
)
from .training_manifest import PROPOSAL_FORMAT


ROOT = Path(__file__).resolve().parents[2]
STRICT_QUALITY_CACHE_ROOT = ROOT / "runs" / "v3_quality_cache" / "strict929_v4_sharded" / "merged"
PROMPT_VARIANTS = [
    "abstract",
    "full_refs",
    "top_k_refs",
    "related_work",
    "top_k_related_work",
    "with_research_question",
]


def _find_classified_row(cfg: dict[str, Any], arxiv_id: str) -> dict[str, Any] | None:
    path = Path(cfg["classified_papers"])
    if not path.exists():
        return None
    for row in iter_jsonl(path):
        if str(row.get("arxiv_id") or "") == arxiv_id:
            return row
    return None


def _find_dataset_record(cfg: dict[str, Any], arxiv_id: str) -> dict[str, Any] | None:
    records = load_dataset_records(Path(cfg["dataset_dir"]), ["train", "val", "test"])
    return records.get(arxiv_id)


def _extract_target_tex_details(cfg: dict[str, Any], arxiv_id: str, max_tex_chars: int = 24000) -> dict[str, Any]:
    ensure_src_paths(cfg)
    from data.synthesize_tex_targets import extract_tex_context

    ctx = extract_tex_context(arxiv_id, Path(cfg["arxiv_root"]), max_tex_chars=max_tex_chars)
    sections = ctx.get("tex_sections") or []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        kind = str(section.get("kind") or "other")
        if kind not in {"method", "implementation", "evaluation", "results", "problem", "abstract"}:
            continue
        by_kind.setdefault(kind, []).append({
            "kind": kind,
            "heading": section.get("heading"),
            "source": section.get("source"),
            "text": short_text(section.get("text"), 1800),
        })
    return {
        "tex_status": ctx.get("tex_status"),
        "paper_dir": ctx.get("paper_dir"),
        "tex_dir": ctx.get("tex_dir"),
        "tex_file_count": ctx.get("tex_file_count"),
        "tex_section_count": ctx.get("tex_section_count"),
        "snippets_by_kind": {k: v[:3] for k, v in sorted(by_kind.items())},
        "snippet_counts_by_kind": dict(Counter(s.get("kind") for s in sections)),
    }


def _quality_count_map(snippets_by_kind: dict[str, Any]) -> dict[str, int]:
    return {str(kind): len(items or []) for kind, items in sorted(snippets_by_kind.items())}


def _clip_snippets(snippets_by_kind: dict[str, Any], *, max_items: int = 3, max_chars: int = 1800) -> dict[str, list[dict[str, Any]]]:
    clipped: dict[str, list[dict[str, Any]]] = {}
    for kind, snippets in sorted((snippets_by_kind or {}).items()):
        items: list[dict[str, Any]] = []
        for snippet in (snippets or [])[:max_items]:
            if not isinstance(snippet, dict):
                continue
            items.append({
                **snippet,
                "text": short_text(snippet.get("text"), max_chars),
            })
        clipped[str(kind)] = items
    return clipped


def _quality_article_cache_path(arxiv_id: str) -> Path | None:
    path = STRICT_QUALITY_CACHE_ROOT / "articles" / arxiv_id / "article_cache.json"
    return path if path.exists() else None


def _quality_prompt_variants(article: dict[str, Any]) -> dict[str, Any]:
    prompts = article.get("prompts") or {}
    variants: dict[str, Any] = {}
    for strategy in PROMPT_VARIANTS:
        item = prompts.get(strategy) or {}
        prompt = item.get("prompt") or ""
        metadata = {key: value for key, value in item.items() if key != "prompt"}
        variants[strategy] = {
            "status": "ready" if prompt else "missing",
            "chars": len(prompt),
            "metadata": {
                **metadata,
                "cache_source": "strict929_quality_cache",
            },
            "prompt": prompt,
        }
    return variants


def _quality_tex_details(article: dict[str, Any]) -> dict[str, Any]:
    curated = article.get("tex_snippets") or {}
    raw = article.get("raw_tex_snippets") or {}
    target = article.get("target_tex_snippets") or {}
    return {
        "tex_status": "quality_cache_ready" if curated else "missing",
        "paper_dir": (article.get("source") or {}).get("paper_dir"),
        "tex_dir": (article.get("source") or {}).get("tex_dir"),
        "tex_file_count": (article.get("source") or {}).get("tex_file_count"),
        "tex_section_count": sum(_quality_count_map(raw).values()),
        "snippets_by_kind": _clip_snippets(curated),
        "raw_snippets_by_kind": _clip_snippets(raw),
        "target_snippets_by_kind": _clip_snippets(target),
        "snippet_counts_by_kind": _quality_count_map(curated),
        "raw_snippet_counts_by_kind": _quality_count_map(raw),
        "target_snippet_counts_by_kind": _quality_count_map(target),
    }


def _inspect_quality_article_cache(arxiv_id: str) -> dict[str, Any] | None:
    path = _quality_article_cache_path(arxiv_id)
    if not path:
        return None
    article = json.loads(path.read_text())
    question = article.get("research_question") or {}
    question_quality = question.get("quality") or {}
    target_abstract = article.get("target_abstract") or {}
    compact_abstract = target_abstract.get("compact") or {}
    refs = article.get("refs") or []
    top_refs = _quality_top_refs(article, refs)
    source = article.get("source") or {}
    article_dir = path.parent

    return {
        "kind": "article_cache_inspection",
        "arxiv_id": arxiv_id,
        "cache_source": "strict929_quality_cache",
        "quality_cache_path": str(path),
        "found_quality_cache": True,
        "found_dataset_record": True,
        "found_classified_row": True,
        "metadata": {
            "title": article.get("title"),
            "created": article.get("created"),
            "categories": article.get("categories") or [],
            "category_family": article.get("category_family"),
            "paper_type": article.get("paper_type"),
            "training_route": article.get("training_route"),
            "detail_support_level": article.get("detail_support_level"),
            "sample_id": article.get("sample_id"),
            "ref_count": len(refs),
            "top_ref_count": len(top_refs),
            "schema_version": article.get("schema_version"),
        },
        "quality": article.get("quality") or {},
        "quality_audit": article.get("quality_audit") or {},
        "open_question": {
            "text": question.get("text") or "",
            "source": question.get("source"),
            "tokens": question_quality.get("tokens") or question_quality.get("words"),
            "words": question_quality.get("words"),
            "complete": question_quality.get("complete"),
            "quality": question_quality,
            "raw": json.dumps(question, indent=2, ensure_ascii=False),
        },
        "target_abstract": {
            "original": target_abstract.get("original") or "",
            "compact": compact_abstract,
            "compact_text": compact_abstract.get("text") or "",
        },
        "prompt_variants": _quality_prompt_variants(article),
        "tex_details": _quality_tex_details(article),
        "references": {
            "ref_count": len(refs),
            "top_ref_count": len(top_refs),
            "top_ref_selection": article.get("top_ref_selection") or {},
            "top_refs": top_refs,
            "refs_preview": refs[:10],
        },
        "cache_files": {
            "quality_article_cache": [str(path)],
            "quality_article_dir": [str(article_dir)],
            "prompts": [str(article_dir / "prompts" / f"{strategy}.txt") for strategy in PROMPT_VARIANTS],
            "refs": [str(article_dir / "refs.json")],
            "top_k_index": [str(article_dir / "top_k_5_index.json")],
            "tex_snippets": [str(article_dir / "tex_snippets.json")],
            "raw_tex_snippets": [str(article_dir / "raw_tex_snippets.json")],
            "target_tex_snippets": [str(article_dir / "target_tex_snippets.json")],
            "quality_audit": [str(article_dir / "quality_audit.json")],
        },
    }


def _quality_top_refs(article: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indices = ((article.get("top_ref_selection") or {}).get("indices") or [])
    selected: list[dict[str, Any]] = []
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if 0 <= i < len(refs):
            selected.append(refs[i])
    if selected:
        return selected

    raw_top = article.get("top_refs") or []
    detailed: list[dict[str, Any]] = []
    for top_ref in raw_top:
        ref_key = top_ref.get("ref_key")
        arxiv_id = top_ref.get("arxiv_id")
        match = next(
            (
                ref
                for ref in refs
                if (ref_key and ref.get("ref_key") == ref_key) or (arxiv_id and ref.get("arxiv_id") == arxiv_id)
            ),
            None,
        )
        detailed.append(match or top_ref)
    return detailed


def inspect_article_cache(cfg: dict[str, Any], arxiv_id: str) -> dict[str, Any]:
    quality_report = _inspect_quality_article_cache(arxiv_id)
    if quality_report:
        return quality_report

    caches = load_prompt_property_caches(cfg)
    record = _find_dataset_record(cfg, arxiv_id)
    classified = _find_classified_row(cfg, arxiv_id)
    raw_question = caches["research_question"].get(arxiv_id)
    question = compact_research_question(arxiv_id, raw_question, caches)

    prompt_variants: dict[str, Any] = {}
    if record:
        variants = build_prompt_variants(record, caches, PROPOSAL_FORMAT)
        for strategy in PROMPT_STRATEGIES:
            item = variants.get(strategy) or {}
            prompt = item.get("prompt") or ""
            prompt_variants[strategy] = {
                "status": item.get("status"),
                "chars": len(prompt),
                "metadata": item.get("metadata") or {},
                "prompt": prompt,
            }

    tex_details = _extract_target_tex_details(cfg, arxiv_id) if classified else {
        "tex_status": "not_checked",
        "snippets_by_kind": {},
    }

    return {
        "kind": "article_cache_inspection",
        "arxiv_id": arxiv_id,
        "found_dataset_record": bool(record),
        "found_classified_row": bool(classified),
        "metadata": {
            "title": (classified or record or {}).get("title"),
            "created": (classified or {}).get("created"),
            "categories": (classified or {}).get("categories") or [],
            "category_family": (classified or {}).get("category_family"),
            "paper_type": (classified or {}).get("paper_type"),
            "training_route": (classified or {}).get("training_route"),
            "detail_support_level": (classified or {}).get("detail_support_level"),
            "ref_count": len((record or {}).get("refs") or []),
        },
        "open_question": {
            **question,
            "raw": raw_question,
        },
        "prompt_variants": prompt_variants,
        "tex_details": tex_details,
        "cache_files": {
            name: [str(p) for p in cache.paths]
            for name, cache in caches.items()
        },
    }


def render_article_cache_markdown(report: dict[str, Any], *, include_prompts: bool = True) -> str:
    lines = [
        f"# Article Cache: {report.get('arxiv_id')}",
        "",
        f"- cache_source: `{report.get('cache_source') or 'legacy_prompt_property_cache'}`",
        f"- quality_cache_path: `{report.get('quality_cache_path') or ''}`",
        "",
        "## Metadata",
        "```json",
        json.dumps(report.get("metadata") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Open Question",
        f"- source: `{(report.get('open_question') or {}).get('source')}`",
        f"- tokens: `{(report.get('open_question') or {}).get('tokens')}`",
        f"- complete: `{(report.get('open_question') or {}).get('complete')}`",
        "",
        (report.get("open_question") or {}).get("text") or "(missing)",
        "",
        "## Target Abstract",
        f"- compact_source: `{((report.get('target_abstract') or {}).get('compact') or {}).get('source')}`",
        f"- compact_words: `{(((report.get('target_abstract') or {}).get('compact') or {}).get('quality') or {}).get('words')}`",
        "",
        (report.get("target_abstract") or {}).get("compact_text") or "(missing)",
        "",
        "## References",
        "```json",
        json.dumps(report.get("references") or {}, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Prompt Variants",
    ]
    for strategy, item in (report.get("prompt_variants") or {}).items():
        lines.extend([
            "",
            f"### {strategy}",
            f"- status: `{item.get('status')}`",
            f"- chars: `{item.get('chars')}`",
            "- metadata:",
            "```json",
            json.dumps(item.get("metadata") or {}, indent=2, ensure_ascii=False),
            "```",
        ])
        if include_prompts:
            lines.extend(["", "```text", item.get("prompt") or "", "```"])
    lines.extend(["", "## TeX Details"])
    tex = report.get("tex_details") or {}
    lines.extend([
        f"- tex_status: `{tex.get('tex_status')}`",
        f"- tex_dir: `{tex.get('tex_dir')}`",
        f"- curated_counts: `{tex.get('snippet_counts_by_kind')}`",
        f"- raw_counts: `{tex.get('raw_snippet_counts_by_kind')}`",
        f"- target_counts: `{tex.get('target_snippet_counts_by_kind')}`",
        "",
    ])
    for kind, snippets in (tex.get("snippets_by_kind") or {}).items():
        lines.append(f"### curated/{kind}")
        for idx, snippet in enumerate(snippets, start=1):
            lines.extend([
                "",
                f"#### {idx}. {snippet.get('heading') or '(no heading)'}",
                f"`{snippet.get('source') or ''}`",
                "",
                snippet.get("text") or "",
            ])
    return "\n".join(lines).rstrip() + "\n"
