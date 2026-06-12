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


def inspect_article_cache(cfg: dict[str, Any], arxiv_id: str) -> dict[str, Any]:
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
        "",
    ])
    for kind, snippets in (tex.get("snippets_by_kind") or {}).items():
        lines.append(f"### {kind}")
        for idx, snippet in enumerate(snippets, start=1):
            lines.extend([
                "",
                f"#### {idx}. {snippet.get('heading') or '(no heading)'}",
                f"`{snippet.get('source') or ''}`",
                "",
                snippet.get("text") or "",
            ])
    return "\n".join(lines).rstrip() + "\n"
