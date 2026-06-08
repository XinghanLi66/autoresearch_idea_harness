from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import ensure_src_paths, short_text


EVIDENCE_KINDS = {"method", "implementation", "evaluation", "results"}


def extract_ref_evidence(
    arxiv_id: str | None,
    cfg: dict[str, Any],
    max_tex_chars: int,
    max_snippets: int,
    snippet_char_limit: int,
) -> dict[str, Any]:
    """Return reference-side evidence using the existing proposal_rl TeX parser."""
    if not arxiv_id:
        return {
            "evidence_status": "missing_arxiv_id",
            "tex_status": "not_checked",
            "snippets": [],
        }

    ensure_src_paths(cfg)
    from data.synthesize_tex_targets import extract_tex_context

    ctx = extract_tex_context(arxiv_id, Path(cfg["arxiv_root"]), max_tex_chars=max_tex_chars)
    tex_status = ctx.get("tex_status", "unknown")
    if tex_status != "ok":
        return {
            "evidence_status": "no_tex",
            "tex_status": tex_status,
            "paper_dir": ctx.get("paper_dir"),
            "tex_dir": ctx.get("tex_dir"),
            "snippets": [],
        }

    sections = ctx.get("tex_sections", [])
    preferred = [s for s in sections if s.get("kind") in EVIDENCE_KINDS]
    if not preferred:
        preferred = sections

    snippets = []
    for idx, section in enumerate(preferred[:max_snippets]):
        snippets.append({
            "snippet_id": f"{arxiv_id}:snippet:{idx}",
            "kind": section.get("kind", "other"),
            "heading": section.get("heading", ""),
            "source": section.get("source", ""),
            "text": short_text(section.get("text", ""), snippet_char_limit),
            "provenance": {
                "arxiv_id": arxiv_id,
                "paper_dir": ctx.get("paper_dir"),
                "tex_dir": ctx.get("tex_dir"),
                "source": section.get("source", ""),
                "heading": section.get("heading", ""),
            },
        })

    return {
        "evidence_status": "tex_ok" if snippets else "tex_empty",
        "tex_status": tex_status,
        "paper_dir": ctx.get("paper_dir"),
        "tex_dir": ctx.get("tex_dir"),
        "tex_file_count": ctx.get("tex_file_count", 0),
        "tex_section_count": ctx.get("tex_section_count", 0),
        "snippets": snippets,
    }
