from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, load_dataset_records, short_text, stable_id, write_json, write_jsonl
from .tex import extract_ref_evidence


def _ref_key(ref: dict[str, Any]) -> str:
    aid = ref.get("arxiv_id")
    if aid:
        return f"arxiv:{aid}"
    return stable_id("ref", ref.get("title", ""), ref.get("year"), length=16)


def _trim_ref(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "ref_key": _ref_key(ref),
        "arxiv_id": ref.get("arxiv_id"),
        "title": ref.get("title"),
        "year": ref.get("year"),
        "abstract": ref.get("abstract"),
    }


def build_paper_bank(
    cfg: dict[str, Any],
    limit: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    pb_cfg = cfg.get("paper_bank", {})
    splits = list(pb_cfg.get("splits", ["train", "val", "test"]))
    dataset_records = load_dataset_records(Path(cfg["dataset_dir"]), splits)
    output_dir = output_dir or Path(cfg["runs_dir"]) / "paper_bank"

    papers = []
    refs: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(cfg["classified_papers"]):
        aid = row.get("arxiv_id")
        if not aid:
            continue
        raw = dataset_records.get(aid)
        if raw is None:
            continue
        raw_refs = list(raw.get("refs") or [])
        trimmed_refs = [_trim_ref(r) for r in raw_refs]
        for ref in raw_refs:
            refs.setdefault(_ref_key(ref), _trim_ref(ref))

        papers.append({
            "arxiv_id": aid,
            "split": row.get("split"),
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "categories": row.get("categories") or [],
            "primary_category": row.get("primary_category"),
            "category_family": row.get("category_family"),
            "paper_type": row.get("paper_type"),
            "detail_support_level": row.get("detail_support_level"),
            "reference_evidence_need": row.get("reference_evidence_need"),
            "recommended_condition": row.get("recommended_condition"),
            "recommended_target_schema": row.get("recommended_target_schema"),
            "training_route": row.get("training_route"),
            "tex_status": row.get("tex_status"),
            "tex_features": row.get("tex_features") or {},
            "refs": trimmed_refs,
            "ref_count": len(trimmed_refs),
        })
        if limit is not None and len(papers) >= limit:
            break

    ref_evidence = []
    max_tex_chars = int(pb_cfg.get("max_ref_tex_chars", 12000))
    max_snippets = int(pb_cfg.get("max_snippets_per_ref", 4))
    snippet_limit = int(pb_cfg.get("snippet_char_limit", 1800))
    for ref in refs.values():
        evidence = extract_ref_evidence(
            ref.get("arxiv_id"),
            cfg,
            max_tex_chars=max_tex_chars,
            max_snippets=max_snippets,
            snippet_char_limit=snippet_limit,
        )
        ref_evidence.append({
            **ref,
            "abstract_preview": short_text(ref.get("abstract"), 600),
            **evidence,
        })

    write_jsonl(output_dir / "papers.jsonl", papers)
    write_jsonl(output_dir / "ref_evidence.jsonl", ref_evidence)

    summary = {
        "paper_count": len(papers),
        "unique_ref_count": len(ref_evidence),
        "papers_by_split": dict(Counter(p["split"] for p in papers)),
        "papers_by_route": dict(Counter(p["training_route"] for p in papers)),
        "papers_by_type": dict(Counter(p["paper_type"] for p in papers)),
        "ref_evidence_status": dict(Counter(r["evidence_status"] for r in ref_evidence)),
        "ref_tex_status": dict(Counter(r["tex_status"] for r in ref_evidence)),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
