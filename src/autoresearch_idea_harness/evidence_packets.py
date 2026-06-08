from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, short_text, stable_id, write_json, write_jsonl


WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")


def _words(text: str | None) -> set[str]:
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "using", "based",
        "into", "are", "was", "were", "have", "has", "paper", "model", "data",
    }
    return {w.lower() for w in WORD_RE.findall(text or "") if w.lower() not in stop}


def _ref_score(target: dict[str, Any], ref: dict[str, Any], evidence: dict[str, Any] | None) -> float:
    target_words = _words(target.get("title", "") + " " + target.get("abstract", ""))
    ref_words = _words(ref.get("title", "") + " " + (ref.get("abstract") or ""))
    overlap = len(target_words & ref_words) / max(1, len(target_words | ref_words))
    tex_bonus = 1.0 if evidence and evidence.get("evidence_status") == "tex_ok" else 0.0
    abstract_bonus = 0.2 if ref.get("abstract") else 0.0
    recency_bonus = 0.0
    try:
        year = int(ref.get("year") or 0)
        recency_bonus = max(0.0, min(0.2, (year - 2018) * 0.03))
    except Exception:
        pass
    return tex_bonus + abstract_bonus + recency_bonus + overlap


def _baseline_for(paper_type: str) -> dict[str, str]:
    if paper_type == "system_tooling":
        return {
            "baseline_id": "system_existing_methods",
            "metric": "measurable system or task improvement",
            "description": (
                "A competent implementation should improve over the strongest relevant "
                "systems, tools, or pipelines described in the evidence packet."
            ),
        }
    return {
        "baseline_id": "method_existing_methods",
        "metric": "measurable benchmark improvement",
        "description": (
            "A competent implementation should improve over the strongest relevant "
            "method or baseline described in the evidence packet."
        ),
    }


def _evidence_ref(ref: dict[str, Any], evidence: dict[str, Any] | None, max_snippets: int) -> dict[str, Any]:
    evidence = evidence or {}
    snippets = list(evidence.get("snippets") or [])[:max_snippets]
    return {
        "ref_key": ref.get("ref_key"),
        "arxiv_id": ref.get("arxiv_id"),
        "title": ref.get("title"),
        "year": ref.get("year"),
        "abstract": short_text(ref.get("abstract"), 1200),
        "evidence_status": evidence.get("evidence_status", "not_indexed"),
        "tex_status": evidence.get("tex_status", "not_indexed"),
        "snippets": snippets,
    }


def _packet_for(
    paper: dict[str, Any],
    ref_evidence: dict[str, dict[str, Any]],
    top_k: int,
    max_ref_snippets_per_packet: int,
) -> dict[str, Any] | None:
    refs = list(paper.get("refs") or [])
    if not refs:
        return None

    scored = sorted(
        refs,
        key=lambda r: _ref_score(paper, r, ref_evidence.get(r.get("ref_key"))),
        reverse=True,
    )
    selected = scored[:top_k]
    if not selected:
        return None

    per_ref_snippets = max(1, max_ref_snippets_per_packet // max(1, len(selected)))
    evidence_refs = [
        _evidence_ref(ref, ref_evidence.get(ref.get("ref_key")), per_ref_snippets)
        for ref in selected
    ]
    snippet_count = sum(len(r.get("snippets") or []) for r in evidence_refs)

    target_private = {
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "categories": paper.get("categories") or [],
        "tex_features": paper.get("tex_features") or {},
    }
    packet_id = stable_id("pkt", paper.get("arxiv_id"), [r.get("ref_key") for r in evidence_refs])
    return {
        "packet_id": packet_id,
        "split": paper.get("split"),
        "version": "v1",
        "route": paper.get("training_route"),
        "paper_type": paper.get("paper_type"),
        "category_family": paper.get("category_family"),
        "detail_support_level": paper.get("detail_support_level"),
        "recommended_condition": paper.get("recommended_condition"),
        "baseline": _baseline_for(str(paper.get("paper_type"))),
        "success_definition": (
            "Expert forecast probability that this proposal, if implemented by a "
            "competent worker, would exceed the stated baseline or produce a clear "
            "measurable improvement."
        ),
        "expert_view": {
            "task": (
                "Based only on the evidence below, propose a novel, implementable "
                "research idea. The proposal should be specific enough to evaluate."
            ),
            "domain": {
                "paper_type": paper.get("paper_type"),
                "category_family": paper.get("category_family"),
                "categories": paper.get("categories") or [],
            },
            "baseline": _baseline_for(str(paper.get("paper_type"))),
            "references": evidence_refs,
        },
        "private_target": target_private,
        "diagnostics": {
            "selected_ref_count": len(evidence_refs),
            "selected_ref_with_tex": sum(1 for r in evidence_refs if r.get("evidence_status") == "tex_ok"),
            "snippet_count": snippet_count,
        },
    }


def build_evidence_packets(
    cfg: dict[str, Any],
    limit: int | None = None,
    paper_bank_dir: Path | None = None,
    output_dir: Path | None = None,
    allow_metadata_only_refs: bool = False,
) -> dict[str, Any]:
    ep_cfg = cfg.get("evidence_packets", {})
    paper_bank_dir = paper_bank_dir or Path(cfg["runs_dir"]) / "paper_bank"
    output_dir = output_dir or Path(cfg["runs_dir"]) / "evidence_packets" / str(ep_cfg.get("version", "v1"))

    route = ep_cfg.get("route", "tex_target_with_ref_detail")
    allowed_types = set(ep_cfg.get("paper_types", ["method_algorithm", "system_tooling"]))
    require_ref_tex = bool(ep_cfg.get("require_ref_tex", True)) and not allow_metadata_only_refs
    min_ref_tex_snippets = int(ep_cfg.get("min_ref_tex_snippets", 1))
    ref_evidence = {
        row.get("ref_key"): row
        for row in iter_jsonl(paper_bank_dir / "ref_evidence.jsonl")
        if row.get("ref_key")
    }

    packets = []
    skipped_no_ref_tex = 0
    for paper in iter_jsonl(paper_bank_dir / "papers.jsonl"):
        if paper.get("training_route") != route:
            continue
        if paper.get("paper_type") not in allowed_types:
            continue
        pkt = _packet_for(
            paper,
            ref_evidence=ref_evidence,
            top_k=int(ep_cfg.get("top_k_refs", 5)),
            max_ref_snippets_per_packet=int(ep_cfg.get("max_ref_snippets_per_packet", 12)),
        )
        if pkt is None:
            continue
        if pkt["diagnostics"]["selected_ref_count"] < int(ep_cfg.get("min_refs", 1)):
            continue
        if require_ref_tex and pkt["diagnostics"]["snippet_count"] < min_ref_tex_snippets:
            skipped_no_ref_tex += 1
            continue
        packets.append(pkt)
        if limit is not None and len(packets) >= limit:
            break

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for pkt in packets:
        by_split.setdefault(pkt.get("split") or "unknown", []).append(pkt)
    for split, rows in by_split.items():
        if split == "unknown":
            continue
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "all.jsonl", packets)

    summary = {
        "packet_count": len(packets),
        "by_split": dict(Counter(p.get("split") for p in packets)),
        "by_paper_type": dict(Counter(p.get("paper_type") for p in packets)),
        "with_ref_tex": sum(1 for p in packets if p["diagnostics"]["selected_ref_with_tex"] > 0),
        "total_snippets": sum(p["diagnostics"]["snippet_count"] for p in packets),
        "require_ref_tex": require_ref_tex,
        "skipped_no_ref_tex": skipped_no_ref_tex,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
