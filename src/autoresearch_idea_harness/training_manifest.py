from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from .io import iter_jsonl, load_dataset_records, short_text, stable_id, write_json, write_jsonl


TARGET_SCHEMA_VERSION = "proposal_xml_v3_0"
TARGET_XML_TAGS = [
    "title",
    "problem",
    "gap",
    "core_idea",
    "implementation_plan",
    "algorithm_or_system",
    "training_or_data_recipe",
    "evaluation_plan",
    "expected_results",
    "risks_and_limitations",
]

SYSTEM_PROMPT = (
    "You are an autoresearch idea proposal model. Given a preprocessed "
    "with_research_question task packet, write one concrete, implementable "
    "research proposal. The proposal should be specific enough for a worker to "
    "code and evaluate, but it should remain one idea rather than a full paper."
)

PROPOSAL_FORMAT = """\
Output one proposal using this XML schema:
<proposal>
<title>...</title>
<problem>...</problem>
<gap>...</gap>
<core_idea>...</core_idea>
<implementation_plan>...</implementation_plan>
<algorithm_or_system>...</algorithm_or_system>
<training_or_data_recipe>...</training_or_data_recipe>
<evaluation_plan>...</evaluation_plan>
<expected_results>...</expected_results>
<risks_and_limitations>...</risks_and_limitations>
</proposal>

The expected granularity is one worker-implementable research idea: concrete
mechanism, code-level recipe, evaluation metric, and likely failure modes. Do
not write a broad survey direction, a full paper, or a trivial baseline rename.
"""

ARXIV_MONTH_RE = re.compile(r"^(\d{2})(\d{2})\.")


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except Exception:
        return None


def _parse_arxiv_month(arxiv_id: str | None) -> tuple[int, int] | None:
    if not arxiv_id:
        return None
    m = ARXIV_MONTH_RE.match(arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    year = 2000 + yy if yy < 80 else 1900 + yy
    return year, int(m.group(2))


def _ref_key(ref: dict[str, Any]) -> str:
    aid = ref.get("arxiv_id")
    if aid:
        return f"arxiv:{aid}"
    return stable_id("ref", ref.get("title", ""), ref.get("year"), length=16)


def _load_key_value_cache(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not path.exists():
        return values
    for row in iter_jsonl(path):
        key = row.get("key")
        if key is None:
            continue
        value = row.get("value")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    value = json.loads(stripped)
                except Exception:
                    value = row.get("value")
        values[str(key)] = value
    return values


def _load_prompt_caches(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    cache_dir = dataset_dir / "prompt_cache"
    return {
        "research_question": _load_key_value_cache(cache_dir / "research_question.jsonl"),
        "top_k_indices": _load_key_value_cache(cache_dir / "top_k_5_index.jsonl"),
        "top_k_related_work": _load_key_value_cache(cache_dir / "top_k_5_related_work.jsonl"),
    }


def _load_target_cache(paths: list[Path]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            aid = row.get("arxiv_id")
            if not aid:
                continue
            text = row.get("target_impl_proposal") or row.get("target_proposal")
            if not text:
                continue
            cache[str(aid)] = {
                "target_text": text,
                "cot_text": row.get("cot_impl_proposal") or row.get("cot_proposal"),
                "target_source": row.get("target_source") or path.name,
                "quality": row.get("target_quality") or {},
                "cache_path": str(path),
            }
    return cache


def _ref_date_bucket(ref: dict[str, Any], target_date: date) -> str:
    ref_month = _parse_arxiv_month(ref.get("arxiv_id"))
    if ref_month is not None:
        ry, rm = ref_month
        if ry > target_date.year or (ry == target_date.year and rm > target_date.month):
            return "future"
        if ry == target_date.year and rm == target_date.month:
            return "same_month"
        return "past"
    try:
        year = int(ref.get("year") or 0)
    except Exception:
        year = 0
    if year <= 0:
        return "unknown"
    if year > target_date.year:
        return "future"
    if year == target_date.year:
        return "same_year_unknown_month"
    return "past"


def _filter_refs_for_leakage(refs: list[dict[str, Any]], target_date: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    dropped: list[dict[str, Any]] = []
    for ref in refs:
        bucket = _ref_date_bucket(ref, target_date)
        counts[bucket] += 1
        if bucket == "future":
            dropped.append({
                "ref_key": _ref_key(ref),
                "arxiv_id": ref.get("arxiv_id"),
                "title": ref.get("title"),
                "year": ref.get("year"),
                "reason": "future_reference",
            })
            continue
        kept.append(ref)
    return kept, {
        "input_ref_count": len(refs),
        "kept_ref_count": len(kept),
        "dropped_future_ref_count": len(dropped),
        "unknown_ref_date_count": counts.get("unknown", 0),
        "same_month_ref_count": counts.get("same_month", 0),
        "same_year_unknown_month_count": counts.get("same_year_unknown_month", 0),
        "dropped_future_refs_preview": dropped[:5],
    }


def _quality_score(row: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    detail = row.get("detail_support_level")
    if detail == "tex_rich":
        score += 1.5
        reasons.append("tex_rich")
    elif detail == "tex_good":
        score += 1.0
        reasons.append("tex_good")

    tex = row.get("tex_features") or {}
    for key, weight in [
        ("has_method", 1.0),
        ("has_implementation", 1.0),
        ("has_evaluation", 1.0),
        ("has_results", 0.75),
    ]:
        if tex.get(key):
            score += weight
            reasons.append(key)

    selected_chars = int(tex.get("selected_tex_chars") or 0)
    if selected_chars >= 16000:
        score += 0.5
        reasons.append("tex_chars>=16k")
    elif selected_chars >= 8000:
        score += 0.25
        reasons.append("tex_chars>=8k")

    kind_chars = tex.get("section_kind_chars") or {}
    for key, threshold, weight in [
        ("method", 1000, 0.75),
        ("implementation", 1000, 0.75),
        ("evaluation", 1000, 0.5),
        ("results", 500, 0.5),
    ]:
        if int(kind_chars.get(key) or 0) >= threshold:
            score += weight
            reasons.append(f"{key}_chars>={threshold}")

    ref_count = int(row.get("ref_count") or 0)
    if ref_count >= 10:
        score += 0.5
        reasons.append("ref_count>=10")
    elif ref_count >= 5:
        score += 0.25
        reasons.append("ref_count>=5")

    if row.get("category_family") in {"ml", "vision", "ai"}:
        score += 0.5
        reasons.append("mls_relevant_family")

    signals = row.get("classification_signals") or {}
    if int(signals.get("method_algorithm") or 0) >= 10:
        score += 0.25
        reasons.append("strong_method_signal")

    return round(score, 3), reasons


def _required_tex_features_ok(row: dict[str, Any], required: list[str]) -> bool:
    tex = row.get("tex_features") or {}
    return all(bool(tex.get(key)) for key in required)


def _trim_ref(ref: dict[str, Any], abstract_limit: int) -> dict[str, Any]:
    return {
        "ref_key": _ref_key(ref),
        "arxiv_id": ref.get("arxiv_id"),
        "title": ref.get("title"),
        "year": ref.get("year"),
        "abstract": short_text(ref.get("abstract"), abstract_limit),
        "reference_tex_evidence_status": "pending",
    }


def _select_refs(
    arxiv_id: str,
    refs: list[dict[str, Any]],
    prompt_caches: dict[str, dict[str, Any]],
    top_k: int,
    abstract_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_indices = prompt_caches["top_k_indices"].get(arxiv_id)
    selected: list[dict[str, Any]] = []
    source = "first_k_after_leakage_filter"
    if isinstance(raw_indices, list):
        source = "cached_top_k_indices"
        for idx in raw_indices:
            try:
                ref = refs[int(idx)]
            except Exception:
                continue
            selected.append(ref)
    if not selected:
        selected = refs[:top_k]
    selected = selected[:top_k]
    return [_trim_ref(ref, abstract_limit) for ref in selected], {
        "top_k_source": source,
        "cached_top_k_indices": raw_indices if isinstance(raw_indices, list) else None,
    }


def _build_condition_prompt(selected_refs: list[dict[str, Any]], research_question: str | None) -> str:
    ref_block = "\n\n".join(
        (
            f"[{idx}] {ref.get('title') or 'Unknown'} ({ref.get('year') or 'n.d.'})\n"
            f"Abstract: {ref.get('abstract') or '(no abstract available)'}"
        )
        for idx, ref in enumerate(selected_refs, start=1)
    )
    question = research_question or "(research question cache missing; synthesize one from the references before training)"
    return (
        f"Below are {len(selected_refs)} papers from a researcher's reading list. "
        "Based on these references, propose a novel research direction.\n\n"
        f"{ref_block}\n\n"
        "The researcher has identified the following open question as their primary motivation:\n"
        f"\"{question}\"\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _target_payload(arxiv_id: str, target_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cached = target_cache.get(arxiv_id)
    payload = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "canonical_xml_tags": TARGET_XML_TAGS,
        "format_policy": (
            "Keep raw model output for training/debugging; parse or normalize to the "
            "canonical XML schema for dashboard/eval. Proposal modules do not need "
            "to be hard-format-trained before they can be adapted."
        ),
        "granularity": {
            "unit": "one implementable research idea",
            "expected": [
                "mechanism or algorithmic change",
                "code-level implementation recipe",
                "training/data recipe when relevant",
                "evaluation plan with metric and baseline",
                "risks and likely failure modes",
            ],
            "too_coarse": "broad research agenda or survey paragraph",
            "too_fine": "single hyperparameter tweak without a nontrivial mechanism",
        },
    }
    if not cached:
        return {
            **payload,
            "target_status": "missing_synthesis",
            "target_text": None,
            "cot_text": None,
        }
    return {
        **payload,
        "target_status": "ready",
        "target_text": cached.get("target_text"),
        "cot_text": cached.get("cot_text"),
        "target_source": cached.get("target_source"),
        "target_quality": cached.get("quality"),
        "cache_path": cached.get("cache_path"),
    }


def _split_for_index(i: int, n: int, train_ratio: float, val_ratio: float) -> str:
    train_cut = int(n * train_ratio)
    val_cut = int(n * (train_ratio + val_ratio))
    if i < train_cut:
        return "train"
    if i < val_cut:
        return "val"
    return "test"


def build_v3_training_manifest(
    cfg: dict[str, Any],
    base_model_release_date: str,
    base_model_metadata: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    limit: int | None = None,
    target_cache_paths: list[Path] | None = None,
    min_quality_score: float | None = None,
    through_date: str | None = None,
) -> dict[str, Any]:
    tm_cfg = cfg.get("training_manifest", {})
    output_dir = output_dir or Path(cfg["runs_dir"]) / "training_data" / str(tm_cfg.get("version", "v3_0_manifest"))

    base_date = _parse_date(base_model_release_date)
    if base_date is None:
        raise ValueError("--base-model-release-date must be YYYY-MM-DD")
    max_date = _parse_date(through_date) if through_date else None

    dataset_records = load_dataset_records(Path(cfg["dataset_dir"]), ["train", "val", "test"])
    prompt_caches = _load_prompt_caches(Path(cfg["dataset_dir"]))
    target_cache = _load_target_cache(target_cache_paths or [Path(cfg["dataset_dir"]) / "demo_tex_impl_targets.jsonl"])

    allowed_types = set(tm_cfg.get("paper_types", ["method_algorithm"]))
    allowed_routes = set(tm_cfg.get("training_routes", ["tex_target_with_ref_detail"]))
    allowed_details = set(tm_cfg.get("detail_support_levels", ["tex_good", "tex_rich"]))
    allowed_families = set(tm_cfg.get("category_families", ["ml", "vision", "ai"]))
    allowed_categories = set(tm_cfg.get("categories", ["cs.LG", "cs.CV", "cs.AI", "stat.ML", "eess.IV"]))
    required_tex = list(tm_cfg.get("required_tex_features", ["has_method", "has_implementation", "has_evaluation"]))
    min_refs = int(tm_cfg.get("min_refs", 3))
    top_k = int(tm_cfg.get("top_k_refs", 5))
    abstract_limit = int(tm_cfg.get("reference_abstract_chars", 700))
    require_research_question = bool(tm_cfg.get("require_research_question", True))
    train_ratio = float(tm_cfg.get("train_ratio", 0.8))
    val_ratio = float(tm_cfg.get("val_ratio", 0.1))
    min_quality = float(min_quality_score if min_quality_score is not None else tm_cfg.get("min_quality_score", 4.0))

    skipped: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []

    for row in iter_jsonl(cfg["classified_papers"]):
        arxiv_id = row.get("arxiv_id")
        if not arxiv_id:
            skipped["missing_arxiv_id"] += 1
            continue
        target_date = _parse_date(row.get("created"))
        if target_date is None:
            skipped["missing_created_date"] += 1
            continue
        if target_date < base_date:
            skipped["before_base_model_release"] += 1
            continue
        if max_date is not None and target_date > max_date:
            skipped["after_through_date"] += 1
            continue
        if row.get("tex_status") != "ok":
            skipped["tex_not_ok"] += 1
            continue
        if row.get("paper_type") not in allowed_types:
            skipped["paper_type"] += 1
            continue
        if row.get("training_route") not in allowed_routes:
            skipped["training_route"] += 1
            continue
        if row.get("detail_support_level") not in allowed_details:
            skipped["detail_support_level"] += 1
            continue
        family = row.get("category_family")
        cats = set(row.get("categories") or [])
        if family not in allowed_families and not (cats & allowed_categories):
            skipped["category_not_mls_relevant"] += 1
            continue
        if not _required_tex_features_ok(row, required_tex):
            skipped["missing_required_tex_features"] += 1
            continue
        raw = dataset_records.get(arxiv_id)
        if raw is None:
            skipped["missing_raw_dataset_record"] += 1
            continue
        refs, ref_guard = _filter_refs_for_leakage(list(raw.get("refs") or []), target_date)
        if len(refs) < min_refs:
            skipped["not_enough_refs_after_leakage_filter"] += 1
            continue
        quality_score, quality_reasons = _quality_score(row)
        if quality_score < min_quality:
            skipped["quality_score_below_min"] += 1
            continue

        selected_refs, ref_selection = _select_refs(arxiv_id, refs, prompt_caches, top_k, abstract_limit)
        if len(selected_refs) < min_refs:
            skipped["not_enough_selected_refs"] += 1
            continue
        research_question = prompt_caches["research_question"].get(arxiv_id)
        if require_research_question and not (isinstance(research_question, str) and research_question.strip()):
            skipped["missing_research_question"] += 1
            continue
        target = _target_payload(arxiv_id, target_cache)
        condition_prompt = _build_condition_prompt(selected_refs, research_question if isinstance(research_question, str) else None)
        sample = {
            "sample_id": stable_id("v3s", arxiv_id, row.get("created"), row.get("paper_type"), length=14),
            "manifest_version": tm_cfg.get("version", "v3_0_manifest"),
            "arxiv_id": arxiv_id,
            "created": row.get("created"),
            "arxiv_month": row.get("arxiv_month"),
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "categories": row.get("categories") or [],
            "primary_category": row.get("primary_category"),
            "category_family": family,
            "paper_type": row.get("paper_type"),
            "training_route": row.get("training_route"),
            "detail_support_level": row.get("detail_support_level"),
            "tex_features": row.get("tex_features") or {},
            "quality": {
                "score": quality_score,
                "reasons": quality_reasons,
                "min_required_score": min_quality,
            },
            "leakage_guard": {
                "base_model_release_date": base_model_release_date,
                "target_created": row.get("created"),
                "eligible_after_base_release": True,
                "condition_uses_future_refs": False,
                "reference_filter": ref_guard,
                "ordering_policy": "sort by target created date ascending before training",
            },
            "condition_packet": {
                "strategy": "with_research_question",
                "research_question": research_question,
                "research_question_status": "cached" if isinstance(research_question, str) and research_question.strip() else "missing",
                "selected_refs": selected_refs,
                "related_work_cache": prompt_caches["top_k_related_work"].get(arxiv_id),
                "ref_selection": ref_selection,
                "reference_detail_policy": (
                    "This manifest starts from metadata/abstract refs. A later ref-evidence "
                    "stage should attach TeX snippets when available before final collate."
                ),
                "prompt": condition_prompt,
            },
            "target": target,
        }
        if target.get("target_status") == "ready":
            sample["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": condition_prompt},
                {"role": "assistant", "content": target.get("target_text") or ""},
            ]
        candidates.append(sample)

    candidates.sort(key=lambda x: (x.get("created") or "", x.get("arxiv_id") or ""))
    if limit is not None:
        candidates = candidates[:limit]

    n = len(candidates)
    for i, sample in enumerate(candidates):
        split = _split_for_index(i, n, train_ratio, val_ratio)
        sample["chronological_rank"] = i
        sample["split"] = split

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    ready_rows: list[dict[str, Any]] = []
    synthesis_queue: list[dict[str, Any]] = []
    ref_queue: dict[str, dict[str, Any]] = {}
    for sample in candidates:
        by_split[sample["split"]].append(sample)
        for ref in sample["condition_packet"]["selected_refs"]:
            ref_queue.setdefault(ref["ref_key"], ref)
        if sample["target"]["target_status"] == "ready":
            ready_rows.append(sample)
        else:
            synthesis_queue.append({
                "sample_id": sample["sample_id"],
                "arxiv_id": sample["arxiv_id"],
                "created": sample["created"],
                "split": sample["split"],
                "paper_type": sample["paper_type"],
                "category_family": sample["category_family"],
                "quality_score": sample["quality"]["score"],
                "target_schema_version": sample["target"]["schema_version"],
                "condition_strategy": sample["condition_packet"]["strategy"],
                "tex_features": sample["tex_features"],
            })

    write_jsonl(output_dir / "samples.jsonl", candidates)
    write_jsonl(output_dir / "ready_sft.jsonl", ready_rows)
    write_jsonl(output_dir / "synthesis_queue.jsonl", synthesis_queue)
    write_jsonl(output_dir / "ref_evidence_queue.jsonl", ref_queue.values())
    for split, rows in by_split.items():
        write_jsonl(output_dir / "splits" / f"{split}.jsonl", rows)

    summary = {
        "manifest_version": tm_cfg.get("version", "v3_0_manifest"),
        "base_model_release_date": base_model_release_date,
        "base_model_metadata": base_model_metadata or {},
        "through_date": through_date,
        "candidate_count": len(candidates),
        "ready_sft_count": len(ready_rows),
        "synthesis_queue_count": len(synthesis_queue),
        "unique_selected_ref_count": len(ref_queue),
        "output_dir": str(output_dir),
        "filters": {
            "paper_types": sorted(allowed_types),
            "training_routes": sorted(allowed_routes),
            "detail_support_levels": sorted(allowed_details),
            "category_families": sorted(allowed_families),
            "categories": sorted(allowed_categories),
            "required_tex_features": required_tex,
            "min_refs": min_refs,
            "require_research_question": require_research_question,
            "min_quality_score": min_quality,
        },
        "splits": {split: len(rows) for split, rows in by_split.items()},
        "created_range": {
            "first": candidates[0]["created"] if candidates else None,
            "last": candidates[-1]["created"] if candidates else None,
        },
        "by_created_month": dict(Counter((s.get("created") or "")[:7] for s in candidates)),
        "by_arxiv_month": dict(Counter(s.get("arxiv_month") for s in candidates)),
        "by_category_family": dict(Counter(s.get("category_family") for s in candidates)),
        "by_primary_category": dict(Counter(s.get("primary_category") for s in candidates)),
        "by_detail_support_level": dict(Counter(s.get("detail_support_level") for s in candidates)),
        "target_status": dict(Counter(s["target"]["target_status"] for s in candidates)),
        "research_question_status": dict(Counter(s["condition_packet"]["research_question_status"] for s in candidates)),
        "skipped": dict(skipped),
        "target_schema": {
            "schema_version": TARGET_SCHEMA_VERSION,
            "canonical_xml_tags": TARGET_XML_TAGS,
            "raw_plus_canonical_policy": "train on raw/canonical target text; parse canonical XML for eval/dashboard.",
        },
    }
    write_json(output_dir / "manifest.json", summary)
    write_json(output_dir / "target_schema.json", summary["target_schema"])
    return summary


def _load_samples(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _sample_with_target(sample: dict[str, Any], cached: dict[str, Any], include_cot: bool) -> dict[str, Any]:
    target_text = cached.get("cot_text") if include_cot else cached.get("target_text")
    if not target_text:
        target_text = cached.get("target_text") or cached.get("cot_text") or ""
    prompt = sample.get("condition_packet", {}).get("prompt", "")
    return {
        "sample_id": sample.get("sample_id"),
        "arxiv_id": sample.get("arxiv_id"),
        "created": sample.get("created"),
        "chronological_rank": sample.get("chronological_rank"),
        "split": sample.get("split"),
        "title": sample.get("title"),
        "categories": sample.get("categories") or [],
        "category_family": sample.get("category_family"),
        "paper_type": sample.get("paper_type"),
        "training_route": sample.get("training_route"),
        "quality": sample.get("quality") or {},
        "leakage_guard": sample.get("leakage_guard") or {},
        "condition_strategy": sample.get("condition_packet", {}).get("strategy", "with_research_question"),
        "target_schema_version": sample.get("target", {}).get("schema_version", TARGET_SCHEMA_VERSION),
        "target_cache": {
            "target_source": cached.get("target_source"),
            "target_quality": cached.get("quality") or {},
            "cache_path": cached.get("cache_path"),
            "include_cot": include_cot,
        },
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "target": target_text,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_text},
        ],
    }


def collate_v3_sft_dataset(
    manifest_dir: Path,
    target_cache_paths: list[Path],
    output_dir: Path | None = None,
    include_cot: bool = False,
    min_quality_score: float | None = None,
) -> dict[str, Any]:
    """Create message-format SFT JSONL from a V3 manifest and target cache."""
    output_dir = output_dir or manifest_dir / "sft"
    samples = _load_samples(manifest_dir / "samples.jsonl")
    target_cache = _load_target_cache(target_cache_paths)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    missing_target: list[dict[str, Any]] = []
    skipped_quality = 0

    for sample in samples:
        if min_quality_score is not None:
            score = float((sample.get("quality") or {}).get("score") or 0.0)
            if score < min_quality_score:
                skipped_quality += 1
                continue
        aid = sample.get("arxiv_id")
        cached = target_cache.get(str(aid))
        if not cached:
            missing_target.append({
                "sample_id": sample.get("sample_id"),
                "arxiv_id": aid,
                "split": sample.get("split"),
                "created": sample.get("created"),
                "quality_score": (sample.get("quality") or {}).get("score"),
            })
            continue
        row = _sample_with_target(sample, cached, include_cot=include_cot)
        by_split.setdefault(row["split"] or "unknown", []).append(row)

    all_rows = [row for split in ("train", "val", "test") for row in by_split.get(split, [])]
    write_jsonl(output_dir / "all.jsonl", all_rows)
    for split, rows in by_split.items():
        if split == "unknown":
            continue
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "missing_target.jsonl", missing_target)
    summary = {
        "manifest_dir": str(manifest_dir),
        "output_dir": str(output_dir),
        "target_cache_paths": [str(p) for p in target_cache_paths],
        "sample_count": len(samples),
        "collated_count": len(all_rows),
        "missing_target_count": len(missing_target),
        "skipped_quality_count": skipped_quality,
        "include_cot": include_cot,
        "splits": {split: len(rows) for split, rows in by_split.items() if split != "unknown"},
        "created_range": {
            "first": all_rows[0]["created"] if all_rows else None,
            "last": all_rows[-1]["created"] if all_rows else None,
        },
        "target_schema": {
            "schema_version": TARGET_SCHEMA_VERSION,
            "canonical_xml_tags": TARGET_XML_TAGS,
            "raw_plus_canonical_policy": "train target text as cached; parse canonical XML for eval/dashboard.",
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary
