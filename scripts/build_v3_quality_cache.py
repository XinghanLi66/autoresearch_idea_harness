#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, load_config, load_dataset_records, stable_id, write_json  # noqa: E402
from autoresearch_idea_harness.prompt_properties import (  # noqa: E402
    approx_token_count,
    load_prompt_property_caches,
    trim_to_tokens,
)
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402
from autoresearch_idea_harness.training_manifest import PROPOSAL_FORMAT, _filter_refs_for_leakage, _parse_date  # noqa: E402


QUESTION_PROMPT = """\
You are preparing a high-quality training condition for an autoresearch idea proposal model.

Given the reference papers below, write one open research question plus its motivation.
The output must be 120-200 words, self-contained, specific, and forward-looking.
It must not mention the target paper, claim the proposed method already exists, or include markdown.
It should name the concrete technical tension across the references and make clear what kind of
implementable research idea would answer the question.

Output only the final text.

Reference papers:
{ref_block}
"""

ABSTRACT_PROMPT = """\
Rewrite the abstract below into a dense, complete summary of at most 200 words.
Keep the problem, method, data/task, and key result if present.
Do not use ellipses. Do not leave the sentence unfinished. Output only the summary.

Title: {title}

Abstract:
{abstract}
"""

RELATED_WORK_PROMPT = """\
Synthesize the reference papers below into a concise related-work context for a research idea proposal model.
Write 220-360 words. Emphasize concrete methods, limitations, and the gap that a new implementable idea could address.
Do not mention the target paper. Do not use markdown. Output only the narrative.

Reference papers:
{ref_block}
"""

TEX_SNIPPET_PROMPT = """\
You are cleaning TeX-derived evidence for a training cache.

Target paper: {title}
Target abstract: {abstract}
Evidence kind: {kind}

From the candidate excerpts below, select only material that directly describes the target paper's
own {kind}. Discard related-work, baseline-only, citation-list, and malformed formula-only text.
Rewrite into 1-3 clean, human-readable snippets. Preserve concrete technical details, datasets,
metrics, implementation choices, and results when present. Remove LaTeX commands and repair mangled
symbols into readable prose. If no useful evidence is present, output an empty JSON array.

Return only valid JSON:
[
  {{"heading": "...", "text": "..."}}
]

Candidate excerpts:
{candidate_block}
"""


PROMPT_STRATEGIES = [
    "full_refs",
    "top_k_refs",
    "related_work",
    "top_k_related_work",
    "with_research_question",
]

TEX_KINDS = [
    "abstract",
    "problem",
    "method",
    "implementation",
    "algorithm_or_system",
    "training_or_data_recipe",
    "evaluation",
    "results",
    "risks_and_limitations",
]
RAW_TEX_EXTRA_KINDS = ["discussion"]


def _collapse(text: str | None) -> str:
    return " ".join((text or "").split())


def _jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_name(arxiv_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", arxiv_id)


def _load_done(index_path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not index_path.exists():
        return done
    for row in iter_jsonl(index_path):
        aid = row.get("arxiv_id")
        if aid:
            done[str(aid)] = row
    return done


def _load_compact_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    for row in iter_jsonl(path):
        key = row.get("cache_key")
        compact = row.get("compact")
        if key and isinstance(compact, dict) and compact.get("text"):
            cache[str(key)] = compact
    return cache


def _load_rows(path: Path, limit: int | None = None, arxiv_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        aid = str(row.get("arxiv_id") or "")
        if arxiv_ids and aid not in arxiv_ids:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _ref_key(ref: dict[str, Any]) -> str:
    aid = ref.get("arxiv_id")
    if aid:
        return f"arxiv:{aid}"
    return stable_id("ref", ref.get("title", ""), ref.get("year"), length=16)


def _clean_title(title: str | None) -> str:
    title = _collapse(title)
    title = re.sub(r"^\s*\d{4}\s*\.\s*", "", title)
    title = re.sub(r"^[\s.,;:]+", "", title)
    return title


def _find_local_metadata(arxiv_id: str | None, arxiv_root: Path) -> Path | None:
    if not arxiv_id:
        return None
    match = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if match:
        path = arxiv_root / ("20" + match.group(1)) / match.group(2) / arxiv_id / "metadata.json"
        if path.exists():
            return path
    # Slow fallback is only for missing abstracts, so it is acceptable during
    # cache building and avoids silently empty ref fields.
    for candidate in arxiv_root.glob(f"*/*/{arxiv_id}/metadata.json"):
        if candidate.exists():
            return candidate
    return None


def _fetch_s2_metadata(arxiv_id: str | None, title: str | None, *, allow_title_search: bool) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=30.0) as client:
            if arxiv_id:
                resp = client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}",
                    params={"fields": "title,year,externalIds,abstract"},
                )
                resp.raise_for_status()
                data = resp.json()
                ext = data.get("externalIds") or {}
                return {
                    "arxiv_id": ext.get("ArXiv") or arxiv_id,
                    "title": _clean_title(data.get("title") or title),
                    "year": data.get("year"),
                    "abstract": _collapse(data.get("abstract")),
                    "metadata_source": "semantic_scholar_arxiv",
                }
            if title and allow_title_search:
                resp = client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": title, "fields": "title,year,externalIds,abstract", "limit": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                paper = (data.get("data") or [{}])[0]
                ext = paper.get("externalIds") or {}
                return {
                    "arxiv_id": ext.get("ArXiv"),
                    "title": _clean_title(paper.get("title") or title),
                    "year": paper.get("year"),
                    "abstract": _collapse(paper.get("abstract")),
                    "metadata_source": "semantic_scholar_title_search",
                }
    except Exception as exc:
        return {"metadata_source": "semantic_scholar_error", "metadata_error": repr(exc)}
    return {"metadata_source": "missing"}


def _fetch_arxiv_metadata(arxiv_id: str | None) -> dict[str, Any]:
    if not arxiv_id:
        return {"metadata_source": "missing"}
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get("https://export.arxiv.org/api/query", params={"id_list": arxiv_id, "max_results": 1})
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return {"metadata_source": "arxiv_api_empty"}
        title = _collapse(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _collapse(entry.findtext("atom:summary", default="", namespaces=ns))
        published = _collapse(entry.findtext("atom:published", default="", namespaces=ns))
        return {
            "arxiv_id": arxiv_id,
            "title": _clean_title(title),
            "year": published[:4] if published else None,
            "abstract": abstract,
            "metadata_source": "arxiv_api",
        }
    except Exception as exc:
        return {"metadata_source": "arxiv_api_error", "metadata_error": repr(exc)}


def _hydrate_ref(ref: dict[str, Any], *, arxiv_root: Path, allow_title_search: bool) -> dict[str, Any]:
    hydrated = dict(ref)
    hydrated["title"] = _clean_title(hydrated.get("title"))
    hydrated["abstract"] = _collapse(hydrated.get("abstract"))
    hydrated["abstract_source"] = "input" if hydrated.get("abstract") else "missing"
    if hydrated.get("abstract"):
        return hydrated

    meta_path = _find_local_metadata(hydrated.get("arxiv_id"), arxiv_root)
    if meta_path:
        try:
            meta = json.loads(meta_path.read_text())
            abstract = _collapse(meta.get("abstract"))
            if abstract:
                hydrated.update({
                    "title": _clean_title(meta.get("title") or hydrated.get("title")),
                    "abstract": abstract,
                    "year": hydrated.get("year") or meta.get("year") or str(meta.get("created") or "")[:4],
                    "abstract_source": "local_arxiv_metadata",
                    "metadata_path": str(meta_path),
                })
                return hydrated
        except Exception as exc:
            hydrated["local_metadata_error"] = repr(exc)

    arxiv_meta = _fetch_arxiv_metadata(hydrated.get("arxiv_id"))
    if arxiv_meta.get("abstract"):
        hydrated.update({
            "arxiv_id": arxiv_meta.get("arxiv_id") or hydrated.get("arxiv_id"),
            "title": _clean_title(arxiv_meta.get("title") or hydrated.get("title")),
            "year": hydrated.get("year") or arxiv_meta.get("year"),
            "abstract": arxiv_meta.get("abstract"),
            "abstract_source": arxiv_meta.get("metadata_source"),
        })
        return hydrated
    if arxiv_meta.get("metadata_error"):
        hydrated["arxiv_api_error"] = arxiv_meta.get("metadata_error")

    s2 = _fetch_s2_metadata(hydrated.get("arxiv_id"), hydrated.get("title"), allow_title_search=allow_title_search)
    if s2.get("abstract"):
        hydrated.update({
            "arxiv_id": s2.get("arxiv_id") or hydrated.get("arxiv_id"),
            "title": _clean_title(s2.get("title") or hydrated.get("title")),
            "year": hydrated.get("year") or s2.get("year"),
            "abstract": s2.get("abstract"),
            "abstract_source": s2.get("metadata_source"),
        })
    else:
        hydrated["abstract_source"] = s2.get("metadata_source") or "missing_after_download"
        if s2.get("metadata_error"):
            hydrated["metadata_error"] = s2.get("metadata_error")
    return hydrated


def _abstract_cache_key(title: str | None, abstract: str | None) -> str:
    title = _collapse(title).lower()
    abstract = _collapse(abstract)
    digest = hashlib.sha1(f"{title}\n{abstract}".encode("utf-8")).hexdigest()[:20]
    return f"abstract:{digest}"


def _sentence_limited(text: str, max_words: int) -> str:
    text = _collapse(text)
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    kept: list[str] = []
    count = 0
    for sentence in sentences:
        sw = sentence.split()
        if not sw:
            continue
        if kept and count + len(sw) > max_words:
            break
        kept.append(sentence)
        count += len(sw)
        if count >= max_words:
            break
    candidate = _collapse(" ".join(kept))
    if candidate:
        return candidate
    candidate = " ".join(words[:max_words]).rstrip(" ,;:")
    if candidate and candidate[-1] not in ".!?。！？":
        candidate += "."
    return candidate


def _llm(
    client: RunwayClient,
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
    log_dir: Path,
    call_id: str,
) -> dict[str, Any]:
    started = time.time()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "prompt.txt").write_text(prompt)
    try:
        try:
            result = client.complete(
                endpoint=endpoint,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as exc:
            if temperature is None or "temperature" not in str(exc) or "deprecated" not in str(exc):
                raise
            result = client.complete(
                endpoint=endpoint,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=None,
                max_tokens=max_tokens,
                stream=False,
            )
        latency = time.time() - started
        text = _collapse(result.text)
        write_json(log_dir / "response.json", {
            "call_id": call_id,
            "model": result.model,
            "latency_s": round(latency, 3),
            "usage": result.usage,
            "finish_reason": result.raw_finish_reason,
            "text": text,
        })
        return {
            "ok": True,
            "text": text,
            "latency_s": latency,
            "usage": result.usage,
            "finish_reason": result.raw_finish_reason,
            "model": result.model,
        }
    except Exception as exc:
        latency = time.time() - started
        write_json(log_dir / "error.json", {
            "call_id": call_id,
            "latency_s": round(latency, 3),
            "error": repr(exc),
        })
        return {"ok": False, "text": "", "latency_s": latency, "error": repr(exc), "model": model}


def _quality_text(text: str, *, min_words: int = 1, max_words: int | None = None) -> dict[str, Any]:
    text = _collapse(text)
    words = text.split()
    final_char_ok = bool(re.search(r"""[.!?。！？)"'\]]$""", text))
    return {
        "chars": len(text),
        "words": len(words),
        "tokens": approx_token_count(text),
        "complete": bool(text) and final_char_ok and not text.endswith("...") and not text.endswith(",") and not text.endswith(";"),
        "final_char_ok": final_char_ok,
        "has_ellipsis": "..." in text or "…" in text,
        "min_words_ok": len(words) >= min_words,
        "max_words_ok": max_words is None or len(words) <= max_words,
    }


def _build_ref_block(refs: list[dict[str, Any]], *, use_compact: bool = True) -> str:
    blocks = []
    for idx, ref in enumerate(refs, start=1):
        abstract = ref.get("compact_abstract") if use_compact else ref.get("abstract")
        blocks.append(
            f"[{idx}] {ref.get('title') or 'Unknown'} ({ref.get('year') or 'n.d.'})\n"
            f"Abstract: {abstract or '(no abstract available)'}"
        )
    return "\n\n".join(blocks)


def _prompt_ref_block(refs: list[dict[str, Any]]) -> str:
    return _build_ref_block(refs, use_compact=True)


def _condition_prompt(refs: list[dict[str, Any]], question: str) -> str:
    return (
        f"Below are {len(refs)} papers from a researcher's reading list. "
        "Based on these references, propose a novel research direction.\n\n"
        f"{_prompt_ref_block(refs)}\n\n"
        "The researcher has identified the following open question as their primary motivation:\n"
        f"\"{question}\"\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _plain_refs_prompt(refs: list[dict[str, Any]]) -> str:
    return (
        f"Below are {len(refs)} papers from a researcher's reading list. "
        "Based on these references, propose a novel research direction.\n\n"
        f"{_prompt_ref_block(refs)}\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _related_prompt(narrative: str) -> str:
    return (
        "A researcher has been studying the following area of the literature:\n\n"
        f"{narrative}\n\n"
        "Based on this background, propose a novel research direction.\n\n"
        f"{PROPOSAL_FORMAT}"
    )


def _section_snippets(row: dict[str, Any], max_chars: int, per_kind: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in TEX_KINDS}
    for section in row.get("tex_sections") or []:
        kind = str(section.get("kind") or "other")
        if kind not in grouped:
            continue
        text = _collapse(section.get("text"))
        if not text:
            continue
        grouped[kind].append({
            "kind": kind,
            "heading": section.get("heading"),
            "source": section.get("source"),
            "chars_raw": len(text),
            "text": text[:max_chars].rstrip(),
        })
    return {k: v[:per_kind] for k, v in grouped.items()}


def _raw_section_candidates(row: dict[str, Any], max_chars: int, per_kind: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in [*TEX_KINDS, *RAW_TEX_EXTRA_KINDS]}
    for section in row.get("tex_sections") or []:
        kind = str(section.get("kind") or "other")
        if kind not in grouped:
            continue
        text = _collapse(section.get("text"))
        if not text:
            continue
        grouped[kind].append({
            "kind": kind,
            "heading": section.get("heading"),
            "source": section.get("source"),
            "chars_raw": len(text),
            "text": text[:max_chars].rstrip(),
        })
    return {k: v[:per_kind] for k, v in grouped.items()}


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else []
    except Exception:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _curate_tex_snippets(
    client: RunwayClient | None,
    *,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    raw: dict[str, list[dict[str, Any]]],
    log_root: Path,
    temperature: float | None,
) -> dict[str, list[dict[str, Any]]]:
    if client is None:
        return {k: raw.get(k, []) for k in TEX_KINDS}
    curated: dict[str, list[dict[str, Any]]] = {k: [] for k in TEX_KINDS}
    title = row.get("title") or ""
    abstract = _collapse(row.get("abstract"))[:1600]
    extra_candidates = {
        "method": ["abstract", "problem", "discussion"],
        "algorithm_or_system": ["abstract", "method", "implementation", "discussion"],
        "training_or_data_recipe": ["implementation", "method"],
        "risks_and_limitations": ["discussion", "problem"],
    }
    for kind in TEX_KINDS:
        candidates = list(raw.get(kind) or [])
        for extra_kind in extra_candidates.get(kind, []):
            candidates.extend(raw.get(extra_kind) or [])
        seen = set()
        deduped = []
        for candidate in candidates:
            key = (candidate.get("source"), candidate.get("heading"), candidate.get("text", "")[:80])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        candidates = deduped[:10]
        if not candidates:
            continue
        candidate_block = "\n\n".join(
            f"### {idx}. {c.get('heading') or '(no heading)'}\n"
            f"source: {c.get('source') or ''}\n"
            f"{c.get('text') or ''}"
            for idx, c in enumerate(candidates, start=1)
        )
        result = _llm(
            client,
            endpoint=endpoint,
            model=model,
            prompt=TEX_SNIPPET_PROMPT.format(
                title=title,
                abstract=abstract,
                kind=kind,
                candidate_block=candidate_block[:12000],
            ),
            max_tokens=1400,
            temperature=temperature,
            log_dir=log_root / kind,
            call_id=f"tex_{kind}",
        )
        snippets = []
        for item in _parse_json_array(result.get("text") or ""):
            text = _collapse(item.get("text"))
            if not text:
                continue
            snippets.append({
                "kind": kind,
                "heading": _collapse(item.get("heading")) or kind,
                "source": "llm_curated_tex",
                "text": text,
                "quality": _quality_text(text, min_words=20, max_words=260),
            })
        curated[kind] = snippets[:3]
    return curated


def _select_top_refs(
    refs: list[dict[str, Any]],
    top_k: int,
    cached_indices: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices: list[int] = []
    source = "dataset_order_first_k"
    if cached_indices:
        source = "legacy_cached_top_k_indices"
        for idx in cached_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 0 <= i < len(refs) and i not in indices:
                indices.append(i)
            if len(indices) >= top_k:
                break
    if not indices:
        indices = list(range(min(top_k, len(refs))))
    return [refs[i] for i in indices], {"source": source, "top_k": top_k, "indices": indices}


def _summarize_abstract(
    client: RunwayClient | None,
    *,
    endpoint: str,
    model: str,
    title: str,
    abstract: str,
    max_words: int,
    attempts: int,
    temperature: float | None,
    log_root: Path,
    force_api: bool,
) -> dict[str, Any]:
    abstract = _collapse(abstract)
    if not abstract:
        return {"text": "", "source": "missing", "quality": _quality_text("", max_words=max_words), "attempts": []}
    if client is None or not force_api:
        text = trim_to_tokens(abstract, max_words)
        return {
            "text": text,
            "source": "raw_trim_without_api" if approx_token_count(abstract) > max_words else "raw_under_limit_without_api",
            "quality": _quality_text(text, max_words=max_words),
            "attempts": [],
        }
    attempt_rows = []
    best = ""
    for i in range(attempts):
        prompt = ABSTRACT_PROMPT.format(title=title or "Unknown", abstract=abstract[:6000])
        result = _llm(
            client,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            max_tokens=700,
            temperature=temperature,
            log_dir=log_root / f"attempt_{i:02d}",
            call_id=f"abstract_{i:02d}",
        )
        text = result.get("text") or ""
        quality = _quality_text(text, min_words=20, max_words=max_words)
        attempt_rows.append({"attempt": i, "ok": result.get("ok"), "quality": quality, "error": result.get("error")})
        if text and (not best or quality["words"] > _quality_text(best)["words"]):
            best = text
        if quality["complete"] and not quality["has_ellipsis"] and quality["min_words_ok"] and quality["max_words_ok"]:
            return {"text": text, "source": "llm_rewrite", "quality": quality, "attempts": attempt_rows}
    if not best:
        best = _sentence_limited(abstract, max_words)
    if len(best.split()) > max_words:
        best = _sentence_limited(best, max_words)
    return {"text": best, "source": "llm_best_effort", "quality": _quality_text(best, max_words=max_words), "attempts": attempt_rows}


def _generate_question(
    client: RunwayClient | None,
    *,
    endpoint: str,
    model: str,
    refs: list[dict[str, Any]],
    min_words: int,
    max_words: int,
    attempts: int,
    temperature: float | None,
    log_root: Path,
) -> dict[str, Any]:
    ref_block = _build_ref_block(refs, use_compact=True)
    if client is None:
        text = (
            "How can the technical limitations shared across these references be turned into a concrete, "
            "worker-implementable research idea with a clear mechanism, measurable baseline, and realistic "
            "evaluation plan, while preserving the strongest empirical lessons from the prior methods?"
        )
        return {"text": text, "source": "mock_without_api", "quality": _quality_text(text, min_words=min_words, max_words=max_words), "attempts": []}
    attempt_rows = []
    best = ""
    best_distance = 10**9
    lenses = [
        "balance local convolutional priors, global context, data scale, and deployable evaluation",
        "identify the most worker-implementable algorithmic mechanism implied by the references",
        "emphasize measurable failure modes, baselines, and why current architectures remain incomplete",
        "focus on data-scarce medical segmentation and architecture adaptation without target-paper leakage",
        "connect computational efficiency, high-frequency detail, and long-range dependency modeling",
    ]
    for i in range(attempts):
        prompt = QUESTION_PROMPT.format(ref_block=ref_block)
        prompt += f"\n\nAttempt focus lens: {lenses[i % len(lenses)]}."
        result = _llm(
            client,
            endpoint=endpoint,
            model=model,
            prompt=prompt,
            max_tokens=700,
            temperature=temperature,
            log_dir=log_root / f"attempt_{i:02d}",
            call_id=f"question_{i:02d}",
        )
        text = result.get("text") or ""
        quality = _quality_text(text, min_words=min_words, max_words=max_words)
        words = quality["words"]
        distance = 0 if min_words <= words <= max_words else min(abs(words - min_words), abs(words - max_words))
        attempt_rows.append({"attempt": i, "ok": result.get("ok"), "quality": quality, "error": result.get("error")})
        if text and distance < best_distance and quality["complete"] and not quality["has_ellipsis"]:
            best = text
            best_distance = distance
        if quality["complete"] and not quality["has_ellipsis"] and quality["min_words_ok"] and quality["max_words_ok"]:
            return {"text": text, "source": "llm_sampled_in_range", "quality": quality, "attempts": attempt_rows}
    if best and approx_token_count(best) > max_words:
        best = trim_to_tokens(best, max_words)
    return {"text": best, "source": "llm_best_effort", "quality": _quality_text(best, min_words=min_words, max_words=max_words), "attempts": attempt_rows}


def _generate_related_work(
    client: RunwayClient | None,
    *,
    endpoint: str,
    model: str,
    refs: list[dict[str, Any]],
    log_root: Path,
    temperature: float | None,
) -> dict[str, Any]:
    ref_block = _build_ref_block(refs, use_compact=True)
    if client is None:
        text = " ".join(f"{r.get('title')} studies {r.get('compact_abstract')}" for r in refs)
        return {"text": trim_to_tokens(text, 320), "source": "mock_without_api", "quality": _quality_text(text, max_words=380)}
    result = _llm(
        client,
        endpoint=endpoint,
        model=model,
        prompt=RELATED_WORK_PROMPT.format(ref_block=ref_block),
        max_tokens=900,
        temperature=temperature,
        log_dir=log_root,
        call_id="related_work",
    )
    text = result.get("text") or ""
    return {"text": text, "source": "llm_rewrite" if result.get("ok") else "missing", "quality": _quality_text(text, min_words=80, max_words=420)}


def _article_quality(article: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    q = article.get("research_question") or {}
    q_quality = q.get("quality") or {}
    if not q_quality.get("min_words_ok") or not q_quality.get("max_words_ok"):
        errors.append("research_question_length")
    if q_quality.get("has_ellipsis") or not q_quality.get("complete"):
        errors.append("research_question_incomplete")
    for key in ("target_abstract",):
        quality = ((article.get(key) or {}).get("compact") or {}).get("quality") or {}
        if quality.get("has_ellipsis") or not quality.get("complete") or not quality.get("max_words_ok"):
            errors.append(f"{key}_compact_quality")
    for ref in article.get("refs") or []:
        if not ref.get("abstract"):
            errors.append("ref_missing_abstract")
            break
        if not ref.get("compact_abstract"):
            errors.append("ref_missing_compact_abstract")
            break
        quality = (ref.get("compact") or {}).get("quality") or {}
        if ref.get("abstract") and (quality.get("has_ellipsis") or not quality.get("complete") or not quality.get("max_words_ok")):
            errors.append("ref_compact_quality")
            break
    for strategy in PROMPT_STRATEGIES:
        if not ((article.get("prompts") or {}).get(strategy) or {}).get("prompt"):
            errors.append(f"missing_prompt_{strategy}")
    for strategy in ("related_work", "top_k_related_work"):
        quality = (((article.get("prompts") or {}).get(strategy) or {}).get("related_work") or {}).get("quality") or {}
        if quality and (quality.get("has_ellipsis") or not quality.get("complete")):
            errors.append(f"{strategy}_incomplete")
    snippets = article.get("tex_snippets") or {}
    for kind in ("method", "implementation", "evaluation"):
        if not snippets.get(kind):
            errors.append(f"missing_tex_{kind}")
        for snippet in snippets.get(kind) or []:
            text = snippet.get("text") or ""
            quality = snippet.get("quality") or _quality_text(text, min_words=20, max_words=300)
            if not quality.get("complete") or quality.get("has_ellipsis"):
                errors.append(f"bad_tex_{kind}")
                break
            if any(token in text.lower() for token in ["\\begin", "\\end", "itemize", "equation"]):
                errors.append(f"raw_tex_artifact_{kind}")
                break
    return {"passed": not errors, "errors": errors}


def _process_row(
    row: dict[str, Any],
    *,
    client: RunwayClient | None,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    aid = str(row.get("arxiv_id") or "")
    article_dir = out_dir / "articles" / _safe_name(aid)
    llm_dir = article_dir / "llm_calls"
    print(json.dumps({"event": "article_start", "arxiv_id": aid, "time": int(time.time())}, ensure_ascii=False), flush=True)
    raw_record = getattr(args, "dataset_records", {}).get(aid) or {}
    raw_refs = list(raw_record.get("refs") or row.get("refs") or [])
    target_date = _parse_date(row.get("created"))
    leakage_guard: dict[str, Any] = {}
    if target_date is not None:
        raw_refs, leakage_guard = _filter_refs_for_leakage(raw_refs, target_date)
    target_abstract_raw = row.get("abstract") or ""

    target_compact = _summarize_abstract(
        client,
        endpoint=args.endpoint,
        model=args.model,
        title=row.get("title") or "",
        abstract=target_abstract_raw,
        max_words=args.max_abstract_words,
        attempts=args.abstract_attempts,
        temperature=args.temperature,
        log_root=llm_dir / "target_abstract",
        force_api=args.use_api,
    )

    refs: list[dict[str, Any]] = []
    for idx, ref in enumerate(raw_refs):
        hydrated = _hydrate_ref(
            ref,
            arxiv_root=Path(args.arxiv_root),
            allow_title_search=args.allow_title_search,
        )
        compact_cache_key = _abstract_cache_key(hydrated.get("title"), hydrated.get("abstract"))
        compact_cache = getattr(args, "abstract_compact_cache", {})
        compact = compact_cache.get(compact_cache_key)
        if compact:
            compact = dict(compact)
            compact["source"] = f"reused_{compact.get('source') or 'compact_cache'}"
        else:
            compact = _summarize_abstract(
                client,
                endpoint=args.endpoint,
                model=args.model,
                title=hydrated.get("title") or "",
                abstract=hydrated.get("abstract") or "",
                max_words=args.max_abstract_words,
                attempts=args.abstract_attempts,
                temperature=args.temperature,
                log_root=llm_dir / "refs" / f"{idx:03d}_{_safe_name(_ref_key(ref))}",
                force_api=args.use_api,
            )
            if compact.get("text"):
                compact_cache[compact_cache_key] = compact
                _jsonl_append(out_dir / "ref_compact_by_key.jsonl", {
                    "cache_key": compact_cache_key,
                    "ref_key": _ref_key(hydrated),
                    "ref_arxiv_id": hydrated.get("arxiv_id"),
                    "title": hydrated.get("title"),
                    "compact": compact,
                })
        refs.append({
            "ref_key": _ref_key(hydrated),
            "arxiv_id": hydrated.get("arxiv_id"),
            "title": hydrated.get("title"),
            "year": hydrated.get("year"),
            "abstract": _collapse(hydrated.get("abstract")),
            "abstract_source": hydrated.get("abstract_source"),
            "metadata_path": hydrated.get("metadata_path"),
            "metadata_error": hydrated.get("metadata_error"),
            "compact_abstract": compact["text"],
            "compact": compact,
            "compact_cache_key": compact_cache_key,
        })
    missing_after_download = [idx for idx, ref in enumerate(refs) if not ref.get("abstract")]
    if missing_after_download:
        print(json.dumps({
            "event": "ref_abstract_missing_after_download",
            "arxiv_id": aid,
            "missing_ref_indices": missing_after_download,
            "count": len(missing_after_download),
        }, ensure_ascii=False), flush=True)

    cached_indices = None
    if getattr(args, "prompt_caches", None):
        cached = args.prompt_caches["top_k_indices"].get(aid)
        if isinstance(cached, list):
            cached_indices = cached
    top_refs, top_meta = _select_top_refs(refs, args.top_k, cached_indices=cached_indices)
    question = _generate_question(
        client,
        endpoint=args.endpoint,
        model=args.model,
        refs=top_refs,
        min_words=args.min_question_words,
        max_words=args.max_question_words,
        attempts=args.question_attempts,
        temperature=args.temperature,
        log_root=llm_dir / "research_question",
    )

    related_all = _generate_related_work(
        client,
        endpoint=args.endpoint,
        model=args.model,
        refs=refs[: args.full_refs_cap],
        log_root=llm_dir / "related_work_all",
        temperature=args.temperature,
    )
    related_top = _generate_related_work(
        client,
        endpoint=args.endpoint,
        model=args.model,
        refs=top_refs,
        log_root=llm_dir / "related_work_top",
        temperature=args.temperature,
    )
    raw_tex_snippets = _raw_section_candidates(row, args.tex_snippet_chars, args.raw_snippets_per_kind)
    tex_snippets = _curate_tex_snippets(
        client,
        endpoint=args.endpoint,
        model=args.model,
        row=row,
        raw=raw_tex_snippets,
        log_root=llm_dir / "tex_snippets",
        temperature=args.temperature,
    )

    full_refs = refs[: args.full_refs_cap]
    prompts = {
        "full_refs": {
            "prompt": _plain_refs_prompt(full_refs),
            "n_refs": len(full_refs),
            "source": "rebuilt_compact_abstracts",
        },
        "top_k_refs": {
            "prompt": _plain_refs_prompt(top_refs),
            "n_refs": len(top_refs),
            "source": top_meta["source"],
        },
        "related_work": {
            "prompt": _related_prompt(related_all["text"]),
            "n_refs": len(full_refs),
            "source": related_all["source"],
            "related_work": related_all,
        },
        "top_k_related_work": {
            "prompt": _related_prompt(related_top["text"]),
            "n_refs": len(top_refs),
            "source": related_top["source"],
            "related_work": related_top,
        },
        "with_research_question": {
            "prompt": _condition_prompt(top_refs, question["text"]),
            "n_refs": len(top_refs),
            "source": question["source"],
            "research_question": question,
        },
    }

    article = {
        "schema_version": "v3_quality_cache_v1",
        "built_at": int(time.time()),
        "arxiv_id": aid,
        "sample_id": row.get("sample_id"),
        "title": row.get("title"),
        "created": row.get("created"),
        "categories": row.get("categories") or [],
        "paper_type": row.get("paper_type"),
        "category_family": row.get("category_family"),
        "quality": row.get("quality") or {},
        "target_abstract": {
            "original": _collapse(target_abstract_raw),
            "compact": target_compact,
        },
        "refs": refs,
        "top_refs": [{"ref_key": r["ref_key"], "title": r.get("title"), "arxiv_id": r.get("arxiv_id")} for r in top_refs],
        "top_ref_selection": top_meta,
        "leakage_guard": leakage_guard,
        "research_question": question,
        "raw_tex_snippets": raw_tex_snippets,
        "tex_snippets": tex_snippets,
        "prompts": prompts,
        "source": {
            "accepted_target_cache": str(args.input),
            "target_cache_path": row.get("target_cache", {}).get("cache_path") if isinstance(row.get("target_cache"), dict) else None,
        },
    }
    article["quality_audit"] = _article_quality(article)

    write_json(article_dir / "article_cache.json", article)
    for strategy, prompt in prompts.items():
        (article_dir / "prompts").mkdir(parents=True, exist_ok=True)
        (article_dir / "prompts" / f"{strategy}.txt").write_text(prompt["prompt"])
    write_json(article_dir / "tex_snippets.json", article["tex_snippets"])
    write_json(article_dir / "refs.json", refs)
    write_json(article_dir / "quality_audit.json", article["quality_audit"])
    top_k_index_row = {
        "key": aid,
        "value": top_meta.get("indices") or [],
        "source": top_meta.get("source"),
        "top_k": top_meta.get("top_k"),
        "top_refs": article["top_refs"],
    }
    write_json(article_dir / "top_k_5_index.json", top_k_index_row)
    for ref_idx, ref in enumerate(refs):
        _jsonl_append(out_dir / "ref_abstract_cache.jsonl", {
            "article_arxiv_id": aid,
            "ref_index": ref_idx,
            "ref_key": ref.get("ref_key"),
            "ref_arxiv_id": ref.get("arxiv_id"),
            "title": ref.get("title"),
            "year": ref.get("year"),
            "abstract": ref.get("abstract"),
            "abstract_source": ref.get("abstract_source"),
            "compact_abstract": ref.get("compact_abstract"),
            "compact_source": (ref.get("compact") or {}).get("source"),
            "compact_quality": (ref.get("compact") or {}).get("quality"),
        })
    _jsonl_append(out_dir / "top_k_5_index.jsonl", top_k_index_row)
    return {
        "arxiv_id": aid,
        "sample_id": row.get("sample_id"),
        "title": row.get("title"),
        "article_dir": str(article_dir),
        "quality_passed": article["quality_audit"]["passed"],
        "quality_errors": article["quality_audit"]["errors"],
        "question_words": question["quality"]["words"],
        "question_source": question["source"],
        "ref_count": len(refs),
        "raw_ref_count": len(raw_refs),
        "ref_missing_abstract_count": sum(1 for r in refs if not r.get("abstract")),
        "prompt_strategies": sorted(prompts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build high-quality V3 article/ref/prompt/TeX caches for strict training articles.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--input", type=Path, default=ROOT / "runs" / "training_data" / "v3_0_targets_qwen25_32b_strict_batch1000_audit" / "tex_targets.accepted.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "v3_quality_cache" / "strict929_v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--arxiv-id", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--key-env", default="RUNWAY_OPUS47_API_KEY")
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--endpoint", default="google_anthropic")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--question-attempts", type=int, default=5)
    parser.add_argument("--abstract-attempts", type=int, default=3)
    parser.add_argument("--min-question-words", type=int, default=120)
    parser.add_argument("--max-question-words", type=int, default=200)
    parser.add_argument("--max-abstract-words", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full-refs-cap", type=int, default=40)
    parser.add_argument("--tex-snippet-chars", type=int, default=1800)
    parser.add_argument("--raw-snippets-per-kind", type=int, default=8)
    parser.add_argument("--arxiv-root", default=None)
    parser.add_argument("--allow-title-search", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.arxiv_root is None:
        args.arxiv_root = cfg.get("arxiv_root") or ROOT.parent / "data" / "arxiv" / "papers"
    args.dataset_records = load_dataset_records(Path(cfg["dataset_dir"]), ["train", "val", "test"])
    args.prompt_caches = load_prompt_property_caches(cfg)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    ids = set(args.arxiv_id or []) or None
    rows = _load_rows(args.input, arxiv_ids=ids)
    if args.num_shards > 1:
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.jsonl"
    done = {} if args.force else _load_done(index_path)
    args.abstract_compact_cache = {} if args.force else _load_compact_cache(args.output / "ref_compact_by_key.jsonl")
    client = RunwayClient(cfg, key_env=args.key_env) if args.use_api else None

    summary = {
        "started_at": int(time.time()),
        "input": str(args.input),
        "output": str(args.output),
        "rows_requested": len(rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "use_api": args.use_api,
        "model": args.model if args.use_api else None,
        "endpoint": args.endpoint if args.use_api else None,
        "processed": 0,
        "skipped_existing": 0,
        "quality_passed": 0,
        "quality_failed": 0,
        "errors": 0,
        "ref_missing_abstract_total": 0,
        "ref_records_total": 0,
        "question_words": [],
        "quality_error_counts": {},
    }
    error_counts: Counter[str] = Counter()
    for row in rows:
        aid = str(row.get("arxiv_id") or "")
        if not aid:
            continue
        if aid in done and not args.force:
            summary["skipped_existing"] += 1
            continue
        try:
            result = _process_row(row, client=client, args=args, out_dir=args.output)
            _jsonl_append(index_path, result)
            summary["processed"] += 1
            summary["question_words"].append(result["question_words"])
            summary["ref_records_total"] += result["ref_count"]
            summary["ref_missing_abstract_total"] += result["ref_missing_abstract_count"]
            if result["quality_passed"]:
                summary["quality_passed"] += 1
            else:
                summary["quality_failed"] += 1
                error_counts.update(result["quality_errors"])
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            summary["errors"] += 1
            err = {"arxiv_id": aid, "error": repr(exc)}
            _jsonl_append(args.output / "errors.jsonl", err)
            print(json.dumps(err, ensure_ascii=False), file=sys.stderr, flush=True)
    summary["finished_at"] = int(time.time())
    summary["quality_error_counts"] = dict(error_counts)
    if summary["question_words"]:
        words = summary["question_words"]
        summary["question_word_stats"] = {
            "min": min(words),
            "max": max(words),
            "mean": round(sum(words) / len(words), 2),
        }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
