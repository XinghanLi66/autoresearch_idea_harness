from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from .io import iter_jsonl


PROMPT_STRATEGIES = [
    "full_refs",
    "top_k_refs",
    "related_work",
    "top_k_related_work",
    "with_research_question",
]


class KeyValueCache:
    """Append-only JSONL key/value cache reader with last-write-wins semantics."""

    def __init__(self, *paths: Path) -> None:
        self.paths = [Path(p) for p in paths if p]
        self.data: dict[str, Any] = {}
        for path in self.paths:
            if not path.exists():
                continue
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
                self.data[str(key)] = value

    def get(self, key: str | None, default: Any = None) -> Any:
        if key is None:
            return default
        return self.data.get(str(key), default)

    def set_if_missing(self, path: Path, key: str, value: Any) -> None:
        if key in self.data:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        self.data[key] = value


def abstract_cache_key(abstract: str | None) -> str:
    return hashlib.md5((abstract or "").strip().encode()).hexdigest()


def prompt_cache_dirs(cfg: dict[str, Any]) -> list[Path]:
    dirs: list[Path] = []
    dataset_dir = cfg.get("dataset_dir")
    if dataset_dir:
        dirs.append(Path(dataset_dir) / "prompt_cache")
    proposal_rl_root = cfg.get("proposal_rl_root")
    if proposal_rl_root:
        dirs.append(Path(proposal_rl_root) / "runs" / "dataset" / "prompt_cache")
    runs_dir = cfg.get("runs_dir")
    if runs_dir:
        dirs.append(Path(runs_dir) / "dataset" / "prompt_cache")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def load_prompt_property_caches(cfg: dict[str, Any]) -> dict[str, KeyValueCache]:
    dirs = prompt_cache_dirs(cfg)

    def cache(name: str) -> KeyValueCache:
        return KeyValueCache(*(d / name for d in dirs))

    return {
        "abstract_summary": cache("abstract_summary.jsonl"),
        "research_question": cache("research_question.jsonl"),
        "research_question_short": cache("research_question_short.jsonl"),
        "top_k_indices": cache("top_k_5_index.jsonl"),
        "related_work": cache("related_work.jsonl"),
        "related_work_annotated": cache("related_work_annotated.jsonl"),
        "top_k_related_work": cache("top_k_5_related_work.jsonl"),
    }


def _collapse(text: str | None) -> str:
    return " ".join((text or "").split())


def approx_token_count(text: str | None) -> int:
    return len(_collapse(text).split())


def trim_to_tokens(text: str | None, max_tokens: int, *, min_sentence_tokens: int = 40) -> str:
    words = _collapse(text).split()
    if len(words) <= max_tokens:
        return " ".join(words)
    cut = " ".join(words[:max_tokens])
    sentence_end = max(cut.rfind("."), cut.rfind("?"), cut.rfind("!"))
    if sentence_end > 0 and len(cut[:sentence_end].split()) >= min_sentence_tokens:
        return cut[: sentence_end + 1]
    return cut.rstrip(",;:") + "..."


def compact_research_question(
    arxiv_id: str,
    raw_question: str | None,
    caches: dict[str, KeyValueCache],
    *,
    max_tokens: int = 200,
) -> dict[str, Any]:
    cached = caches.get("research_question_short", KeyValueCache()).get(arxiv_id)
    if isinstance(cached, str) and cached.strip():
        text = _collapse(cached)
        return {
            "text": text,
            "source": "cached_research_question_short",
            "raw_chars": len(raw_question or ""),
            "tokens": approx_token_count(text),
            "complete": not text.endswith("..."),
        }

    raw = raw_question or caches.get("research_question", KeyValueCache()).get(arxiv_id)
    if not isinstance(raw, str) or not raw.strip():
        return {
            "text": "",
            "source": "missing",
            "raw_chars": 0,
            "tokens": 0,
            "complete": False,
        }

    text = raw.strip()
    bold = re.findall(r"\*\*(.+?)\*\*", text, flags=re.DOTALL)
    if bold:
        text = max(bold, key=len)
    else:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
        text = " ".join(lines) if lines else text
    text = text.replace("**", "")
    text = _collapse(text)
    source = "normalized_research_question"
    if approx_token_count(text) > max_tokens:
        text = trim_to_tokens(text, max_tokens)
        source = "token_trimmed_research_question"
    return {
        "text": text,
        "source": source,
        "raw_chars": len(raw),
        "tokens": approx_token_count(text),
        "complete": not text.endswith("..."),
    }


def compact_reference_abstract(
    ref: dict[str, Any],
    caches: dict[str, KeyValueCache],
    *,
    max_tokens: int = 200,
) -> dict[str, Any]:
    abstract = (ref.get("abstract") or "").strip()
    if not abstract:
        return {
            "text": "(no abstract available)",
            "source": "missing",
            "raw_chars": 0,
            "tokens": 0,
            "cache_key": None,
            "complete": False,
        }

    key = abstract_cache_key(abstract)
    cached = caches.get("abstract_summary", KeyValueCache()).get(key)
    if isinstance(cached, str) and cached.strip():
        text = _collapse(cached)
        return {
            "text": text,
            "source": "cached_abstract_summary",
            "raw_chars": len(abstract),
            "tokens": approx_token_count(text),
            "cache_key": key,
            "complete": not text.endswith("..."),
        }

    if approx_token_count(abstract) <= max_tokens:
        text = _collapse(abstract)
        return {
            "text": text,
            "source": "raw_abstract_under_limit",
            "raw_chars": len(abstract),
            "tokens": approx_token_count(text),
            "cache_key": key,
            "complete": True,
        }

    text = trim_to_tokens(abstract, max_tokens)
    return {
        "text": text,
        "source": "fallback_token_trimmed_abstract",
        "raw_chars": len(abstract),
        "tokens": approx_token_count(text),
        "cache_key": key,
        "complete": not text.endswith("..."),
    }


def ref_key(ref: dict[str, Any]) -> str:
    aid = ref.get("arxiv_id")
    if aid:
        return f"arxiv:{aid}"
    payload = json.dumps([ref.get("title", ""), ref.get("year")], ensure_ascii=False, sort_keys=True)
    return "ref_" + hashlib.sha1(payload.encode()).hexdigest()[:16]


def trim_ref_with_prompt_properties(
    ref: dict[str, Any],
    caches: dict[str, KeyValueCache],
    *,
    max_abstract_tokens: int = 200,
) -> dict[str, Any]:
    abstract = compact_reference_abstract(ref, caches, max_tokens=max_abstract_tokens)
    return {
        "ref_key": ref_key(ref),
        "arxiv_id": ref.get("arxiv_id"),
        "title": ref.get("title"),
        "year": ref.get("year"),
        "abstract": abstract["text"],
        "abstract_source": abstract["source"],
        "abstract_raw_chars": abstract["raw_chars"],
        "abstract_tokens": abstract["tokens"],
        "abstract_cache_key": abstract["cache_key"],
        "abstract_complete": abstract["complete"],
        "reference_tex_evidence_status": "pending",
    }


def _ref_entry(idx: int, ref: dict[str, Any]) -> str:
    return (
        f"[{idx}] {ref.get('title') or 'Unknown'} ({ref.get('year') or 'n.d.'})\n"
        f"Abstract: {ref.get('abstract') or '(no abstract available)'}"
    )


def build_ref_block(refs: list[dict[str, Any]]) -> str:
    return "\n\n".join(_ref_entry(i + 1, ref) for i, ref in enumerate(refs))


def select_refs_for_prompt(
    arxiv_id: str,
    refs: list[dict[str, Any]],
    caches: dict[str, KeyValueCache],
    *,
    top_k: int = 5,
    max_abstract_tokens: int = 200,
    mode: str = "top_k",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = "first_k_after_leakage_filter"
    selected: list[dict[str, Any]] = []
    if mode == "all":
        selected = refs[:40]
        source = "all_refs_capped_40"
    else:
        raw_indices = caches.get("top_k_indices", KeyValueCache()).get(arxiv_id)
        if isinstance(raw_indices, list):
            source = "cached_top_k_indices"
            for idx in raw_indices:
                try:
                    selected.append(refs[int(idx)])
                except Exception:
                    continue
        if not selected:
            selected = refs[:top_k]
    selected = selected[:top_k] if mode != "all" else selected
    trimmed = [
        trim_ref_with_prompt_properties(ref, caches, max_abstract_tokens=max_abstract_tokens)
        for ref in selected
    ]
    abstract_source_counts = {
        src: sum(1 for r in trimmed if r.get("abstract_source") == src)
        for src in sorted({str(r.get("abstract_source") or "unknown") for r in trimmed})
    }
    return trimmed, {
        "top_k_source": source,
        "cached_top_k_indices": caches.get("top_k_indices", KeyValueCache()).get(arxiv_id),
        "abstract_source_counts": abstract_source_counts,
    }


def build_v3_condition_prompt(selected_refs: list[dict[str, Any]], research_question: str | None, proposal_format: str) -> str:
    ref_block = build_ref_block(selected_refs)
    question = research_question or "(research question cache missing; synthesize one from the references before training)"
    return (
        f"Below are {len(selected_refs)} papers from a researcher's reading list. "
        "Based on these references, propose a novel research direction.\n\n"
        f"{ref_block}\n\n"
        "The researcher has identified the following open question as their primary motivation:\n"
        f"\"{question}\"\n\n"
        f"{proposal_format}"
    )


def build_prompt_variants(record: dict[str, Any], caches: dict[str, KeyValueCache], proposal_format: str) -> dict[str, dict[str, Any]]:
    arxiv_id = str(record.get("arxiv_id") or "")
    refs = list(record.get("refs") or [])
    shuffled = list(refs)
    random.Random(42).shuffle(shuffled)
    variants: dict[str, dict[str, Any]] = {}

    full_refs, full_meta = select_refs_for_prompt(
        arxiv_id,
        shuffled,
        caches,
        top_k=40,
        mode="all",
    )
    variants["full_refs"] = {
        "status": "built",
        "metadata": full_meta,
        "prompt": (
            f"Below are {len(full_refs)} papers from a researcher's reading list. "
            "Based on these references, propose a novel research direction.\n\n"
            f"{build_ref_block(full_refs)}\n\n{proposal_format}"
        ),
    }

    top_refs, top_meta = select_refs_for_prompt(arxiv_id, refs, caches, top_k=5)
    variants["top_k_refs"] = {
        "status": "built",
        "metadata": top_meta,
        "prompt": (
            f"Below are {len(top_refs)} papers from a researcher's reading list. "
            "Based on these references, propose a novel research direction.\n\n"
            f"{build_ref_block(top_refs)}\n\n{proposal_format}"
        ),
    }

    def title_index(selected_refs: list[dict[str, Any]]) -> str:
        lines = []
        for idx, ref in enumerate(selected_refs, start=1):
            title = str(ref.get("title") or "Unknown").strip()
            lines.append(f"[{idx}] {title}")
        return "\n".join(lines)

    related = caches.get("related_work_annotated", KeyValueCache()).get(arxiv_id) or caches.get("related_work", KeyValueCache()).get(arxiv_id)
    related_index = title_index(full_refs)
    related_block = related or "(related_work cache missing)"
    if related and "**References:**" not in related:
        related_block = f"{related}\n\n**References:**\n{related_index or '(reference list missing)'}"
    variants["related_work"] = {
        "status": "cached" if related else "missing",
        "metadata": {"cache": "related_work_annotated|related_work"},
        "prompt": (
            "A researcher has been studying the following area of the literature:\n\n"
            f"{related_block}\n\n"
            f"Based on this background, propose a novel research direction.\n\n{proposal_format}"
        ),
    }

    top_related = caches.get("top_k_related_work", KeyValueCache()).get(arxiv_id)
    top_related_index = title_index(top_refs)
    top_related_block = top_related or "(top_k_related_work cache missing)"
    if top_related and "**References:**" not in top_related:
        top_related_block = f"{top_related}\n\n**References:**\n{top_related_index or '(reference list missing)'}"
    variants["top_k_related_work"] = {
        "status": "cached" if top_related else "missing",
        "metadata": {"cache": "top_k_5_related_work"},
        "prompt": (
            "A researcher has been studying the following focused area of the literature:\n\n"
            f"{top_related_block}\n\n"
            f"Based on this background, propose a novel research direction.\n\n{proposal_format}"
        ),
    }

    raw_question = caches.get("research_question", KeyValueCache()).get(arxiv_id)
    question = compact_research_question(arxiv_id, raw_question, caches)
    variants["with_research_question"] = {
        "status": "cached" if question["text"] else "missing",
        "metadata": {
            "question_source": question["source"],
            "question_tokens": question["tokens"],
            "question_complete": question["complete"],
            **full_meta,
        },
        "open_question": question,
        "prompt": build_v3_condition_prompt(full_refs, question["text"], proposal_format),
    }
    return variants
