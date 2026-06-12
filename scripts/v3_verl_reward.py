#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


_PROC_CACHE_KEY = "_v3_verl_reward_proc_cache"
if not hasattr(sys, _PROC_CACHE_KEY):
    setattr(sys, _PROC_CACHE_KEY, {})
_proc_cache: dict = getattr(sys, _PROC_CACHE_KEY)

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

_EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_PRS_WEIGHT = float(os.environ.get("V3_PRS_WEIGHT", "0.8"))
_FORMAT_WEIGHT = float(os.environ.get("V3_FORMAT_WEIGHT", "0.2"))
_ROLLOUT_LOG = os.environ.get("V3_RL_ROLLOUT_LOG", "")
_rollout_count = 0


def _get_encoder() -> SentenceTransformer:
    if "encoder" not in _proc_cache:
        _proc_cache["encoder"] = SentenceTransformer(_EMBED_MODEL)
    return _proc_cache["encoder"]


def _extract_proposal(text: str) -> str:
    m = re.search(r"<proposal>(.*?)</proposal>", text, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else text
    return re.sub(r"<[^>]+>", " ", body).strip()


def _format_score(text: str) -> float:
    if "<proposal" not in text or "</proposal>" not in text:
        return 0.0
    present = 0
    for tag in TARGET_XML_TAGS:
        if re.search(fr"<{tag}>.*?</{tag}>", text, re.DOTALL | re.IGNORECASE):
            present += 1
    return present / len(TARGET_XML_TAGS)


def _prs_score(proposal_text: str, target_text: str) -> float:
    if not proposal_text.strip() or not target_text.strip():
        return 0.0
    enc = _get_encoder()
    embs = enc.encode(
        [proposal_text, _extract_proposal(target_text)],
        batch_size=2,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    # Cosine can be negative. Clamp to [0, 1] because verl rewards are easier to
    # interpret when bounded and V1 PRS normally lives in the positive range.
    return float(min(1.0, max(0.0, np.dot(embs[0], embs[1]))))


def _write_rollout_log(data: dict) -> None:
    if not _ROLLOUT_LOG:
        return
    try:
        path = Path(_ROLLOUT_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        pass


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **_: object,
) -> dict:
    """verl custom reward entry point.

    V3 PRS mirrors V1 PRS but uses the strict synthesized proposal target as
    ground truth instead of the paper abstract:

        reward = 0.8 * cosine(generated_proposal, target_proposal) + 0.2 * XML_format
    """
    global _rollout_count
    proposal_text = _extract_proposal(solution_str)
    fmt = _format_score(solution_str)
    if data_source != "prs":
        result = {"score": fmt, "format": fmt, "unsupported_data_source": data_source}
    else:
        prs = _prs_score(proposal_text, ground_truth)
        score = _PRS_WEIGHT * prs + _FORMAT_WEIGHT * fmt
        result = {"score": float(score), "prs": float(prs), "format": float(fmt)}

    _rollout_count += 1
    if _rollout_count <= 20 or _rollout_count % 50 == 0:
        _write_rollout_log({
            "ts": time.time(),
            "rollout_count": _rollout_count,
            "data_source": data_source,
            "sample_id": (extra_info or {}).get("sample_id"),
            "arxiv_id": (extra_info or {}).get("arxiv_id"),
            "solution_preview": solution_str[:800],
            "target_preview": ground_truth[:500],
            **result,
        })
    return result
