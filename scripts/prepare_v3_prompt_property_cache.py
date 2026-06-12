#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, load_config, load_dataset_records, write_json  # noqa: E402
from autoresearch_idea_harness.prompt_properties import (  # noqa: E402
    KeyValueCache,
    approx_token_count,
    compact_research_question,
    load_prompt_property_caches,
    prompt_cache_dirs,
    trim_to_tokens,
)
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402


ABSTRACT_PROMPT = """\
Rewrite the following paper abstract into a dense summary under {max_tokens} tokens.
Keep the problem, method, data/task, and most important result if present.
Output only the summary.

Abstract:
{abstract}
"""

QUESTION_PROMPT = """\
Rewrite the following open research question into one complete, specific question
under {max_tokens} tokens. Remove markdown headings and unfinished explanatory
tails. Output only the rewritten question.

Open question:
{question}
"""


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _v3_cache_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["runs_dir"]) / "dataset" / "prompt_cache"


def _iter_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    records = load_dataset_records(Path(cfg["dataset_dir"]), ["train", "val", "test"])
    rows = []
    for row in iter_jsonl(cfg["classified_papers"]):
        aid = row.get("arxiv_id")
        if aid and aid in records:
            rows.append({**records[aid], "_classified": row})
    return rows


def _client(cfg: dict[str, Any], key_env: str) -> RunwayClient:
    return RunwayClient(cfg, key_env=key_env)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare V3 short abstract/open-question caches.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--arxiv-id")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-abstract-tokens", type=int, default=200)
    p.add_argument("--max-question-tokens", type=int, default=200)
    p.add_argument("--use-api", action="store_true", help="Call Runway for missing long abstracts/questions.")
    p.add_argument("--key-env", default="RUNWAY_OPUS47_API_KEY")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--endpoint", default="google_anthropic")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary", default=str(ROOT / "runs" / "reports" / "v3_prompt_property_cache_prep.json"))
    args = p.parse_args()

    cfg = load_config(args.config)
    caches = load_prompt_property_caches(cfg)
    out_dir = _v3_cache_dir(cfg)
    abstract_out = out_dir / "abstract_summary.jsonl"
    question_out = out_dir / "research_question_short.jsonl"
    llm = _client(cfg, args.key_env) if args.use_api and not args.dry_run else None

    records = _iter_records(cfg)
    if args.arxiv_id:
        records = [r for r in records if r.get("arxiv_id") == args.arxiv_id]
    if args.limit is not None:
        records = records[: args.limit]

    summary = {
        "created_at": int(time.time()),
        "records_seen": len(records),
        "cache_dirs_read": [str(p) for p in prompt_cache_dirs(cfg)],
        "cache_dir_written": str(out_dir),
        "dry_run": args.dry_run,
        "use_api": args.use_api,
        "abstracts": {"already_cached": 0, "raw_under_limit": 0, "would_api": 0, "written": 0},
        "questions": {"already_short": 0, "normalized": 0, "would_api": 0, "written": 0},
    }

    for record in records:
        aid = str(record.get("arxiv_id") or "")
        raw_q = caches["research_question"].get(aid)
        q_info = compact_research_question(aid, raw_q, caches, max_tokens=args.max_question_tokens)
        if q_info["source"] == "cached_research_question_short":
            summary["questions"]["already_short"] += 1
        elif raw_q:
            value = q_info["text"]
            needs_api = not q_info["complete"] or approx_token_count(value) > args.max_question_tokens
            if needs_api:
                summary["questions"]["would_api"] += 1
            if needs_api and llm is not None:
                result = llm.complete(
                    endpoint=args.endpoint,
                    model=args.model,
                    messages=[{"role": "user", "content": QUESTION_PROMPT.format(question=raw_q, max_tokens=args.max_question_tokens)}],
                    temperature=None,
                    max_tokens=256,
                    stream=False,
                )
                value = trim_to_tokens(result.text, args.max_question_tokens)
            if value and not args.dry_run:
                caches["research_question_short"].set_if_missing(question_out, aid, value)
                summary["questions"]["written"] += 1
            else:
                summary["questions"]["normalized"] += 1

        for ref in record.get("refs") or []:
            abstract = (ref.get("abstract") or "").strip()
            if not abstract:
                continue
            key = hashlib.md5(abstract.encode()).hexdigest()
            if caches["abstract_summary"].get(key):
                summary["abstracts"]["already_cached"] += 1
                continue
            if approx_token_count(abstract) <= args.max_abstract_tokens:
                summary["abstracts"]["raw_under_limit"] += 1
                continue
            summary["abstracts"]["would_api"] += 1
            if llm is None:
                continue
            result = llm.complete(
                endpoint=args.endpoint,
                model=args.model,
                messages=[{"role": "user", "content": ABSTRACT_PROMPT.format(abstract=abstract[:5000], max_tokens=args.max_abstract_tokens)}],
                temperature=None,
                max_tokens=256,
                stream=False,
            )
            value = trim_to_tokens(result.text, args.max_abstract_tokens)
            if value and not args.dry_run:
                caches["abstract_summary"].set_if_missing(abstract_out, key, value)
                summary["abstracts"]["written"] += 1

    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
