#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import iter_jsonl, write_json


def _target_text(row: dict[str, Any]) -> str:
    target = row.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
    return ""


def _prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and len(messages) >= 2:
        return [
            {"role": str(messages[0].get("role")), "content": _normalize_prompt_text(str(messages[0].get("content") or ""))},
            {"role": str(messages[1].get("role")), "content": _normalize_prompt_text(str(messages[1].get("content") or ""))},
        ]
    system = str(row.get("system") or "")
    prompt = str(row.get("prompt") or "")
    if system and prompt:
        return [{"role": "system", "content": _normalize_prompt_text(system)}, {"role": "user", "content": _normalize_prompt_text(prompt)}]
    return []


def _normalize_prompt_text(text: str) -> str:
    """Keep old collated rows usable while applying the current hard schema wording."""
    return text.replace(
        "Output one proposal. Prefer this XML schema when possible:",
        "Output one proposal using this XML schema:",
    )


def build_v3_rl_parquet(
    input_jsonl: Path,
    output_parquet: Path,
    tokenizer_path: Path,
    reward_type: str = "prs",
    limit: int | None = None,
) -> dict[str, Any]:
    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in iter_jsonl(input_jsonl):
        if limit is not None and len(records) >= limit:
            break
        target = _target_text(row)
        messages = _prompt_messages(row)
        if not target or len(messages) < 2:
            skipped.append({
                "sample_id": row.get("sample_id"),
                "arxiv_id": row.get("arxiv_id"),
                "reason": "missing_target_or_prompt",
            })
            continue
        # Important: do not include the assistant target in the prompt.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        records.append({
            "prompt": prompt,
            "data_source": reward_type,
            "reward_model": {"ground_truth": target},
            "extra_info": {
                "sample_id": row.get("sample_id"),
                "arxiv_id": row.get("arxiv_id"),
                "created": row.get("created"),
                "split": row.get("split"),
                "condition_strategy": row.get("condition_strategy"),
                "target_schema_version": row.get("target_schema_version"),
                "quality_score": (row.get("quality") or {}).get("score"),
            },
            "sample_id": row.get("sample_id"),
            "arxiv_id": row.get("arxiv_id"),
            "created": row.get("created"),
        })

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output_parquet, index=False)
    summary = {
        "input_jsonl": str(input_jsonl),
        "output_parquet": str(output_parquet),
        "tokenizer_path": str(tokenizer_path),
        "reward_type": reward_type,
        "row_count": len(records),
        "skipped_count": len(skipped),
        "skipped_preview": skipped[:20],
        "leakage_policy": "prompt is system+user only; assistant target is reward_model.ground_truth only",
    }
    write_json(output_parquet.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Build V3 RL parquet for verl GRPO/PRS training.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--reward-type", default="prs", choices=["prs"])
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    summary = build_v3_rl_parquet(
        input_jsonl=Path(args.input),
        output_parquet=Path(args.output),
        tokenizer_path=Path(args.tokenizer_path),
        reward_type=args.reward_type,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
