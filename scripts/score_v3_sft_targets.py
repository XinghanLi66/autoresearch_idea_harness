#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from autoresearch_idea_harness.io import iter_jsonl, write_json
from score_v3_proposal_quality import score_proposal


def _task_packet_from_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": row.get("paper_type") or row.get("task") or "v3_sft_target",
        "subtask": row.get("category_family") or row.get("primary_category") or row.get("arxiv_id"),
        "metric_name": "research_quality",
        "pass_metric": None,
        "condition_strategy": row.get("condition_strategy"),
        "proposal_granularity": "one worker-implementable idea",
        "worker_constraints": {
            "modifiable_files": ["proposal target text"],
            "entrypoint": "human/expert downstream evaluation",
        },
    }


def _target_from_sft_row(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if messages and isinstance(messages[-1], dict):
        return str(messages[-1].get("content") or "")
    return str(row.get("target") or "")


def score_targets(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scores = []
    verdicts = Counter()
    hard_failures = Counter()
    by_category: dict[str, list[int]] = {}
    for idx, row in enumerate(iter_jsonl(input_path), 1):
        if args.limit and idx > args.limit:
            break
        target = _target_from_sft_row(row)
        task_packet = _task_packet_from_sft_row(row)
        result = score_proposal(target, task_packet, allow_raw_target=args.allow_raw_target)
        scores.append(int(result["score"]))
        verdicts[str(result["verdict"])] += 1
        for failure in result.get("hard_failures") or []:
            hard_failures[str(failure)] += 1
        category = str(row.get("category_family") or "unknown")
        by_category.setdefault(category, []).append(int(result["score"]))
        rows.append({
            "idx": idx,
            "sample_id": row.get("sample_id"),
            "arxiv_id": row.get("arxiv_id"),
            "created": row.get("created"),
            "title": row.get("title"),
            "category_family": category,
            "paper_type": row.get("paper_type"),
            "score": result["score"],
            "verdict": result["verdict"],
            "hard_failures": result.get("hard_failures") or [],
            "criteria": result.get("criteria") or {},
            "target_chars": len(target),
        })

    summary: dict[str, Any] = {
        "input": str(input_path),
        "row_count": len(rows),
        "allow_raw_target": args.allow_raw_target,
        "score": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": round(statistics.median(scores), 3) if scores else None,
            "p25": sorted(scores)[len(scores) // 4] if scores else None,
            "p75": sorted(scores)[(len(scores) * 3) // 4] if scores else None,
        },
        "verdicts": dict(verdicts),
        "hard_failures": dict(hard_failures),
        "by_category": {
            key: {
                "count": len(vals),
                "mean_score": round(statistics.mean(vals), 3),
                "min_score": min(vals),
                "max_score": max(vals),
            }
            for key, vals in sorted(by_category.items())
        },
        "rows": rows,
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "rows.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = [
        "# V3 SFT Target Quality",
        "",
        f"- Input: `{input_path}`",
        f"- Rows: {summary['row_count']}",
        f"- Allow raw target: {summary['allow_raw_target']}",
        f"- Score mean/median: {summary['score']['mean']} / {summary['score']['median']}",
        f"- Verdicts: {summary['verdicts']}",
        f"- Hard failures: {summary['hard_failures']}",
        "",
        "| Category | Count | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in summary["by_category"].items():
        lines.append(
            f"| `{key}` | {value['count']} | {value['mean_score']} | {value['min_score']} | {value['max_score']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Score V3 SFT targets with the local proposal quality scorer.")
    parser.add_argument("--input", default=str(ROOT / "runs" / "training_data" / "v3_0_sft_qwen25_32b_200" / "train.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "training_data" / "v3_0_sft_qwen25_32b_200_quality"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-raw-target", action="store_true")
    args = parser.parse_args()
    summary = score_targets(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
