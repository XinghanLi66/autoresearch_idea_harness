#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v3_target_cache import audit_cache
from autoresearch_idea_harness.io import load_config, write_json
from autoresearch_idea_harness.target_synthesis import synthesize_v3_targets
from autoresearch_idea_harness.training_manifest import collate_v3_sft_dataset
from autoresearch_idea_harness.v3_report import collect_v3_status, write_v3_report


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def arxiv_ids(path: Path) -> set[str]:
    return {str(row.get("arxiv_id")) for row in iter_jsonl(path) if row.get("arxiv_id")}


def paths(batch_name: str) -> dict[str, Path]:
    data_root = ROOT / "runs" / "training_data"
    target_dir = data_root / f"v3_0_targets_qwen25_32b_{batch_name}"
    audit_dir = data_root / f"v3_0_targets_qwen25_32b_{batch_name}_audit"
    sft_dir = data_root / f"v3_0_sft_qwen25_32b_{batch_name}"
    return {
        "target_dir": target_dir,
        "target_cache": target_dir / "tex_targets.jsonl",
        "errors": target_dir / "tex_targets.errors.jsonl",
        "repair_root": target_dir / "repairs" / time.strftime("%Y%m%d_%H%M%S"),
        "audit_dir": audit_dir,
        "accepted_cache": audit_dir / "tex_targets.accepted.jsonl",
        "sft_dir": sft_dir,
    }


async def repair_batch(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    cfg.setdefault("target_synthesis", {})
    cfg["target_synthesis"]["prompt_version"] = args.prompt_version
    batch_paths = paths(args.batch_name)
    errors = iter_jsonl(args.errors_file or batch_paths["errors"])
    done = arxiv_ids(batch_paths["target_cache"])
    repair_ids: list[str] = []
    for row in errors:
        aid = str(row.get("arxiv_id") or "")
        if aid and aid not in done and aid not in repair_ids:
            repair_ids.append(aid)
    if args.arxiv_id:
        repair_ids = [args.arxiv_id]
    if args.limit:
        repair_ids = repair_ids[: args.limit]

    repaired: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for aid in repair_ids:
        repair_file = batch_paths["repair_root"] / aid.replace(".", "_") / "tex_targets.jsonl"
        summary = await synthesize_v3_targets(
            cfg,
            manifest_dir=Path(args.manifest_dir),
            output_file=repair_file,
            limit=1,
            arxiv_id=aid,
            model=args.model or cfg.get("target_synthesis", {}).get("model"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=1,
            max_tex_chars=args.max_tex_chars,
            connect_timeout=args.connect_timeout,
            chunk_timeout=args.chunk_timeout,
            call_timeout=args.call_timeout,
            mock=args.mock,
        )
        rows = iter_jsonl(repair_file)
        if len(rows) != 1:
            failed.append({"arxiv_id": aid, "summary": summary, "repair_file": str(repair_file)})
            continue
        row = rows[0]
        if row.get("arxiv_id") not in done:
            append_jsonl(batch_paths["target_cache"], row)
            done.add(str(row.get("arxiv_id")))
        repaired.append({"arxiv_id": aid, "repair_file": str(repair_file), "summary": summary})

    audit_summary = audit_cache(SimpleNamespace(
        input=str(batch_paths["target_cache"]),
        output_dir=str(batch_paths["audit_dir"]),
        limit=0,
        require_prompt_version=args.prompt_version,
    ))
    collate_summary = collate_v3_sft_dataset(
        manifest_dir=Path(args.manifest_dir),
        target_cache_paths=[batch_paths["accepted_cache"]],
        output_dir=batch_paths["sft_dir"],
        include_cot=args.include_cot,
        min_quality_score=args.min_quality_score,
    )
    status = collect_v3_status(ROOT / "runs" / "training_data", ROOT / "runs" / "training", cfg=cfg)
    report_path = ROOT / "runs" / "reports" / "v3_training_status.md"
    write_v3_report(status, output_md=report_path, output_json=report_path.with_suffix(".json"))
    pipeline_summary = {
        "batch_name": args.batch_name,
        "repaired_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repair_ids": repair_ids,
        "repaired": repaired,
        "failed": failed,
        "paths": {key: str(value) for key, value in batch_paths.items()},
        "audit": audit_summary,
        "collate": collate_summary,
        "report": {"md": str(report_path), "json": str(report_path.with_suffix(".json"))},
    }
    write_json(batch_paths["audit_dir"] / "pipeline_summary.json", pipeline_summary)
    write_json(batch_paths["repair_root"] / "repair_summary.json", pipeline_summary)
    return pipeline_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry failed V3 target synthesis rows and rerun audit/collate.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "runs" / "training_data" / "v3_1_manifest_qwen25_32b_v1sem_strict929"))
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--errors-file", type=Path, default=None)
    parser.add_argument("--arxiv-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt-version", default="v3_strict")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tex-chars", type=int, default=16000)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--chunk-timeout", type=float, default=180.0)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument("--min-quality-score", type=float, default=None)
    parser.add_argument("--include-cot", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(repair_batch(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failed = summary.get("failed") or []
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
