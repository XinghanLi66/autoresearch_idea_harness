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


def _set_prompt_version(cfg: dict[str, Any], prompt_version: str) -> None:
    cfg.setdefault("target_synthesis", {})
    cfg["target_synthesis"]["prompt_version"] = prompt_version


def _paths(batch_name: str) -> dict[str, Path]:
    data_root = ROOT / "runs" / "training_data"
    target_dir = data_root / f"v3_0_targets_qwen25_32b_{batch_name}"
    audit_dir = data_root / f"v3_0_targets_qwen25_32b_{batch_name}_audit"
    sft_dir = data_root / f"v3_0_sft_qwen25_32b_{batch_name}"
    return {
        "target_dir": target_dir,
        "target_cache": target_dir / "tex_targets.jsonl",
        "audit_dir": audit_dir,
        "accepted_cache": audit_dir / "tex_targets.accepted.jsonl",
        "sft_dir": sft_dir,
    }


async def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    _set_prompt_version(cfg, args.prompt_version)
    paths = _paths(args.batch_name)
    start = time.time()

    synthesis_summary: dict[str, Any] | None = None
    if not args.skip_synthesis:
        synthesis_summary = await synthesize_v3_targets(
            cfg,
            manifest_dir=Path(args.manifest_dir),
            output_file=paths["target_cache"],
            limit=args.limit,
            arxiv_id=args.arxiv_id,
            split=args.split,
            model=args.model or cfg.get("target_synthesis", {}).get("model"),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            max_tex_chars=args.max_tex_chars,
            connect_timeout=args.connect_timeout,
            chunk_timeout=args.chunk_timeout,
            call_timeout=args.call_timeout,
            dry_run=args.dry_run,
            mock=args.mock,
        )
        if args.dry_run:
            return {"batch_name": args.batch_name, "dry_run": True, "synthesis": synthesis_summary, "paths": {k: str(v) for k, v in paths.items()}}

    audit_summary = audit_cache(SimpleNamespace(
        input=str(paths["target_cache"]),
        output_dir=str(paths["audit_dir"]),
        limit=0,
        require_prompt_version=args.prompt_version,
    ))

    collate_summary: dict[str, Any] | None = None
    if int(audit_summary.get("accepted_count") or 0) > 0 and not args.skip_collate:
        collate_summary = collate_v3_sft_dataset(
            manifest_dir=Path(args.manifest_dir),
            target_cache_paths=[paths["accepted_cache"]],
            output_dir=paths["sft_dir"],
            include_cot=args.include_cot,
            min_quality_score=args.min_quality_score,
        )

    status = collect_v3_status(ROOT / "runs" / "training_data", ROOT / "runs" / "training", cfg=cfg)
    report_path = ROOT / "runs" / "reports" / "v3_training_status.md"
    write_v3_report(status, output_md=report_path, output_json=report_path.with_suffix(".json"))

    summary = {
        "batch_name": args.batch_name,
        "elapsed_sec": round(time.time() - start, 3),
        "paths": {k: str(v) for k, v in paths.items()},
        "synthesis": synthesis_summary,
        "audit": audit_summary,
        "collate": collate_summary,
        "report": {
            "md": str(report_path),
            "json": str(report_path.with_suffix(".json")),
        },
    }
    write_json(paths["audit_dir"] / "pipeline_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V3 strict target synthesis -> audit -> SFT collate.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "runs" / "training_data" / "v3_0_manifest_qwen25_32b"))
    parser.add_argument("--batch-name", required=True, help="Suffix, e.g. strict_batch100.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--arxiv-id", default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--prompt-version", default="v3_strict")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tex-chars", type=int, default=16000)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--chunk-timeout", type=float, default=180.0)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument("--min-quality-score", type=float, default=None)
    parser.add_argument("--include-cot", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--skip-collate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    summary = asyncio.run(run_pipeline(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    audit = summary.get("audit") or {}
    return 0 if args.dry_run or int(audit.get("accepted_count") or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
