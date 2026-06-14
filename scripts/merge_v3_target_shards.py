#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
from autoresearch_idea_harness.training_manifest import collate_v3_sft_dataset
from autoresearch_idea_harness.v3_report import collect_v3_status, write_v3_report


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def batch_paths(batch_name: str) -> dict[str, Path]:
    data_root = ROOT / "runs" / "training_data"
    target_dir = data_root / f"v3_0_targets_qwen25_32b_{batch_name}"
    audit_dir = data_root / f"v3_0_targets_qwen25_32b_{batch_name}_audit"
    sft_dir = data_root / f"v3_0_sft_qwen25_32b_{batch_name}"
    return {
        "target_dir": target_dir,
        "target_cache": target_dir / "tex_targets.jsonl",
        "errors": target_dir / "tex_targets.errors.jsonl",
        "skipped": target_dir / "tex_targets.skipped.jsonl",
        "audit_dir": audit_dir,
        "accepted_cache": audit_dir / "tex_targets.accepted.jsonl",
        "sft_dir": sft_dir,
    }


def backup(path: Path, stamp: str) -> str | None:
    if not path.exists():
        return None
    dest = path.with_name(f"{path.name}.pre_shard_merge_{stamp}")
    shutil.copy2(path, dest)
    return str(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge accelerated V3 target shard caches into a base batch.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--base-batch-name", required=True)
    parser.add_argument("--shard-batch-name", action="append", default=[])
    parser.add_argument("--shard-prefix", default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--prompt-version", default="v3_strict")
    parser.add_argument("--include-cot", action="store_true")
    parser.add_argument("--min-quality-score", type=float, default=None)
    args = parser.parse_args()

    shard_names = list(args.shard_batch_name)
    if args.shard_prefix and args.num_shards is not None:
        shard_names.extend(f"{args.shard_prefix}_s{idx:02d}" for idx in range(args.num_shards))
    if not shard_names:
        raise SystemExit("provide --shard-batch-name or --shard-prefix with --num-shards")

    cfg = load_config(args.config)
    cfg.setdefault("target_synthesis", {})
    cfg["target_synthesis"]["prompt_version"] = args.prompt_version

    base = batch_paths(args.base_batch_name)
    manifest_rows = iter_jsonl(Path(args.manifest_dir) / "samples.jsonl")
    manifest_order = {str(row.get("arxiv_id")): idx for idx, row in enumerate(manifest_rows) if row.get("arxiv_id")}

    source_paths = [base["target_cache"]]
    error_paths = [base["errors"]]
    skipped_paths = [base["skipped"]]
    for name in shard_names:
        paths = batch_paths(name)
        source_paths.append(paths["target_cache"])
        error_paths.append(paths["errors"])
        skipped_paths.append(paths["skipped"])

    by_aid: dict[str, dict[str, Any]] = {}
    duplicate_rows = 0
    source_counts: dict[str, int] = {}
    for path in source_paths:
        rows = iter_jsonl(path)
        source_counts[str(path)] = len(rows)
        for row in rows:
            aid = row.get("arxiv_id")
            if not aid:
                continue
            aid = str(aid)
            if aid in by_aid:
                duplicate_rows += 1
                continue
            by_aid[aid] = row

    merged_rows = [
        by_aid[aid]
        for aid in sorted(by_aid, key=lambda value: manifest_order.get(value, 10**9))
    ]
    accepted_ids = set(by_aid)

    merged_errors: dict[str, dict[str, Any]] = {}
    for path in error_paths:
        for row in iter_jsonl(path):
            aid = str(row.get("arxiv_id") or "")
            if not aid or aid in accepted_ids:
                continue
            merged_errors.setdefault(aid, row)

    merged_skipped: dict[str, dict[str, Any]] = {}
    for path in skipped_paths:
        for row in iter_jsonl(path):
            aid = str(row.get("arxiv_id") or "")
            if not aid or aid in accepted_ids:
                continue
            merged_skipped.setdefault(aid, row)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backups = {
        "target_cache": backup(base["target_cache"], stamp),
        "errors": backup(base["errors"], stamp),
        "skipped": backup(base["skipped"], stamp),
    }
    write_jsonl(base["target_cache"], merged_rows)
    write_jsonl(base["errors"], list(merged_errors.values()))
    write_jsonl(base["skipped"], list(merged_skipped.values()))

    audit_summary = audit_cache(SimpleNamespace(
        input=str(base["target_cache"]),
        output_dir=str(base["audit_dir"]),
        limit=0,
        require_prompt_version=args.prompt_version,
    ))
    collate_summary = collate_v3_sft_dataset(
        manifest_dir=Path(args.manifest_dir),
        target_cache_paths=[base["accepted_cache"]],
        output_dir=base["sft_dir"],
        include_cot=args.include_cot,
        min_quality_score=args.min_quality_score,
    )
    status = collect_v3_status(ROOT / "runs" / "training_data", ROOT / "runs" / "training", cfg=cfg)
    report_path = ROOT / "runs" / "reports" / "v3_training_status.md"
    write_v3_report(status, output_md=report_path, output_json=report_path.with_suffix(".json"))

    missing_ids = [aid for aid in manifest_order if aid not in accepted_ids]
    summary = {
        "base_batch_name": args.base_batch_name,
        "shard_batch_names": shard_names,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_counts": source_counts,
        "merged_count": len(merged_rows),
        "duplicate_rows_ignored": duplicate_rows,
        "merged_error_count": len(merged_errors),
        "merged_skipped_count": len(merged_skipped),
        "expected_count": args.expected_count,
        "missing_count": len(missing_ids),
        "missing_ids_preview": missing_ids[:50],
        "backups": backups,
        "paths": {key: str(value) for key, value in base.items()},
        "audit": audit_summary,
        "collate": collate_summary,
        "report": {"md": str(report_path), "json": str(report_path.with_suffix(".json"))},
    }
    write_json(base["audit_dir"] / "pipeline_summary.json", summary)
    write_json(base["target_dir"] / f"shard_merge_summary_{stamp}.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.expected_count is not None and len(merged_rows) < args.expected_count:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
