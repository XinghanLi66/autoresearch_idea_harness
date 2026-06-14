#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def done_arxiv_ids(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for path in paths:
        for row in iter_jsonl(path):
            aid = row.get("arxiv_id")
            if aid:
                done.add(str(aid))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Create disjoint manifest shards for V3 target synthesis acceleration.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--base-batch-name", required=True)
    parser.add_argument("--shard-prefix", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--include-errors", action="store_true", help="Retry errored rows by not treating error rows as done.")
    args = parser.parse_args()

    if args.num_shards <= 0:
        raise SystemExit("--num-shards must be positive")

    manifest_dir = Path(args.manifest_dir)
    samples_path = manifest_dir / "samples.jsonl"
    samples = iter_jsonl(samples_path)
    if not samples:
        raise SystemExit(f"no samples found at {samples_path}")

    base_target_dir = ROOT / "runs" / "training_data" / f"v3_0_targets_qwen25_32b_{args.base_batch_name}"
    done_paths = [
        base_target_dir / "tex_targets.jsonl",
        base_target_dir / "tex_targets.skipped.jsonl",
    ]
    if not args.include_errors:
        done_paths.append(base_target_dir / "tex_targets.errors.jsonl")
    done = done_arxiv_ids(done_paths)
    remaining = [row for row in samples if str(row.get("arxiv_id") or "") not in done]

    output_root = Path(args.output_root) if args.output_root else (
        ROOT / "runs" / "training_data" / f"{manifest_dir.name}_{args.shard_prefix}_manifests"
    )
    shards: list[list[dict[str, Any]]] = [[] for _ in range(args.num_shards)]
    for idx, row in enumerate(remaining):
        shards[idx % args.num_shards].append(row)

    shard_infos: list[dict[str, Any]] = []
    for idx, rows in enumerate(shards):
        shard_name = f"{args.shard_prefix}_s{idx:02d}"
        shard_dir = output_root / shard_name
        write_jsonl(shard_dir / "samples.jsonl", rows)
        summary = {
            "source_manifest_dir": str(manifest_dir),
            "base_batch_name": args.base_batch_name,
            "shard_name": shard_name,
            "shard_index": idx,
            "num_shards": args.num_shards,
            "sample_count": len(rows),
        }
        write_json(shard_dir / "summary.json", summary)
        shard_infos.append({
            "shard_name": shard_name,
            "manifest_dir": str(shard_dir),
            "sample_count": len(rows),
        })

    plan = {
        "source_manifest_dir": str(manifest_dir),
        "base_batch_name": args.base_batch_name,
        "base_target_dir": str(base_target_dir),
        "output_root": str(output_root),
        "input_sample_count": len(samples),
        "already_done_count": len(done),
        "remaining_count": len(remaining),
        "num_shards": args.num_shards,
        "shards": shard_infos,
    }
    write_json(output_root / "shard_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
