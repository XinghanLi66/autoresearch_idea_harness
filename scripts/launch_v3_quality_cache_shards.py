#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import write_json  # noqa: E402


def _session_exists(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch V3 quality cache builder shards in tmux.")
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "v3_quality_cache" / "strict929_v1")
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--tmux-prefix", default="v3_quality_cache")
    parser.add_argument("--use-api", action="store_true", default=True)
    parser.add_argument("--no-api", action="store_false", dest="use_api")
    parser.add_argument("--allow-title-search", action="store_true", default=True)
    parser.add_argument("--question-attempts", type=int, default=5)
    parser.add_argument("--abstract-attempts", type=int, default=2)
    parser.add_argument("--limit-per-shard", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    args.root.mkdir(parents=True, exist_ok=True)

    launched: list[dict[str, Any]] = []
    for shard_idx in range(args.num_shards):
        shard_name = f"shard_{shard_idx:02d}"
        session = f"{args.tmux_prefix}_{shard_idx:02d}"
        output = args.root / shard_name
        output.mkdir(parents=True, exist_ok=True)
        cmd_parts = [
            "python",
            "scripts/build_v3_quality_cache.py",
            "--output",
            str(output),
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(shard_idx),
            "--question-attempts",
            str(args.question_attempts),
            "--abstract-attempts",
            str(args.abstract_attempts),
        ]
        if args.use_api:
            cmd_parts.append("--use-api")
        if args.allow_title_search:
            cmd_parts.append("--allow-title-search")
        if args.limit_per_shard is not None:
            cmd_parts.extend(["--limit", str(args.limit_per_shard)])
        command = (
            f"cd {shlex.quote(str(ROOT))} && "
            f"{_shell_join(cmd_parts)} 2>&1 | tee {shlex.quote(str(output / 'build.log'))}"
        )
        row = {
            "session": session,
            "shard_index": shard_idx,
            "num_shards": args.num_shards,
            "output": str(output),
            "command": command,
        }
        if _session_exists(session):
            row["status"] = "exists"
            launched.append(row)
            continue
        if not args.dry_run:
            subprocess.run(["tmux", "new-session", "-d", "-s", session, command], check=True)
        row["status"] = "dry_run" if args.dry_run else "launched"
        launched.append(row)

    manifest = {
        "schema_version": "v3_quality_cache_shard_launch_v1",
        "created_at": int(time.time()),
        "root": str(args.root),
        "num_shards": args.num_shards,
        "sessions": launched,
        "merge_command": (
            f"python scripts/merge_v3_quality_cache_shards.py "
            f"--shards-root {shlex.quote(str(args.root))} --output {shlex.quote(str(args.root / 'merged'))}"
        ),
    }
    write_json(args.root / "launch_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    existing = [row["session"] for row in launched if row["status"] == "exists"]
    if existing:
        raise SystemExit(f"Some tmux sessions already exist and were not replaced: {existing}")


if __name__ == "__main__":
    main()
