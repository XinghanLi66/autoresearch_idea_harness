#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.target_synthesis import synthesize_v3_targets


def main() -> None:
    p = argparse.ArgumentParser(description="Synthesize/resume V3 TeX-grounded target cache.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument(
        "--manifest-dir",
        default=str(ROOT / "runs" / "training_data" / "v3_0_manifest"),
        help="Directory containing V3 samples.jsonl.",
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "runs" / "training_data" / "v3_0_targets" / "tex_targets.jsonl"),
    )
    p.add_argument("--skipped-output", default=None)
    p.add_argument("--errors-output", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--arxiv-id", default=None)
    p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--model", default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-tex-chars", type=int, default=None)
    p.add_argument("--prompt-version", default=None, help="Override target_synthesis.prompt_version, e.g. v3_strict or legacy.")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument("--chunk-timeout", type=float, default=None)
    p.add_argument("--call-timeout", type=float, default=None, help="Per-sample timeout in seconds.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true", help="Write deterministic mock targets for offline plumbing tests.")
    args = p.parse_args()

    cfg = load_config(args.config)
    ts_cfg = cfg.get("target_synthesis", {})
    if args.prompt_version:
        ts_cfg["prompt_version"] = args.prompt_version
        cfg["target_synthesis"] = ts_cfg
    temperature = args.temperature if args.temperature is not None else ts_cfg.get("temperature", 0.3)
    summary = asyncio.run(
        synthesize_v3_targets(
            cfg,
            manifest_dir=Path(args.manifest_dir),
            output_file=Path(args.output),
            skipped_file=Path(args.skipped_output) if args.skipped_output else None,
            errors_file=Path(args.errors_output) if args.errors_output else None,
            limit=args.limit,
            arxiv_id=args.arxiv_id,
            split=args.split,
            model=args.model or ts_cfg.get("model"),
            max_tokens=int(args.max_tokens or ts_cfg.get("max_tokens", 8192)),
            temperature=temperature,
            concurrency=int(args.concurrency or ts_cfg.get("concurrency", 2)),
            max_tex_chars=int(args.max_tex_chars or ts_cfg.get("max_tex_chars", 24000)),
            connect_timeout=args.connect_timeout,
            chunk_timeout=float(args.chunk_timeout if args.chunk_timeout is not None else ts_cfg.get("chunk_timeout", 180.0)),
            call_timeout=float(args.call_timeout if args.call_timeout is not None else ts_cfg.get("call_timeout", 420.0)),
            dry_run=args.dry_run,
            mock=args.mock,
        )
    )
    print(summary)


if __name__ == "__main__":
    main()
