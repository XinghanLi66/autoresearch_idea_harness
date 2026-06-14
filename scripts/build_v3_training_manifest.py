#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.model_registry import resolve_base_model
from autoresearch_idea_harness.training_manifest import build_v3_training_manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a chronological V3 training manifest from classified TeX papers "
            "and raw with_research_question reference packets."
        )
    )
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument(
        "--base-model-id",
        default=None,
        help="Registered base model id in config base_models.registry.",
    )
    p.add_argument(
        "--base-model-path",
        default=None,
        help="Optional base model path for validation when not using registry.",
    )
    p.add_argument(
        "--base-model-release-date",
        default=None,
        help="YYYY-MM-DD. Target papers before this date are excluded to avoid base-model leakage.",
    )
    p.add_argument("--allow-non-32b", action="store_true", help="Only for smoke/debug manifests.")
    p.add_argument("--allow-missing-model", action="store_true", help="Only for planning without local model files.")
    p.add_argument("--allow-smoke-model", action="store_true", help="Allow registry entries marked smoke_only.")
    p.add_argument(
        "--through-date",
        default=None,
        help="Optional YYYY-MM-DD upper bound for target papers.",
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-quality-score", type=float, default=None)
    p.add_argument(
        "--quality-cache-root",
        default=None,
        help="Optional audited V3 article cache root. When set, with_research_question prompts are read directly from article_cache.json.",
    )
    p.add_argument(
        "--require-quality-cache",
        action="store_true",
        help="Skip any paper that is not present in --quality-cache-root.",
    )
    p.add_argument(
        "--target-cache",
        action="append",
        default=None,
        help="JSONL cache containing target_impl_proposal or target_proposal. May be repeated.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    base_model = None
    if args.base_model_id or args.base_model_path:
        base_model = resolve_base_model(
            cfg,
            model_id=args.base_model_id,
            model_path=args.base_model_path,
            release_date=args.base_model_release_date,
            require_32b=not args.allow_non_32b,
            allow_missing_model=args.allow_missing_model,
            allow_smoke_model=args.allow_smoke_model,
        )
        base_model_release_date = base_model["release_date"]
    else:
        if not args.base_model_release_date:
            raise ValueError("pass --base-model-id or --base-model-release-date")
        base_model_release_date = args.base_model_release_date
    summary = build_v3_training_manifest(
        cfg,
        base_model_release_date=base_model_release_date,
        base_model_metadata=base_model,
        through_date=args.through_date,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limit=args.limit,
        min_quality_score=args.min_quality_score,
        target_cache_paths=[Path(p) for p in args.target_cache] if args.target_cache else None,
        quality_cache_root=Path(args.quality_cache_root) if args.quality_cache_root else None,
        require_quality_cache=args.require_quality_cache,
    )
    print(summary)


if __name__ == "__main__":
    main()
