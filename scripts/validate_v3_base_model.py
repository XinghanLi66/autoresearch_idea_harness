#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.io import load_config
from autoresearch_idea_harness.model_registry import resolve_base_model


def main() -> None:
    p = argparse.ArgumentParser(description="Validate V3 base-model metadata for leakage-safe training.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--base-model-id", default=None)
    p.add_argument("--base-model-path", default=None)
    p.add_argument("--base-model-release-date", default=None)
    p.add_argument("--allow-non-32b", action="store_true", help="Only for smoke/debug validation.")
    p.add_argument("--allow-missing-model", action="store_true", help="Only for planning without local files.")
    p.add_argument("--allow-smoke-model", action="store_true", help="Allow registry entries marked smoke_only.")
    args = p.parse_args()
    cfg = load_config(args.config)
    resolved = resolve_base_model(
        cfg,
        model_id=args.base_model_id,
        model_path=args.base_model_path,
        release_date=args.base_model_release_date,
        require_32b=not args.allow_non_32b,
        allow_missing_model=args.allow_missing_model,
        allow_smoke_model=args.allow_smoke_model,
    )
    print(json.dumps(resolved, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
