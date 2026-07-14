#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.formal_sweep import (
    V2_3_MODULES,
    V2_3_NEW_CALIBRATION_TASKS,
    V2_3_TASKS,
    SweepOptions,
    run_formal_sweep,
)
from autoresearch_idea_harness.io import load_config


def _split_csv(value: str, default: list[str]) -> list[str]:
    if value == "all":
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Run V2.3 reliable-expert formal sweep.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--root", default=str(ROOT / "runs" / "formal_sweeps" / "v2_3_mls10_modules9"))
    p.add_argument("--mode", choices=["formal", "calibration", "dry-run"], default="formal")
    p.add_argument("--tasks", default="all", help="Comma-separated V2 task ids, or all.")
    p.add_argument("--modules", default="all", help="Comma-separated module ids, or all.")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--experts", default="opus47,gpt55")
    p.add_argument("--worker-mode", choices=["claude", "fixture", "skip"], default="claude")
    p.add_argument("--mock-llm", action="store_true")
    p.add_argument("--mock-experts", action="store_true")
    p.add_argument("--gpu", default="0")
    p.add_argument("--worker-timeout", type=int, default=7200)
    p.add_argument("--max-turns", type=int, default=30)
    p.add_argument("--max-master-advice", type=int, default=1)
    p.add_argument("--free-hparams", action="store_true",
                   help="Allow the worker to choose its own training hyperparameters "
                        "(LR, batch size, epochs, etc.). Loosens Rule 3. Use for "
                        "evaluating optimisation/training-recipe proposals.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--write-thresholds", action="store_true",
                   help="For calibration mode, update configs/v2_3_thresholds.json.")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--result-wait-timeout", type=int, default=7200)
    p.add_argument("--no-eval-wait-timeout", type=int, default=180)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mode == "calibration" and args.tasks == "all":
        tasks = list(V2_3_NEW_CALIBRATION_TASKS)
    else:
        tasks = _split_csv(args.tasks, V2_3_TASKS)
    modules = ["empty_worker_only"] if args.mode == "calibration" and args.modules == "all" else _split_csv(args.modules, list(V2_3_MODULES))
    unknown_tasks = [x for x in tasks if x not in V2_3_TASKS]
    unknown_modules = [x for x in modules if x not in V2_3_MODULES]
    if unknown_tasks:
        raise SystemExit(f"Unknown tasks: {unknown_tasks}. Available: {V2_3_TASKS}")
    if unknown_modules:
        raise SystemExit(f"Unknown modules: {unknown_modules}. Available: {list(V2_3_MODULES)}")
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index must satisfy 0 <= shard-index < shard-count")
    experts = _split_csv(args.experts, ["opus47", "gpt55"])

    opts = SweepOptions(
        root=Path(args.root),
        tasks=tasks,
        modules=modules,
        n_samples=args.n_samples,
        experts=experts,
        mode=args.mode,
        worker_mode=args.worker_mode,
        mock_llm=args.mock_llm,
        mock_experts=args.mock_experts,
        gpu=args.gpu,
        worker_timeout=args.worker_timeout,
        max_turns=args.max_turns,
        max_master_advice=args.max_master_advice,
        free_hparams=args.free_hparams,
        force=args.force,
        write_thresholds=args.write_thresholds,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        result_wait_timeout=args.result_wait_timeout,
        no_eval_wait_timeout=args.no_eval_wait_timeout,
    )
    summary = run_formal_sweep(cfg, opts)
    print(summary)


if __name__ == "__main__":
    main()
