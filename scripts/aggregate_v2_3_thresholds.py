#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_idea_harness.formal_sweep import primary_subtask
from autoresearch_idea_harness.io import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate V2.3 worker-only calibration thresholds.")
    p.add_argument("--root", default=str(ROOT / "runs" / "formal_sweeps" / "v2_3_calibration_worker_only"))
    p.add_argument("--out", default=str(ROOT / "configs" / "v2_3_thresholds.json"))
    p.add_argument("--min-samples", type=int, default=20)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    by_task: dict[str, list[dict]] = {}
    for sample_dir in sorted(root.glob("*__empty_worker_only__s*")):
        summary = read_json(sample_dir / "summary.json")
        wr = summary.get("worker_result") or {}
        if wr.get("status") != "done" or wr.get("val_metric") is None:
            continue
        task = summary.get("task")
        if task:
            by_task.setdefault(task, []).append({
                "sample_id": summary.get("sample_id") or sample_dir.name,
                "val_metric": float(wr["val_metric"]),
            })

    thresholds: dict[str, dict] = {}
    incomplete: dict[str, int] = {}
    for task, rows in sorted(by_task.items()):
        if len(rows) < args.min_samples and not args.allow_partial:
            incomplete[task] = len(rows)
            continue
        best = max(rows, key=lambda r: r["val_metric"])
        subtask = primary_subtask(task)
        thresholds.setdefault(task, {})
        thresholds[task][subtask] = {
            "baseline_metric": round(float(best["val_metric"]), 4),
            "pass_threshold": 0.01,
            "pass_metric": round(float(best["val_metric"]) + 0.01, 4),
            "source": "v2_3_empty_worker_only_calibration",
            "sample_count": len(rows),
            "best_sample_id": best["sample_id"],
            "updated_at": utc_now(),
        }
    if incomplete:
        print({"status": "incomplete", "counts": incomplete, "hint": "use --allow-partial to write anyway"})
        raise SystemExit(2)
    out = Path(args.out)
    write_json(out, {"task_thresholds": thresholds})
    print({"status": "ok", "tasks": sorted(thresholds), "out": str(out)})


if __name__ == "__main__":
    main()
