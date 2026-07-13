#!/usr/bin/env python3
"""Preflight a prepared V3 QS smoke run without submitting it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs" / "qs_smoke"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None)
    parser.add_argument("--skip-live", action="store_true", help="Do not query QS resource availability.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(DEFAULT_RUN_ROOT)
    errors: list[str] = []
    warnings: list[str] = []

    plan_path = run_dir / "run_plan.json"
    if not plan_path.exists():
        errors.append(f"missing run_plan.json: {plan_path}")
        plan: dict[str, Any] = {}
    else:
        plan = read_json(plan_path)

    artifacts = plan.get("artifacts") or {}
    for key in ["command", "training_config", "dry_run", "submit"]:
        path = Path(str(artifacts.get(key) or ""))
        if not path.exists():
            errors.append(f"missing artifact {key}: {path}")

    config_path = Path(str(artifacts.get("training_config") or run_dir / "qs_training_config.yaml"))
    if config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        spec = cfg.get("spec") or {}
        resources = spec.get("resources") or {}
        expected = (532, 12, 70, 234)
        actual = (
            int(resources.get("queueId", -1)),
            int(resources.get("cloudId", -1)),
            int(resources.get("clusterId", -1)),
            int(resources.get("resourcePackageId", -1)),
        )
        if actual != expected:
            errors.append(f"QS resource tuple {actual} != expected {expected}")
        if int(spec.get("workerNum", 0)) < 1:
            errors.append("workerNum must be >= 1")
        command = str(spec.get("command") or "")
        for marker in ["nvidia-smi", "/mnt/3fs", "result.json", "smoke.log"]:
            if marker not in command:
                errors.append(f"QS command missing marker {marker!r}")
        if "QS_TOKEN" in command or "RUNWAY_" in command or "OPENAI_API_KEY" in command:
            errors.append("QS command appears to contain secret-like env references")

    live: dict[str, Any] | None = None
    if not args.skip_live:
        proc = subprocess.run(
            ["qs", "resources", "quota", "get", "532", "-o", "json", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            errors.append(f"qs resources quota get failed: {proc.stderr.strip() or proc.stdout[:300]}")
        else:
            try:
                rows = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                errors.append(f"failed to parse quota JSON: {exc}")
            else:
                live = {"quota_rows": rows}
                matching = [
                    row for row in rows
                    if int(row.get("cloudId", -1)) == 12
                    and int(row.get("clusterId", -1)) == 70
                ]
                if not matching:
                    errors.append("live quota has no cloudId=12 clusterId=70 row")
                else:
                    avail = int(matching[0].get("avail", 0))
                    if avail < 4:
                        warnings.append(f"queue 532 has only {avail} visible GPUs available")

    result = {
        "run_dir": str(run_dir),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "live": live,
    }
    (run_dir / "preflight.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


def _latest_run_dir(root: Path) -> Path:
    candidates = [path for path in root.glob("v3_qs_smoke_*") if path.is_dir()]
    if not candidates:
        raise SystemExit(f"no QS smoke run dirs under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    main()
