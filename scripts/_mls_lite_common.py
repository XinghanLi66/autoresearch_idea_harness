#!/usr/bin/env python3
"""Shared helpers for the native-mlsbench MLS-Bench-Lite eval driver.

Reads task metadata straight from the GitHub MLS-Bench checkout (no HuggingFace):
metric column + direction from score_spec.py, baseline values + our run's metric from
leaderboard.csv, subtask labels from config.json. Salvaged from the old audit script.
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

DEFAULT_MLS_ROOT = "/newcpfs/lxh/MLS-Bench"
_IGNORE_COL = re.compile(r"^(timestamp|model|is_final|seed)$|^elapsed")
_COL_DIR = re.compile(r'col\(\s*["\']([^"\']+)["\']\s*\)\s*\.\s*(higher|lower)\s*\(')


def task_dir(slug: str, mls_root: str = DEFAULT_MLS_ROOT) -> Path:
    return Path(mls_root) / "tasks" / slug


def score_spec_directions(slug: str, mls_root: str = DEFAULT_MLS_ROOT) -> dict[str, bool]:
    """{leaderboard_column: higher_is_better} parsed from score_spec.py DSL."""
    p = task_dir(slug, mls_root) / "score_spec.py"
    out: dict[str, bool] = {}
    if p.exists():
        for m in _COL_DIR.finditer(p.read_text(errors="ignore")):
            out[m.group(1)] = (m.group(2) == "higher")
    return out


def visible_settings(slug: str, mls_root: str = DEFAULT_MLS_ROOT) -> list[str]:
    """Non-hidden test_cmd labels from config.json (the runnable settings)."""
    cfg = json.loads((task_dir(slug, mls_root) / "config.json").read_text())
    return [c["label"] for c in cfg.get("test_cmds", [])
            if isinstance(c, dict) and c.get("label") and not c.get("hidden")]


def _leaderboard_rows(slug: str, mls_root: str) -> list[dict]:
    p = task_dir(slug, mls_root) / "leaderboard.csv"
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _cols_for_setting(rows: list[dict], setting: str, dirs: dict[str, bool]) -> list[str]:
    """Metric columns tied to a setting label (e.g. test_acc_resnet20-cifar10 for that subtask)."""
    if not rows:
        return []
    cols = [c for c in rows[0] if c and not _IGNORE_COL.search(c) and c.endswith(setting)]
    # keep only columns with a known direction; fall back to any non-ignored col ending in setting
    known = [c for c in cols if c in dirs]
    return known or cols


def baseline_metrics(slug: str, setting: str, mls_root: str = DEFAULT_MLS_ROOT) -> dict[str, float]:
    """Best baseline value per metric column for a setting (from baseline:* leaderboard rows)."""
    rows = _leaderboard_rows(slug, mls_root)
    dirs = score_spec_directions(slug, mls_root)
    base = [r for r in rows if str(r.get("model", "")).startswith("baseline:")]
    out: dict[str, float] = {}
    for c in _cols_for_setting(rows, setting, dirs):
        vals = _floats(base, c)
        if vals:
            out[c] = max(vals) if dirs.get(c, True) else min(vals)
    return out


def run_metrics(slug: str, setting: str, model_tag: str, mls_root: str = DEFAULT_MLS_ROOT) -> dict[str, float]:
    """Metric per column for the newest leaderboard row whose model matches model_tag."""
    rows = _leaderboard_rows(slug, mls_root)
    dirs = score_spec_directions(slug, mls_root)
    mine = [r for r in rows if model_tag.lower() in str(r.get("model", "")).lower()]
    if not mine:
        return {}
    row = mine[-1]  # leaderboard is append-ordered; take the most recent
    return {c: v for c in _cols_for_setting(rows, setting, dirs)
            if (v := _f(row.get(c))) is not None}


def signed_delta(slug: str, setting: str, baseline: dict[str, float], run: dict[str, float],
                 mls_root: str = DEFAULT_MLS_ROOT) -> float | None:
    """Mean over metric columns of the direction-signed relative Δ (run vs baseline), in %."""
    dirs = score_spec_directions(slug, mls_root)
    deltas = []
    for c, b in baseline.items():
        if c not in run:
            continue
        higher = dirs.get(c, True)
        denom = abs(b) or 1.0
        deltas.append(100.0 * ((run[c] - b) if higher else (b - run[c])) / denom)
    return sum(deltas) / len(deltas) if deltas else None


def _floats(rows: list[dict], col: str) -> list[float]:
    out = []
    for r in rows:
        v = _f(r.get(col))
        if v is not None:
            out.append(v)
    return out


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
