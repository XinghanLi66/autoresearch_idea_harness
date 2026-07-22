"""Extract baseline anchors from leaderboard CSV.

Reads ``tasks/<task>/leaderboard.csv`` and identifies baseline rows
(``model=baseline:*``). Provides worst/best per-metric values for use
in normalization primitives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

META_COLS = {"timestamp", "model", "is_final", "seed"}
INFORMATIONAL_PREFIXES = ("elapsed_", "n_samples", "n_prompts")


@dataclass
class MetricAnchors:
    """Baseline anchor values for a single metric column."""
    worst: float
    best: float
    values: list[float] = field(default_factory=list)


class BaselineAnchors:
    """Read-only access to baseline anchor values from a task leaderboard."""

    def __init__(self, task_dir: Path):
        self._task_dir = Path(task_dir)
        self._anchors: dict[str, MetricAnchors] = {}
        self._baseline_names: list[str] = []
        self._metric_cols: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def worst(self, metric: str) -> float | None:
        a = self._anchors.get(metric)
        return a.worst if a else None

    def best(self, metric: str) -> float | None:
        a = self._anchors.get(metric)
        return a.best if a else None

    def worst_for(self, metric: str, direction: str) -> float | None:
        """Return the raw baseline value that is worst for *direction*.

        ``worst()`` preserves the historical raw minimum API. Scoring code
        should use this helper so lower-is-better metrics floor at the raw
        maximum instead.
        """
        a = self._anchors.get(metric)
        if not a:
            return None
        if direction == "lower":
            return a.best
        return a.worst

    def best_for(self, metric: str, direction: str) -> float | None:
        """Return the raw baseline value that is best for *direction*."""
        a = self._anchors.get(metric)
        if not a:
            return None
        if direction == "lower":
            return a.worst
        return a.best

    def get(self, metric: str) -> MetricAnchors | None:
        return self._anchors.get(metric)

    def metric_columns(self) -> list[str]:
        """All scorable metric column names (excludes meta, elapsed_, *_std)."""
        return list(self._metric_cols)

    def baseline_names(self) -> list[str]:
        return list(self._baseline_names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        from mlsbench.agent.leaderboard import Leaderboard

        lb_path = self._task_dir / "leaderboard.csv"
        if not lb_path.exists():
            return

        lb = Leaderboard(lb_path)
        records = lb.all_records()
        if not records:
            return

        cfg_path = self._task_dir / "config.json"
        bl_keys: set[str] = set()
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            bl_keys = set(cfg.get("baselines", {}).keys())

        # Identify baseline rows. When config.json declares the current baseline
        # set, ignore stale baseline rows left over from older task versions.
        if bl_keys:
            baseline_rows = []
            for r in records:
                model = r.get("model")
                if not isinstance(model, str):
                    continue
                if model.startswith("baseline:") and model.removeprefix("baseline:") in bl_keys:
                    baseline_rows.append(r)
                elif model in bl_keys:
                    baseline_rows.append(r)
        else:
            baseline_rows = [
                r for r in records
                if isinstance(r.get("model"), str) and r["model"].startswith("baseline:")
            ]

        if not baseline_rows:
            return

        self._baseline_names = sorted({str(r.get("model", "")) for r in baseline_rows})

        # Prefer seed=mean rows; fallback to individual seeds
        mean_rows = [r for r in baseline_rows if r.get("seed") == "mean"]
        data_rows = mean_rows if mean_rows else baseline_rows

        # Discover metric columns. ``metric_columns()`` intentionally excludes
        # elapsed_* so autogen does not score wall time by default, but explicit
        # score_specs may still reference elapsed_*; keep anchors for those.
        anchor_cols: list[str] = []
        seen: set[str] = set()
        for r in records:
            for k in r:
                if k in seen:
                    continue
                seen.add(k)
                if k in META_COLS:
                    continue
                if k.endswith("_std"):
                    continue
                if isinstance(r[k], (int, float)):
                    anchor_cols.append(k)

        # Compute per-metric anchors from baseline data rows
        for col in anchor_cols:
            vals: list[float] = []
            for r in data_rows:
                v = r.get(col)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
                    vals.append(float(v))
            if vals:
                self._anchors[col] = MetricAnchors(
                    worst=min(vals),
                    best=max(vals),
                    values=vals,
                )

        self._metric_cols = [
            col for col in anchor_cols
            if col in self._anchors and not col.startswith(INFORMATIONAL_PREFIXES)
        ]
