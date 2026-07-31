#!/usr/bin/env python3
"""Shared helpers for the MLAgentBench (MAB) eval driver.

MAB has NO leaderboard.csv and NO native scoring CLI (unlike MLS-Bench). We therefore:
  * read task metadata (metric column + direction + editable file) from docs/eval/mab_tasks.json;
  * import each task's scripts/eval.py::get_score(submission_folder) via importlib to score a workspace's
    submission.csv (get_score returns the raw metric float — accuracy / test_acc / mae);
  * keep our OWN per-task results sink at runs/researcher_cot/mab_eval/results/<task>.csv
    (columns: timestamp,model,seed,metric_value) since there is no upstream leaderboard.

get_score has cwd- and __file__-relative data dependencies:
  * cifar10 downloads CIFAR10 test into ./data (cwd-relative);
  * ogbn-arxiv reads ./networks (cwd-relative);
  * imdb reads the HF `imdb` dataset;
  * house-price / spaceship-titanic read scripts/answer.csv (dirname(__file__)-relative).
So we chdir into the scored workspace (a copy of env/, which carries the prepared data/ and networks/)
before calling get_score, and load eval.py from the REAL scripts dir so answer.csv still resolves.
"""
from __future__ import annotations
import csv
import importlib.util
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAB_ROOT = os.environ.get("MAB_ROOT", "/newcpfs/lxh/MLAgentBench")
TASKS_JSON = HARNESS_ROOT / "docs" / "eval" / "mab_tasks.json"
RESULTS_DIR = HARNESS_ROOT / "runs" / "researcher_cot" / "mab_eval" / "results"
RESULTS_COLS = ["timestamp", "model", "seed", "metric_value"]


def bench_dir(mab_root: str = DEFAULT_MAB_ROOT) -> Path:
    return Path(mab_root) / "MLAgentBench" / "benchmarks"


def task_dir(slug: str, mab_root: str = DEFAULT_MAB_ROOT) -> Path:
    return bench_dir(mab_root) / slug


def load_tasks() -> dict:
    return json.loads(TASKS_JSON.read_text())


def task_spec(slug: str) -> dict:
    return load_tasks()["tasks"][slug]


def metric_name(slug: str) -> str:
    return task_spec(slug).get("metric", "score")


def lower_is_better(slug: str) -> bool:
    return bool(task_spec(slug).get("lower_is_better", False))


def edit_file(slug: str) -> str:
    return task_spec(slug).get("edit_file", "train.py")


def baseline_mode(slug: str) -> str:
    """'run' = starter train.py is runnable as the baseline; 'worker_fill' = starter is a skeleton,
    so the baseline = worker fills a standard/naive solution (no novel idea)."""
    return task_spec(slug).get("baseline_mode", "run")


# ---------------------------------------------------------------------------
# Scoring (importlib on the task's scripts/eval.py)
# ---------------------------------------------------------------------------

@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def get_score(slug: str, submission_folder: str | os.PathLike,
              mab_root: str = DEFAULT_MAB_ROOT) -> float:
    """Import the task's scripts/eval.py and score the workspace's submission.csv.

    submission_folder must contain submission.csv (a copy of env/ that train.py wrote into). We chdir into
    it so cwd-relative eval data (./data, ./networks) resolves against the copied workspace, and load the
    module from the real scripts dir so dirname(__file__)-relative files (answer.csv) still resolve.
    """
    submission_folder = Path(submission_folder).resolve()
    eval_py = task_dir(slug, mab_root) / "scripts" / "eval.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"no eval.py for task {slug}: {eval_py}")
    spec = importlib.util.spec_from_file_location(f"mab_eval_{slug.replace('-', '_')}", eval_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    with _chdir(submission_folder):
        return float(module.get_score(str(submission_folder)))


# ---------------------------------------------------------------------------
# Δ-over-baseline (signed by direction)
# ---------------------------------------------------------------------------

def signed_delta_pct(slug: str, baseline: float, run: float) -> float | None:
    """Direction-signed relative Δ (run vs baseline), in %. Positive = improvement.
    higher-is-better: (run-base)/|base|; lower-is-better (mae): (base-run)/|base|."""
    if baseline is None or run is None:
        return None
    denom = abs(baseline) or 1.0
    improved = (run - baseline) if not lower_is_better(slug) else (baseline - run)
    return 100.0 * improved / denom


# ---------------------------------------------------------------------------
# Per-task results sink (our own; MAB has no leaderboard.csv)
# ---------------------------------------------------------------------------

def results_path(slug: str, results_dir: str | os.PathLike | None = None) -> Path:
    d = Path(results_dir) if results_dir else RESULTS_DIR
    return d / f"{slug}.csv"


def append_result(slug: str, model: str, seed, metric_value: float,
                  results_dir: str | os.PathLike | None = None) -> Path:
    """Append one (timestamp,model,seed,metric_value) row to results/<task>.csv."""
    p = results_path(slug, results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model, "seed": ("" if seed is None else seed),
                    "metric_value": metric_value})
    return p


def read_results(slug: str, results_dir: str | os.PathLike | None = None) -> list[dict]:
    p = results_path(slug, results_dir)
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
