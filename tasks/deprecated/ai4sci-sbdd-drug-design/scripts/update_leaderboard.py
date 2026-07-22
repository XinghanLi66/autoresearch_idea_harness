#!/usr/bin/env python3
"""
Update the ai4sci-sbdd-drug-design leaderboard with linker/frag eval-only results.

Scans logs/ai4sci-sbdd-drug-design/{baseline}/eval_only/{task}_s42.out
for TEST_METRICS lines and merges them into the leaderboard CSV.

Usage:
    python3 tasks/ai4sci-sbdd-drug-design/scripts/update_leaderboard.py
"""

import os
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/scratch/gpfs/CHIJ/st3812/projects/MLS-Bench")
LOGS_DIR = PROJECT_ROOT / "logs" / "ai4sci-sbdd-drug-design"
LEADERBOARD = PROJECT_ROOT / "tasks" / "ai4sci-sbdd-drug-design" / "leaderboard.csv"

BASELINES = ["targetdiff", "diffbp", "pocket2mol"]
TASKS = ["linker", "frag"]
SEED = 42


def parse_test_metrics(log_path: Path) -> dict[str, float]:
    """Extract TEST_METRICS key=value pairs from a log file."""
    metrics = {}
    if not log_path.is_file():
        return metrics
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("TEST_METRICS "):
            continue
        rest = line[len("TEST_METRICS "):].strip()
        m = re.match(r"(\w+)=([\d.eE+\-]+)", rest)
        if m:
            name, value = m.group(1), m.group(2)
            try:
                metrics[name] = float(value)
            except ValueError:
                pass
    return metrics


def get_elapsed_seconds(log_path: Path) -> int | None:
    """Try to get elapsed time from the log file.

    Looks for SLURM-style elapsed or computes from first/last timestamp.
    Returns seconds or None.
    """
    if not log_path.is_file():
        return None
    text = log_path.read_text(errors="replace")

    # Check for explicit elapsed marker
    m = re.search(r"TEST_METRICS\s+elapsed=(\d+)", text)
    if m:
        return int(m.group(1))

    # Try to get file modification time minus creation/first-write time
    # as a rough proxy for job runtime
    stat = log_path.stat()
    if stat.st_size > 0:
        # Use mtime - ctime as approximate elapsed
        elapsed = int(stat.st_mtime - stat.st_ctime)
        if elapsed > 0:
            return elapsed

    return None


def main():
    print(f"Reading leaderboard: {LEADERBOARD}")
    df = pd.read_csv(LEADERBOARD)
    print(f"Current columns: {list(df.columns)}")
    print(f"Current rows: {len(df)}")

    updated = []
    skipped = []

    for baseline in BASELINES:
        model_name = f"baseline:{baseline}"
        row_mask = df["model"] == model_name

        if not row_mask.any():
            print(f"  WARNING: No row found for {model_name} in leaderboard, skipping")
            skipped.append((baseline, "all", "no row in leaderboard"))
            continue

        for task in TASKS:
            log_path = LOGS_DIR / baseline / "eval_only" / f"{task}_s{SEED}.out"

            if not log_path.is_file():
                print(f"  SKIP {baseline}/{task}: {log_path} not found (job may still be running)")
                skipped.append((baseline, task, "log file not found"))
                continue

            metrics = parse_test_metrics(log_path)
            if not metrics:
                print(f"  SKIP {baseline}/{task}: no TEST_METRICS found in {log_path}")
                skipped.append((baseline, task, "no TEST_METRICS in log"))
                continue

            print(f"  FOUND {baseline}/{task}: {metrics}")

            # Add metrics as columns with task suffix
            for metric_name, value in metrics.items():
                col_name = f"{metric_name}_{task}"
                df.loc[row_mask, col_name] = value

            # Try to get elapsed time
            elapsed = get_elapsed_seconds(log_path)
            if elapsed is not None:
                col_name = f"elapsed_{task}"
                df.loc[row_mask, col_name] = elapsed
                print(f"    elapsed_{task} = {elapsed}s")

            updated.append((baseline, task))

    if not updated:
        print("\nNo updates to apply. All logs are missing or empty.")
        return

    # Write back
    df.to_csv(LEADERBOARD, index=False)
    print(f"\nUpdated leaderboard written to {LEADERBOARD}")
    print(f"  Updated: {updated}")
    if skipped:
        print(f"  Skipped: {skipped}")

    # Print final state
    print(f"\nFinal columns: {list(df.columns)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
