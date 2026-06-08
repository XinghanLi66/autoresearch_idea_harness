#!/usr/bin/env python3
"""
MLE-bench grading helper — runs inside a worker workspace.

Called by run.sh after the agent writes submission.csv:

    python mle_grade.py \
        --competition  spooky-author-identification \
        --submission   submission.csv \
        --data-dir     /path/to/mlebench/data \
        --out-json     result.json \
        [--hmac-key    <secret>]

Writes:
    result.json  {"val_metric": <float 0-1>, "_sig": "<hmac>", "note": "..."}
    or on failure:
    result.json  {"val_metric": null, "error": "<reason>"}

The normalized score is the leaderboard percentile rank:
  - 1.0 = beats or ties the leaderboard best (1st place)
  - 0.5 = at the leaderboard median
  - 0.0 = at or below the leaderboard worst
Both raw_score and val_metric (percentile) are written to result.json.

Grading uses the OOF (out-of-fold) labels included in each prepared
competition's public split, so no internet access is required at eval time.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac as _hmac
import json
import os
import sys
from pathlib import Path


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _sign(val_metric: float, secret: str) -> str:
    payload = f"{float(val_metric):.6f}"
    return _hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def grade(
    competition_id: str,
    submission_path: Path,
    data_dir: Path,
    hmac_key: str,
) -> dict:
    """
    Run mlebench grading and return the result dict ready for result.json.
    Uses OOF grading (public split labels) — same as LoongFlow's eval_program.py.
    """
    try:
        from mlebench.data import is_dataset_prepared
        from mlebench.registry import registry
        from mlebench.utils import load_answers, read_csv
    except ImportError:
        return {
            "val_metric": None,
            "error": "mlebench not installed — activate loongflow_ml env before running",
        }

    try:
        new_registry = registry.set_data_dir(data_dir)
        competition = new_registry.get_competition(competition_id)
    except Exception as exc:
        return {"val_metric": None, "error": f"competition not found: {exc}"}

    if not is_dataset_prepared(competition, grading_only=True):
        return {
            "val_metric": None,
            "error": (
                f"competition data not prepared at {data_dir}/{competition_id}. "
                f"Run: mlebench prepare --competition-id {competition_id} --data-dir {data_dir}"
            ),
        }

    if not submission_path.exists():
        return {"val_metric": None, "error": f"submission file not found: {submission_path}"}

    try:
        submission_df = read_csv(submission_path)
    except Exception as exc:
        return {"val_metric": None, "error": f"cannot read submission CSV: {exc}"}

    # Grade against OOF (public) labels — no held-out test answers needed
    try:
        from mlebench.graders.utils import grade_csv
        answers = load_answers(competition.public_answers)
        raw_score, is_lower_better = grade_csv(competition, submission_df, answers)
    except Exception as exc:
        return {"val_metric": None, "error": f"grading failed: {exc}"}

    # Normalize score to percentile rank against leaderboard
    try:
        from mlebench.data import get_leaderboard
        import numpy as np

        leaderboard = get_leaderboard(competition)
        scores = leaderboard["score"].values.astype(float)

        if is_lower_better:
            percentile = float(np.mean(scores >= raw_score))
        else:
            percentile = float(np.mean(scores <= raw_score))

        percentile = float(max(0.0, min(1.0, percentile)))
    except Exception:
        percentile = None

    if percentile is None:
        return {
            "val_metric": None,
            "raw_score": float(raw_score),
            "error": f"could not compute percentile for raw_score={raw_score} (no leaderboard data)",
        }

    return {
        "val_metric": percentile,
        "raw_score": float(raw_score),
        "_sig": _sign(percentile, hmac_key),
        "note": f"competition={competition_id} raw_score={raw_score:.6f} percentile={percentile:.6f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--hmac-key",
                        default=os.environ.get("BENCHMARK_HMAC_KEY", "benchmark-eval-secret"))
    args = parser.parse_args()

    result = grade(
        competition_id=args.competition,
        submission_path=args.submission,
        data_dir=args.data_dir,
        hmac_key=args.hmac_key,
    )
    _write(args.out_json, result)

    if result.get("val_metric") is not None:
        print(f"[mle_grade] {args.competition}: val_metric={result['val_metric']:.4f}  "
              f"({result.get('note', '')})")
        sys.exit(0)
    else:
        print(f"[mle_grade] ERROR: {result.get('error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
