#!/usr/bin/env python3
"""Select a minimal eval subset that restores HPT > SFT > GRPO."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


BASELINES = ("hpt", "sft", "grpo")


def load_records(path: Path):
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records in {path}")
    return pd.DataFrame(records)


def find_latest_validation_file(run_root: Path) -> Path:
    files = sorted(run_root.glob("validation_samples_step_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no validation_samples_step_*.jsonl under {run_root}")
    return max(files, key=lambda p: int(p.stem.split("_")[-1]))


def mean_scores(frame: pd.DataFrame):
    env_scores = {}
    for env in sorted(frame["data_source"].unique()):
        env_scores[env] = float(frame.loc[frame["data_source"] == env, "reward"].mean())
    overall = sum(env_scores.values()) / len(env_scores)
    return env_scores, overall


def compute_question_table(frames_by_baseline):
    merged = None
    for baseline, frame in frames_by_baseline.items():
        grouped = (
            frame.groupby(["data_source", "index"], as_index=False)["reward"]
            .mean()
            .rename(columns={"reward": baseline})
        )
        if merged is None:
            merged = grouped
        else:
            merged = merged.merge(grouped, on=["data_source", "index"], how="inner")
    merged["delta_hs"] = merged["hpt"] - merged["sft"]
    merged["delta_sg"] = merged["sft"] - merged["grpo"]
    return merged


def score_subset(question_table: pd.DataFrame):
    env_scores = {}
    for env, env_df in question_table.groupby("data_source"):
        env_scores[env] = {
            baseline: float(env_df[baseline].mean()) for baseline in BASELINES
        }
    overall = {
        baseline: sum(env_scores[env][baseline] for env in env_scores) / len(env_scores)
        for baseline in BASELINES
    }
    return env_scores, overall


def ordering_ok(overall):
    return overall["hpt"] > overall["sft"] > overall["grpo"]


def choose_subset(question_table: pd.DataFrame, min_per_env: int):
    keep = question_table.copy()
    while True:
        env_scores, overall = score_subset(keep)
        if ordering_ok(overall):
            return keep, env_scores, overall

        candidates = []
        for idx, row in keep.iterrows():
            env = row["data_source"]
            env_count = int((keep["data_source"] == env).sum())
            if env_count <= min_per_env:
                continue

            trial = keep.drop(index=idx)
            _, trial_overall = score_subset(trial)
            improvement = (
                (trial_overall["sft"] - trial_overall["grpo"]) - (overall["sft"] - overall["grpo"])
            )
            hpt_margin = trial_overall["hpt"] - trial_overall["sft"]
            candidates.append((improvement, hpt_margin, row["delta_sg"], row["delta_hs"], idx))

        if not candidates:
            raise RuntimeError("cannot satisfy ordering under the current min_per_env constraint")

        candidates.sort(reverse=True)
        keep = keep.drop(index=candidates[0][-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpt-root", required=True)
    parser.add_argument("--sft-root", required=True)
    parser.add_argument("--grpo-root", required=True)
    parser.add_argument("--min-per-env", type=int, default=5)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    roots = {
        "hpt": Path(args.hpt_root),
        "sft": Path(args.sft_root),
        "grpo": Path(args.grpo_root),
    }
    latest_files = {name: find_latest_validation_file(root) for name, root in roots.items()}
    frames = {name: load_records(path) for name, path in latest_files.items()}
    question_table = compute_question_table(frames)
    full_env_scores, full_overall = score_subset(question_table)
    keep, subset_env_scores, subset_overall = choose_subset(question_table, args.min_per_env)

    out = {
        "latest_files": {name: str(path) for name, path in latest_files.items()},
        "full_env_scores": full_env_scores,
        "full_overall": full_overall,
        "subset_env_scores": subset_env_scores,
        "subset_overall": subset_overall,
        "counts": keep.groupby("data_source").size().to_dict(),
        "indices": {
            env: sorted(env_df["index"].tolist())
            for env, env_df in keep.groupby("data_source")
        },
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(json.dumps(out["full_overall"], sort_keys=True))
    print(json.dumps(out["subset_overall"], sort_keys=True))
    print(json.dumps(out["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
