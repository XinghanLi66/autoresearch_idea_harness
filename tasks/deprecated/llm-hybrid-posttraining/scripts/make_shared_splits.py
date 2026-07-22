#!/usr/bin/env python3
"""Build one mixed OpenR1 train split and three benchmark eval splits."""

import argparse
from pathlib import Path

import pandas as pd


def _set_split_column(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    if "extra_info" not in df.columns:
        return df

    def update_extra_info(info, idx):
        if isinstance(info, dict):
            info = dict(info)
            info["split"] = split
            info.setdefault("index", f"{split}-{idx}")
            return info
        return info

    df["extra_info"] = [update_extra_info(info, i) for i, info in enumerate(df["extra_info"])]
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare shared HPT train/eval parquet files")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--aime24-input", required=True)
    parser.add_argument("--amc23-input", required=True)
    parser.add_argument("--math500-input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument(
        "--train-source-spec",
        type=str,
        default="",
        help="Comma-separated source:count pairs, e.g. amc_aime:256,aops_forum:256",
    )
    parser.add_argument(
        "--train-source-selection",
        type=str,
        default="random",
        choices=["random", "shortest_target", "shortest_proxy"],
        help="How to choose rows within each train source bucket.",
    )
    parser.add_argument("--aime24-size", type=int, default=30)
    parser.add_argument("--amc23-size", type=int, default=40)
    parser.add_argument("--math500-size", type=int, default=100)
    parser.add_argument("--internal-aime24-size", type=int, default=1)
    parser.add_argument("--internal-amc23-size", type=int, default=1)
    parser.add_argument("--internal-math500-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(args.train_input)
    if args.train_source_spec:
        parts = [p.strip() for p in args.train_source_spec.split(",") if p.strip()]
        sampled_parts = []
        for part in parts:
            if ":" not in part:
                raise ValueError(f"Invalid train source spec entry: {part}")
            source, count_str = part.split(":", 1)
            source = source.strip()
            count = int(count_str.strip())
            source_df = train_df[train_df["data_source"] == source]
            if len(source_df) < count:
                raise ValueError(f"Need at least {count} rows for source={source}, got {len(source_df)}")
            if args.train_source_selection == "random":
                chosen_df = source_df.sample(n=count, random_state=args.seed).reset_index(drop=True)
            else:
                source_df = source_df.copy()
                prompt_len = source_df["prompt"].astype(str).str.len()
                target_len = source_df["target"].astype(str).str.len()
                source_df["_prompt_key"] = source_df["prompt"].astype(str)
                source_df["_target_key"] = source_df["target"].astype(str)
                if args.train_source_selection == "shortest_target":
                    source_df["_selection_len"] = target_len
                else:
                    source_df["_selection_len"] = prompt_len + target_len
                chosen_df = (
                    source_df.sort_values(
                        by=["_selection_len", "_prompt_key", "_target_key"],
                        ascending=[True, True, True],
                        kind="mergesort",
                    )
                    .head(count)
                    .drop(columns=["_selection_len", "_prompt_key", "_target_key"])
                    .reset_index(drop=True)
                )
            sampled_parts.append(chosen_df)
        train_df = pd.concat(sampled_parts, ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    else:
        if len(train_df) < args.train_size:
            raise ValueError(f"Need at least {args.train_size} train rows, got {len(train_df)}")
        train_df = train_df.sample(n=args.train_size, random_state=args.seed).reset_index(drop=True)
    train_df = _set_split_column(train_df, "train")
    train_path = out_dir / "mixed_train.parquet"
    train_df.to_parquet(train_path, index=False)

    aime24_df = pd.read_parquet(args.aime24_input)
    amc23_df = pd.read_parquet(args.amc23_input)
    math500_df = pd.read_parquet(args.math500_input)
    if len(aime24_df) < args.aime24_size:
        raise ValueError(f"Need at least {args.aime24_size} AIME24 rows, got {len(aime24_df)}")
    if len(amc23_df) < args.amc23_size:
        raise ValueError(f"Need at least {args.amc23_size} AMC23 rows, got {len(amc23_df)}")
    if len(math500_df) < args.math500_size:
        raise ValueError(f"Need at least {args.math500_size} MATH-500 rows, got {len(math500_df)}")

    if len(aime24_df) > args.aime24_size:
        aime24_df = aime24_df.sample(n=args.aime24_size, random_state=args.seed).reset_index(drop=True)
    if len(amc23_df) > args.amc23_size:
        amc23_df = amc23_df.sample(n=args.amc23_size, random_state=args.seed).reset_index(drop=True)
    if len(math500_df) > args.math500_size:
        math500_df = math500_df.sample(n=args.math500_size, random_state=args.seed).reset_index(drop=True)
    aime24_df = _set_split_column(aime24_df, "test")
    amc23_df = _set_split_column(amc23_df, "test")
    math500_df = _set_split_column(math500_df, "test")

    aime24_path = out_dir / "AIME24_eval.parquet"
    amc23_path = out_dir / "AMC23_eval.parquet"
    math500_path = out_dir / "MATH-500_eval.parquet"
    aime24_df.to_parquet(aime24_path, index=False)
    amc23_df.to_parquet(amc23_path, index=False)
    math500_df.to_parquet(math500_path, index=False)

    aime24_df.head(args.internal_aime24_size).reset_index(drop=True).to_parquet(
        out_dir / "AIME24_internal_val.parquet", index=False
    )
    amc23_df.head(args.internal_amc23_size).reset_index(drop=True).to_parquet(
        out_dir / "AMC23_internal_val.parquet", index=False
    )
    math500_df.head(args.internal_math500_size).reset_index(drop=True).to_parquet(
        out_dir / "MATH-500_internal_val.parquet", index=False
    )

    print(
        f"Prepared shared HPT splits: train={len(train_df)} "
        f"AIME24={len(aime24_df)} AMC23={len(amc23_df)} MATH-500={len(math500_df)}",
        flush=True,
    )
    print(
        "Prepared internal val splits: "
        f"AIME24={min(len(aime24_df), args.internal_aime24_size)} "
        f"AMC23={min(len(amc23_df), args.internal_amc23_size)} "
        f"MATH-500={min(len(math500_df), args.internal_math500_size)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
