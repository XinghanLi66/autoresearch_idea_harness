#!/usr/bin/env python3
"""Create deterministic source-specific train/val splits from OpenR1."""

import argparse
from pathlib import Path

import pandas as pd


def _set_split_column(df: pd.DataFrame, split: str) -> pd.DataFrame:
    df = df.copy()
    if "extra_info" not in df.columns:
        return df

    def update_extra_info(info):
        if isinstance(info, dict):
            info = dict(info)
            info["split"] = split
            return info
        return info

    df["extra_info"] = df["extra_info"].map(update_extra_info)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build a source-specific OpenR1 split")
    parser.add_argument("--input", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    if "data_source" not in df.columns:
        raise ValueError("Input parquet must contain a data_source column")

    source_df = df[df["data_source"] == args.source].copy()
    required = args.train_size + args.val_size
    if len(source_df) < required:
        raise ValueError(
            f"Source '{args.source}' only has {len(source_df)} rows; need at least {required}"
        )

    source_df = source_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = _set_split_column(source_df.iloc[: args.val_size], "test")
    train_df = _set_split_column(source_df.iloc[args.val_size : required], "train")

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    print(
        f"Prepared source split {args.source}: train={len(train_df)} val={len(val_df)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
