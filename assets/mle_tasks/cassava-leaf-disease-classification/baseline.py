"""
Baseline solution: predict the most frequent training label.

Produces submission.csv with columns: image_id, label.
"""
from pathlib import Path

import pandas as pd


data_dir = Path("data")

train = pd.read_csv(data_dir / "train.csv")
sample = pd.read_csv(data_dir / "sample_submission.csv")

majority_label = int(train["label"].mode().iloc[0])
sample["label"] = majority_label
sample.to_csv("submission.csv", index=False)
print(f"submission.csv written ({len(sample)} rows, label={majority_label})")
