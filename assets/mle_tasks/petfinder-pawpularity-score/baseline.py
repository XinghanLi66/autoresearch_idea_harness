"""
Baseline solution: predict the training-set mean Pawpularity.

Produces submission.csv with columns: Id, Pawpularity.
"""
from pathlib import Path

import pandas as pd


data_dir = Path("data")

train = pd.read_csv(data_dir / "train.csv")
sample = pd.read_csv(data_dir / "sample_submission.csv")

mean_target = float(train["Pawpularity"].mean())
sample["Pawpularity"] = mean_target
sample.to_csv("submission.csv", index=False)
print(f"submission.csv written ({len(sample)} rows, Pawpularity={mean_target:.3f})")
