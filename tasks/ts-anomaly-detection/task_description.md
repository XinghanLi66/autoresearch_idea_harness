# Time Series Anomaly Detection: Custom Model Design

## Objective
Design and implement a custom deep learning model for unsupervised time series anomaly detection via reconstruction. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (DLinear, TimesNet, PatchTST) are provided as read-only.

## Evaluation
Trained and evaluated on three anomaly detection datasets:
- **PSM** (25 variables, server machine dataset)
- **MSL** (55 variables, Mars Science Laboratory)
- **SMAP** (25 variables, Soil Moisture Active Passive satellite)

All use `seq_len=100`, `anomaly_ratio=1`. Metric: F-score (higher is better).
