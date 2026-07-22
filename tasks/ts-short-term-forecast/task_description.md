# Short-Term Time Series Forecasting: Custom Model Design

## Objective
Design and implement a custom deep learning model for univariate short-term time series forecasting on the M4 dataset. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (DLinear, TimesNet, PatchTST) are provided as read-only.

## Evaluation
Trained and evaluated on three M4 seasonal patterns:
- **Monthly** (pred_len=18, seq_len=104)
- **Quarterly** (pred_len=8, seq_len=52)
- **Yearly** (pred_len=6, seq_len=42)

All use `enc_in=1`, `features=M`, `loss=SMAPE`. Metric: SMAPE (lower is better).
