# Long-Term Time Series Forecasting: Custom Model Design

## Objective
Design and implement a custom deep learning model for multivariate long-term time series forecasting. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (DLinear, PatchTST, iTransformer) are provided as read-only.

## Evaluation
Trained and evaluated on three multivariate datasets:
- **ETTh1** (7 variables, hourly electricity transformer temperature)
- **Weather** (21 variables, weather observations)
- **ECL** (321 variables, electricity consumption)

All use `seq_len=96`, `pred_len=96`. Metrics: MSE and MAE (lower is better).
