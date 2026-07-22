# Exogenous Variable Forecasting: Custom Model Design

## Objective
Design and implement a custom deep learning model for time series forecasting with exogenous (external) variables. Uses `features=MS`: all variables as input, predict only the target (last dimension). Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (DLinear, PatchTST, iTransformer) are provided as read-only.

## Evaluation
Trained and evaluated on three datasets with MS features:
- **ETTh1** (7 → 1, hourly electricity data)
- **Weather** (21 → 1, weather observations)
- **ECL** (321 → 1, electricity consumption)

All use `seq_len=96`, `pred_len=96`. Metrics: MSE and MAE on the target variable (lower is better). The framework automatically extracts `outputs[:, :, -1:]`.
