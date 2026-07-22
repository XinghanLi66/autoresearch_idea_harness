# Time Series Imputation: Custom Model Design

## Objective
Design and implement a custom deep learning model for time series missing value imputation. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (DLinear, TimesNet, PatchTST) are provided as read-only.

## Evaluation
Trained and evaluated on three multivariate datasets with 25% random masking:
- **ETTh1** (7 variables)
- **Weather** (21 variables)
- **ECL** (321 variables)

All use `seq_len=96`. Metrics: MSE and MAE on masked regions only (lower is better).
