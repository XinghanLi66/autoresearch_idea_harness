# Quantitative Stock Prediction on CSI300

## Objective
Design and implement a stock prediction model that forecasts next-day returns for CSI300 stocks. Your code goes in `custom_model.py`. Three reference implementations (LightGBM, LSTM, Transformer) are provided as read-only.

## Evaluation
Signal quality: IC, ICIR, Rank IC. Portfolio (TopkDropout, top 50, drop 5): Annualized Return, Max Drawdown, Information Ratio. Evaluation is automatic via qlib's workflow.

## Workflow Configuration
`workflow_config.yaml` lines 13-25 and 31-44 are editable. This is the model plus input-adapter/preprocessor block: you may change the dataset class (e.g., to `TSDatasetH`) or processors if your model needs a different input view. Instruments, date ranges, train/valid/test splits, and evaluation settings are fixed.
