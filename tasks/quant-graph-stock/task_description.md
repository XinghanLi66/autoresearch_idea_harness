# Graph-Based Multi-Stock Prediction on CSI300

## Objective
Design and implement a graph-based stock prediction model that leverages inter-stock relationships through a stock-concept graph. Your code goes in `custom_model.py`. Three reference implementations (HIST, GATs, LightGBM) are provided as read-only.

## Evaluation
Signal quality: IC, ICIR, Rank IC. Portfolio (TopkDropout, top 50, drop 5): Annualized Return, Max Drawdown, Information Ratio. Automatic via qlib's workflow.

## Workflow Configuration
`workflow_config.yaml` lines 14-26 and 32-45 are editable. This covers the model plus dataset adapter/preprocessor configuration. Instruments, date ranges, train/valid/test splits, and evaluation settings are fixed.
