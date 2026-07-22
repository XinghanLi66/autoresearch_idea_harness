# Concept Drift Adaptation in Stock Prediction on CSI300

## Objective
Design and implement a stock prediction model that handles concept drift and temporal distribution shift in CSI300 stocks. Your code goes in `custom_model.py`. Three reference implementations (TRA, AdaRNN, LightGBM) are provided as read-only.

## Evaluation
Signal quality: IC, ICIR, Rank IC. Portfolio (TopkDropout, top 50, drop 5): Annualized Return, Max Drawdown, Information Ratio. Automatic via qlib's workflow.

Evaluation uses three fixed temporal regimes on the same CSI300 universe:
- `csi300`: long-horizon split ending in the 2017-2020 regime
- `csi300_shifted`: shifted split with a 2016-2018 test regime
- `csi300_recent` (hidden): the most recent 2019-2020 regime

This task is about temporal drift adaptation, not cross-universe transfer.

## Workflow Configuration
`workflow_config.yaml` lines 13-26 and 32-45 are editable. This covers the model plus dataset adapter/processor configuration needed by methods like TRA. Instruments, date ranges, train/valid/test splits, and evaluation settings are fixed.
