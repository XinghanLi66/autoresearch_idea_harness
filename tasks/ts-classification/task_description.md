# Time Series Classification: Custom Model Design

## Objective
Design and implement a custom deep learning model for multivariate time series classification. Your code goes in the `Model` class in `models/Custom.py`. Three reference implementations (DLinear, TimesNet, PatchTST) are provided as read-only.

## Evaluation
Trained and evaluated on three UEA datasets:
- **EthanolConcentration** — spectral data classification
- **FaceDetection** — MEG brain imaging classification
- **Handwriting** — accelerometer-based character recognition

Training uses RAdam optimizer, CrossEntropyLoss, patience=10. Metric: accuracy (higher is better).
