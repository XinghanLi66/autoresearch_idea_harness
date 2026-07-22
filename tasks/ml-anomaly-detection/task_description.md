# Unsupervised Anomaly Detection Algorithm Design

## Research Question
Design a novel unsupervised anomaly detection algorithm for tabular data that generalizes across datasets with varying dimensionality, sample sizes, and anomaly ratios.

## Background
Unsupervised anomaly detection identifies rare, unusual patterns in data without labeled examples. Classic methods include Isolation Forest (tree-based isolation), Local Outlier Factor (density-based), and One-Class SVM (boundary-based). Recent advances include ECOD (empirical cumulative distribution tails, TKDE 2022), COPOD (copula-based tail probabilities, ICDM 2020), and Deep Isolation Forest (representation-enhanced isolation, TKDE 2023). Despite progress, no single method dominates across all dataset characteristics, leaving room for novel algorithmic designs that combine strengths of multiple paradigms.

## Task
Implement a custom unsupervised anomaly detection algorithm in the `CustomAnomalyDetector` class in `custom_anomaly.py`. Your algorithm should detect anomalies without using any labels during training.

## Interface
```python
class CustomAnomalyDetector:
    def __init__(self):
        # Initialize hyperparameters and internal state

    def fit(self, X):
        # Train on unlabeled data X: numpy array (n_samples, n_features)
        # Data is already standardized (zero mean, unit variance)
        return self

    def decision_function(self, X):
        # Return anomaly scores: numpy array (n_samples,)
        # Higher scores = more anomalous
        return scores
```

## Available Libraries
- `numpy`, `scipy` (linear algebra, statistics, spatial, optimization)
- `scikit-learn` (PCA, KDE, NearestNeighbors, GaussianMixture, etc.)
- `pyod` (IForest, LOF, OCSVM, ECOD, COPOD, KNN, HBOS, PCA, LODA, SUOD, etc.)

## Evaluation
Evaluated on 4 tabular anomaly detection benchmarks from ADBench/ODDS:
- **Cardio**: 1,831 samples, 21 features, ~9.6% anomalies (cardiotocography)
- **Thyroid**: 3,772 samples, 6 features, ~2.5% anomalies (thyroid disease)
- **Satellite**: 6,435 samples, 36 features, ~31.6% anomalies (Landsat satellite)
- **Shuttle**: 49,097 samples, 9 features, ~7.2% anomalies (NASA shuttle)

Metrics (higher is better): **AUROC** (area under ROC curve) and **F1** score at the optimal contamination threshold. Evaluated via a 60/40 stratified train/test split, following the standard ADBench/ECOD paper protocol.

