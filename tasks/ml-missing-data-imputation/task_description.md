# Missing Data Imputation

## Research Question
Design a novel missing data imputation method that achieves low reconstruction error and preserves downstream predictive performance across diverse tabular datasets.

## Background
Missing data is ubiquitous in real-world datasets. Simple approaches like mean/median imputation ignore feature correlations, while iterative predictive methods can capture them more directly. This task evaluates imputation methods that:
- Captures complex inter-feature dependencies
- Works well on datasets of varying sizes and feature types
- Produces imputations that preserve the statistical structure needed for downstream tasks

## Task
Implement a custom imputation algorithm in the `CustomImputer` class in `custom_imputation.py`. The class follows the scikit-learn transformer interface: `fit(X)` learns from data with missing values (NaN), and `transform(X)` returns a complete matrix with no NaN values.

## Interface
```python
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, max_iter=10):
        ...

    def fit(self, X, y=None):
        # X: numpy array (n_samples, n_features) with NaN for missing values
        # Learn imputation model
        return self

    def transform(self, X):
        # X: numpy array (n_samples, n_features) with NaN for missing values
        # Return: numpy array (n_samples, n_features) with NO NaN values
        return X_imputed
```

Available libraries: numpy, scipy, scikit-learn (all submodules including sklearn.impute, sklearn.ensemble, sklearn.neighbors, etc.).

## Evaluation
Evaluated on three datasets with 20% MCAR (Missing Completely At Random) missing values:
- **Breast Cancer Wisconsin** (569 samples, 30 features, binary classification)
- **Wine** (178 samples, 13 features, 3-class classification)
- **California Housing** (5000 samples, 8 features, regression)

Two metrics per dataset:
- **RMSE**: Root Mean Squared Error between imputed and true values (lower is better)
- **downstream_score**: Classification accuracy (breast_cancer, wine) or R^2 (california) using GradientBoosting on the imputed data (higher is better)
