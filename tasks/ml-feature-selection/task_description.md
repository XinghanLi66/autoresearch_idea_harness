# Feature Selection Method Design

## Research Question
Design a novel univariate feature scoring method that identifies the most informative features for classification, generalizing across diverse data modalities (text, vision, tabular).

## Background
Feature selection is a fundamental preprocessing step in machine learning. By removing irrelevant or redundant features, it can improve model accuracy, reduce overfitting, and speed up training. Classical univariate methods score each feature independently based on its relationship with the target variable:

- **Chi-squared test**: Measures departure from independence between feature and target using contingency tables. Works best with non-negative, count-like features.
- **ANOVA F-value (f_classif)**: Computes the ratio of between-class variance to within-class variance. Effective for normally-distributed features with different means per class.
- **Mutual Information**: Estimates the mutual information between each feature and the target via k-nearest neighbors. Captures non-linear dependencies but is computationally expensive.

Each method has strengths and weaknesses depending on the data distribution. The task is to design a scoring function that performs robustly across different data types and class structures.

## Task
Implement the `score_features(X, y)` function in `custom_featsel.py`. Given a training feature matrix `X` and integer class labels `y`, return a 1-D numpy array of non-negative importance scores (one per feature). The top-k features (by score) will be selected and used to train a LogisticRegression classifier.

## Interface
```python
def score_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Args:
        X: (n_samples, n_features) non-negative float array
        y: (n_samples,) integer class labels

    Returns:
        scores: (n_features,) non-negative float array
    """
```

**Available imports** (already at top of file): `numpy`, `scipy` (via sklearn), `sklearn.feature_selection` (mutual_info_classif, chi2, f_classif), `sklearn.preprocessing`, `sklearn.metrics`.

## Evaluation
Evaluated on three classification benchmarks spanning different data modalities:
- **20newsgroups**: 10,000 TF-IDF text features, 20 classes, top-500 selected
- **MNIST**: 784 pixel intensity features, 10 digit classes, top-200 selected
- **Madelon**: 500 synthetic features (20 informative + 480 noisy), binary classification, top-20 selected

Metric: test classification accuracy using LogisticRegression on the selected features (higher is better).
