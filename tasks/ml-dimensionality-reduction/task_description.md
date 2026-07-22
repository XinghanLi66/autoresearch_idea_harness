# Dimensionality Reduction: Nonlinear Embedding Method Design

## Research Question
Design a novel nonlinear dimensionality reduction method that preserves data structure (both local neighborhoods and global relationships) better than existing methods when embedding high-dimensional data into 2D.

## Background
Dimensionality reduction is fundamental to data analysis and visualization. PCA provides a fast linear baseline but cannot capture nonlinear manifold structure. Other methods trade off local and global structure preservation in different ways. This task evaluates dimensionality reduction methods by neighborhood preservation across diverse data types.

## Task
Modify the `CustomDimReduction` class (lines 14-70) in `custom_dimred.py` to implement a novel nonlinear dimensionality reduction algorithm. Your implementation must:

1. Accept high-dimensional data X of shape (n_samples, n_features) where n_samples <= 5000 and n_features ranges from 50 to 784.
2. Return a 2D embedding of shape (n_samples, 2).
3. Respect the `random_state` parameter for reproducibility.
4. Complete within a reasonable time (under 5 minutes per dataset on CPU).

You may use numpy, scipy, and scikit-learn utilities (already installed). The method is evaluated on three diverse datasets: MNIST (digit images), Fashion-MNIST (clothing images), and 20 Newsgroups (text, pre-processed to 50D via TF-IDF + SVD).

## Interface
```python
class CustomDimReduction:
    def __init__(self, n_components: int = 2, random_state: int | None = None):
        ...
    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # X: (n_samples, n_features), returns: (n_samples, n_components)
        ...
```

## Evaluation
Three metrics are computed on each dataset (k=7 neighbors):
- **kNN accuracy**: Classification accuracy of a 7-NN classifier in the 2D space (higher is better). Measures how well class structure is preserved.
- **Trustworthiness**: Whether points that are neighbors in the embedding are also neighbors in the original space (higher is better, max 1.0).
- **Continuity**: Whether points that are neighbors in the original space remain neighbors in the embedding (higher is better, max 1.0).

Success means improving on existing methods across all three datasets and all three metrics.
