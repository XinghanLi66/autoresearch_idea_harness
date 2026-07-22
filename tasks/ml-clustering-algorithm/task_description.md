# Clustering Algorithm Design

## Research Question
Design a novel clustering algorithm or distance metric that improves cluster quality across diverse dataset geometries — including convex blobs, non-convex shapes (moons), varied-density clusters, and real-world high-dimensional data (handwritten digits).

## Background
Clustering is a fundamental unsupervised learning problem. Classic methods like K-Means assume convex, isotropic clusters; DBSCAN handles arbitrary shapes but requires careful tuning of the `eps` parameter. Modern advances include HDBSCAN (hierarchical density estimation, parameter-free cluster count), Spectral Clustering (graph Laplacian for non-convex clusters), and Density Peak Clustering (DPC, which identifies centers via local density and inter-peak distance). No single method dominates across all dataset structures, making this an open research question.

## Task
Modify the `CustomClustering` class in `scikit-learn/custom_clustering.py` (lines 36--120) to implement a novel clustering algorithm. You may also modify the `custom_distance` function if your approach uses a custom distance metric.

Your algorithm must:
- Accept `n_clusters` (int or None) and `random_state` parameters
- Implement `fit(X)` that sets `self.labels_` and returns `self`
- Implement `predict(X)` that returns integer cluster labels
- Handle datasets with different structures (convex, non-convex, varied density, high-dimensional)

## Interface
```python
class CustomClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=None, random_state=42): ...
    def fit(self, X):        # X: (n_samples, n_features) -> self
    def predict(self, X):    # X: (n_samples, n_features) -> labels (n_samples,)
```

Available imports (already in the FIXED section): `numpy`, `sklearn.base.BaseEstimator`, `sklearn.base.ClusterMixin`, `sklearn.preprocessing.StandardScaler`, `sklearn.metrics.*`. You may import any module from `scikit-learn`, `numpy`, or `scipy`.

## Evaluation
- **Datasets**: blobs (5 Gaussian clusters), moons (2 half-circles), varied_density (3 clusters with different densities), digits (sklearn Digits, 10 classes, 64 features)
- **Metrics**: ARI (Adjusted Rand Index, higher is better), NMI (Normalized Mutual Information, higher is better), Silhouette Score (higher is better)
- Success = consistently improving over baselines across all four datasets

