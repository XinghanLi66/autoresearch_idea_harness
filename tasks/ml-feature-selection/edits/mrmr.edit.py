"""mRMR (minimum Redundancy Maximum Relevance) baseline.

Greedy forward feature selection that balances relevance to the target
(measured by mutual information) against redundancy with already-selected
features (also measured by mutual information).

Reference: Peng, Long & Ding (2005), "Feature Selection Based on Mutual
Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy",
IEEE TPAMI 27(8):1226-1238.
"""

_FILE = "scikit-learn/custom_featsel.py"

_MRMR = """\
def score_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    \"\"\"Score features using mRMR (minimum Redundancy Maximum Relevance).

    Greedy forward selection: at each step, select the feature that
    maximizes Relevance(feature, target) - mean(Redundancy(feature, selected)).
    Relevance is measured by mutual information (k-NN estimator).
    Redundancy is approximated by absolute Pearson correlation (fast, scalable).
    The returned scores encode selection order (higher = selected earlier).
    \"\"\"
    import os
    from sklearn.feature_selection import mutual_info_classif

    n_samples, n_features = X.shape

    # Precompute relevance: MI(feature_j, y) for all features
    relevance = mutual_info_classif(X, y, discrete_features=False,
                                     random_state=SEED, n_neighbors=5)

    # Number of greedy steps = k (the number of features that will be selected).
    # Only select exactly k features, since that's all we need.
    dataset_name = os.environ.get("ENV", "")
    k_map = {"20newsgroups": 500, "mnist": 200, "madelon": 20}
    k = k_map.get(dataset_name, min(200, n_features))
    max_greedy_steps = min(k, n_features)

    # Precompute standardized feature matrix for fast Pearson correlation
    X_std = X - X.mean(axis=0)
    col_norms = np.sqrt((X_std ** 2).sum(axis=0)) + 1e-10
    X_norm = X_std / col_norms  # shape (n_samples, n_features)

    # Redundancy matrix: absolute Pearson correlation between features.
    # Use rolling update: corr_sum[j] = sum of |corr(j, s)| for all selected s.
    corr_sum = np.zeros(n_features)
    n_selected = 0

    selected = []
    selected_set = set()
    scores = np.zeros(n_features)

    for step in range(max_greedy_steps):
        # Compute mRMR scores for all candidates in one vectorized pass
        if n_selected == 0:
            mrmr_scores = relevance.copy()
        else:
            redundancy = corr_sum / n_selected
            mrmr_scores = relevance - redundancy
        # Mask already-selected features
        for s in selected_set:
            mrmr_scores[s] = -np.inf

        best_feat = int(np.argmax(mrmr_scores))
        selected.append(best_feat)
        selected_set.add(best_feat)
        scores[best_feat] = n_features - step

        # Update corr_sum: add |corr(j, best_feat)| for all j
        corr_new = np.abs(X_norm.T @ X_norm[:, best_feat])  # shape (n_features,)
        corr_sum += corr_new
        n_selected += 1

    # For unselected features, assign a small score based on relevance alone
    for j in range(n_features):
        if j not in selected_set:
            scores[j] = relevance[j] * 0.001

    return scores
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 112,
        "content": _MRMR,
    },
]
