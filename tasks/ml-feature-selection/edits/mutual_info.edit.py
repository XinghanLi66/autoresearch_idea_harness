"""Mutual Information baseline — rigorous codebase edit ops.

Scores features using mutual information between each feature and the target.
Uses sklearn.feature_selection.mutual_info_classif which estimates MI via
k-nearest neighbors (Kraskov et al., 2004).

Reference: sklearn.feature_selection.mutual_info_classif
"""

_FILE = "scikit-learn/custom_featsel.py"

_MUTUAL_INFO = """\
def score_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    \"\"\"Score features using mutual information with the target variable.

    Uses sklearn's mutual_info_classif which estimates mutual information
    between each feature and y via k-nearest neighbor distances.
    \"\"\"
    scores = mutual_info_classif(X, y, discrete_features=False, random_state=SEED, n_neighbors=5)
    return scores
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 112,
        "content": _MUTUAL_INFO,
    },
]
