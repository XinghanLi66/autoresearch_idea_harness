"""F-classif (ANOVA F-value) baseline — rigorous codebase edit ops.

Scores features using the ANOVA F-value between each feature and the target.
Computes the ratio of between-class variance to within-class variance.

Reference: sklearn.feature_selection.f_classif
"""

_FILE = "scikit-learn/custom_featsel.py"

_F_CLASSIF = """\
def score_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    \"\"\"Score features using ANOVA F-value between each feature and the target.

    Computes the F-statistic (ratio of between-class to within-class variance)
    for each feature. Higher F-values indicate more discriminative features.
    \"\"\"
    f_scores, _ = f_classif(X, y)
    # Replace any NaN with 0 (can happen with zero-variance features)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    return f_scores
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 112,
        "content": _F_CLASSIF,
    },
]
