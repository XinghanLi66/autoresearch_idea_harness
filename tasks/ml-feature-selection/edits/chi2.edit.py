"""Chi-squared baseline — rigorous codebase edit ops.

Scores features using the chi-squared statistic between each non-negative
feature and the target class. Measures departure from independence.

Reference: sklearn.feature_selection.chi2
Paper: Pearson (1900), "On the criterion that a given system of deviations..."
"""

_FILE = "scikit-learn/custom_featsel.py"

_CHI2 = """\
def score_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    \"\"\"Score features using chi-squared test of independence with the target.

    Computes the chi-squared statistic between each non-negative feature
    and y. Higher chi2 values indicate stronger dependence on the target.
    \"\"\"
    chi2_scores, _ = chi2(X, y)
    # Replace any NaN with 0 (can happen with zero-variance features)
    chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
    return chi2_scores
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 86,
        "end_line": 112,
        "content": _CHI2,
    },
]
