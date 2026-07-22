# Custom feature selection method for MLS-Bench
#
# EDITABLE section: score_features() function.
# FIXED sections: everything else (data loading, classifier, evaluation).
import os
import warnings
import numpy as np
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups, fetch_openml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, chi2, f_classif

warnings.filterwarnings("ignore")


# =====================================================================
# FIXED: Configuration
# =====================================================================
SEED = int(os.environ.get("SEED", "42"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
DATASET_NAME = os.environ.get("ENV", "20newsgroups")
DATA_HOME = os.environ.get("SKLEARN_DATA_HOME", "/data/sklearn")

# Number of top features to select per dataset
TOP_K_MAP = {
    "20newsgroups": 500,
    "mnist": 200,
    "madelon": 20,
}


# =====================================================================
# FIXED: Dataset loading
# =====================================================================
def load_dataset(name: str):
    """Load a dataset and return (X_train, y_train, X_test, y_test).

    All features are numeric. X is a dense numpy array of shape (n_samples, n_features).
    y is an integer numpy array of class labels.
    """
    if name == "20newsgroups":
        # Text classification: 20 newsgroups with TF-IDF features
        train_data = fetch_20newsgroups(
            subset="train", data_home=DATA_HOME, remove=("headers", "footers", "quotes")
        )
        test_data = fetch_20newsgroups(
            subset="test", data_home=DATA_HOME, remove=("headers", "footers", "quotes")
        )
        vectorizer = TfidfVectorizer(max_features=10000, dtype=np.float64)
        X_train = vectorizer.fit_transform(train_data.data).toarray()
        X_test = vectorizer.transform(test_data.data).toarray()
        y_train = train_data.target
        y_test = test_data.target
        return X_train, y_train, X_test, y_test

    elif name == "mnist":
        # Handwritten digit recognition: 784 pixel features
        mnist = fetch_openml("mnist_784", version=1, data_home=DATA_HOME, parser="auto")
        X = mnist.data.to_numpy().astype(np.float64)
        y = mnist.target.to_numpy().astype(int)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx, test_idx = next(sss.split(X, y))
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    elif name == "madelon":
        # Artificial dataset with 500 features (20 informative, 480 noisy)
        madelon = fetch_openml("madelon", version=1, data_home=DATA_HOME, parser="auto")
        X = madelon.data.to_numpy().astype(np.float64)
        y = madelon.target.to_numpy().astype(int)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
        train_idx, test_idx = next(sss.split(X, y))
        return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

    else:
        raise ValueError(f"Unknown dataset: {name}")


# =====================================================================
# EDITABLE: Custom feature scoring function (lines 86-112)
# =====================================================================
def score_features(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Score each feature for its relevance to the classification target.

    This function receives the training data and must return a score for
    each feature indicating its importance. Higher scores = more important.
    The top-k features (by score) will be selected for classification.

    Args:
        X: Training feature matrix of shape (n_samples, n_features).
           All values are non-negative floats (TF-IDF, pixel intensities,
           or pre-processed numeric features depending on the dataset).
        y: Training labels of shape (n_samples,), integer class labels.

    Returns:
        scores: 1-D numpy array of shape (n_features,) with non-negative
                importance scores. Features with higher scores are selected.

    Notes:
        - X values are guaranteed non-negative (suitable for chi2, etc.)
        - Must handle both binary and multi-class problems.
        - Must handle high-dimensional data (up to 10000 features).
        - Returned scores must be finite (no NaN or Inf).
        - Available imports: numpy, scipy, sklearn (see top of file).
    """
    # Default: variance-based scoring (baseline placeholder)
    scores = np.var(X, axis=0)
    return scores


# =====================================================================
# FIXED: Feature selection and evaluation pipeline
# =====================================================================
def select_and_evaluate(X_train, y_train, X_test, y_test, k, seed):
    """Select top-k features using score_features, then train and evaluate a classifier."""
    # Score features on training data
    print(f"TRAIN_METRICS step=scoring dataset={DATASET_NAME} n_features={X_train.shape[1]} k={k}", flush=True)

    scores = score_features(X_train, y_train)
    scores = np.asarray(scores, dtype=np.float64).ravel()

    # Validate scores
    assert scores.shape[0] == X_train.shape[1], (
        f"score_features returned {scores.shape[0]} scores but expected {X_train.shape[1]}"
    )
    # Replace any NaN/Inf with 0
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Select top-k features
    top_k_indices = np.argsort(scores)[::-1][:k]
    top_k_indices = np.sort(top_k_indices)  # keep original feature order

    X_train_sel = X_train[:, top_k_indices]
    X_test_sel = X_test[:, top_k_indices]

    print(f"TRAIN_METRICS step=selected dataset={DATASET_NAME} top_k={len(top_k_indices)}", flush=True)

    # Standardize selected features
    scaler = StandardScaler()
    X_train_sel = scaler.fit_transform(X_train_sel)
    X_test_sel = scaler.transform(X_test_sel)

    # Train LogisticRegression classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
        C=1.0,
        multi_class="multinomial" if len(np.unique(y_train)) > 2 else "auto",
    )
    clf.fit(X_train_sel, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train_sel)
    y_pred_test = clf.predict(X_test_sel)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"TRAIN_METRICS step=eval dataset={DATASET_NAME} train_acc={train_acc:.4f} test_acc={test_acc:.4f}", flush=True)

    return test_acc


# =====================================================================
# FIXED: Main entry point
# =====================================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    k = TOP_K_MAP.get(DATASET_NAME, 100)
    print(f"Dataset: {DATASET_NAME}, Seed: {SEED}, Top-K: {k}", flush=True)

    # Load data
    X_train, y_train, X_test, y_test = load_dataset(DATASET_NAME)
    print(f"Loaded {DATASET_NAME}: train={X_train.shape}, test={X_test.shape}, classes={len(np.unique(y_train))}", flush=True)

    # Ensure non-negative features for compatibility with chi2 etc.
    min_vals = X_train.min(axis=0)
    shift = np.where(min_vals < 0, -min_vals, 0.0)
    X_train = X_train + shift
    X_test = X_test + shift

    # Run feature selection + evaluation
    test_acc = select_and_evaluate(X_train, y_train, X_test, y_test, k, SEED)

    print(f"TEST_METRICS accuracy={test_acc:.4f}", flush=True)
    print(f"Final test accuracy on {DATASET_NAME}: {100 * test_acc:.2f}%", flush=True)
