#!/usr/bin/env python3
"""
Creativity reward model: regress opus-4.8 judge's creativity (and non_triviality)
scores from CoT text, using sentence-transformer embeddings + Ridge regression.

Usage:
    python scripts/build_creativity_rm.py           # train + eval + save
    python scripts/build_creativity_rm.py --demo    # also run on a few samples
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
COTS_PATH  = REPO_ROOT / "runs/researcher_cot/cots/full_cots.jsonl"
SCORE_DIR  = REPO_ROOT / "runs/researcher_cot/score_cache"
OUTPUT_DIR = REPO_ROOT / "runs/reward_models"
MODEL_PATH = OUTPUT_DIR / "creativity_rm.joblib"
REPORT_PATH = OUTPUT_DIR / "creativity_rm_report.json"

EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Reuse masking logic from fingerprint script (or inline minimal version)
import re

def mask_name(text: str, researcher: str) -> str:
    """Strip Mocking header and redact researcher name tokens."""
    text = re.sub(r"^\*\*Mocking:\*\*[^\n]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Mocking:[^\n]*\n?", "", text, flags=re.IGNORECASE)

    STOPWORDS = {"de", "van", "von", "le", "la", "el", "di", "do", "da"}
    tokens = []
    cjk_block = re.sub(r"[^一-鿿㐀-䶿 0-⩭f]", "", researcher)
    if cjk_block:
        tokens.append(cjk_block)
    latin = re.sub(r"[一-鿿㐀-䶿 0-⩭f]", "", researcher)
    for tok in re.split(r"[\s\-\(\)\.]+", latin):
        tok = tok.strip()
        if len(tok) > 1 and tok.lower() not in STOPWORDS:
            tokens.append(tok)
    for tok in tokens:
        text = re.sub(r"\b" + re.escape(tok) + r"\b", "[RESEARCHER]", text, flags=re.IGNORECASE)
    return text


# ── embedder ──────────────────────────────────────────────────────────────────

def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading {EMBEDDER_MODEL} on CPU …", flush=True)
        model = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
        print("[embedder] Loaded sentence-transformer.", flush=True)
        return ("sentence-transformer", model)
    except Exception as e:
        warnings.warn(f"sentence-transformers unavailable ({e}), falling back to TF-IDF.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        return ("tfidf", vec)


def embed(embedder_info, texts: list, batch_size: int = 256, fit: bool = False):
    kind, model = embedder_info
    if kind == "sentence-transformer":
        return model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, device="cpu",
        )
    else:
        if fit:
            return model.fit_transform(texts)
        return model.transform(texts)


# ── data ──────────────────────────────────────────────────────────────────────

def load_data():
    # Index score cache by key
    score_keys = set(os.listdir(SCORE_DIR))

    texts, creativity, non_triviality = [], [], []
    missing_score = 0

    with open(COTS_PATH) as f:
        for line in f:
            d = json.loads(line)
            cot = d.get("cot", "") or ""
            if not cot.strip():
                continue

            key = f"{d['case_shortcut_id']}__{d['route']}.json"
            if key not in score_keys:
                missing_score += 1
                continue

            score_path = SCORE_DIR / key
            with open(score_path) as sf:
                scores = json.load(sf)

            # Get scalar scores; skip if either target is missing
            cr  = scores.get("creativity")
            nt  = scores.get("non_triviality")
            if cr is None or nt is None:
                missing_score += 1
                continue

            masked = mask_name(cot, d.get("researcher", ""))
            texts.append(masked)
            creativity.append(float(cr))
            non_triviality.append(float(nt))

    print(f"[data] Loaded {len(texts)} CoTs with scores. Skipped: {missing_score} (no score).", flush=True)
    creativity   = np.array(creativity,    dtype=float)
    non_triviality = np.array(non_triviality, dtype=float)

    # Combined target: mean of creativity + non_triviality
    combined = (creativity + non_triviality) / 2.0

    print(f"[data] creativity:     mean={creativity.mean():.3f}  std={creativity.std():.3f}  range=[{creativity.min():.0f},{creativity.max():.0f}]")
    print(f"[data] non_triviality: mean={non_triviality.mean():.3f}  std={non_triviality.std():.3f}  range=[{non_triviality.min():.0f},{non_triviality.max():.0f}]")
    print(f"[data] combined:       mean={combined.mean():.3f}  std={combined.std():.3f}")
    return texts, creativity, non_triviality, combined


# ── training ──────────────────────────────────────────────────────────────────

def evaluate_regression(y_true, y_pred, name: str):
    from scipy.stats import pearsonr, spearmanr
    mae = float(np.mean(np.abs(y_pred - y_true)))
    pearson,  _  = pearsonr(y_true, y_pred)
    spearman, _  = spearmanr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    print(f"  [{name}] Pearson={pearson:.4f}  Spearman={spearman:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"pearson": round(pearson, 4), "spearman": round(spearman, 4),
            "mae": round(mae, 4), "rmse": round(rmse, 4)}


def train_ridge(X_train, y_train, X_test, y_test, alpha: float = 1.0, name: str = ""):
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred, name)
    return ridge, metrics, y_pred


def train_and_evaluate(texts, creativity, non_triviality, combined, embedder_info,
                       test_size: float = 0.15, seed: int = 42):
    from sklearn.model_selection import train_test_split

    n = len(texts)
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed)

    train_texts = [texts[i] for i in train_idx]
    test_texts  = [texts[i] for i in test_idx]

    print(f"[split] Train: {len(train_texts)}, Test: {len(test_texts)}", flush=True)

    print("[embed] Embedding training set …", flush=True)
    X_train = embed(embedder_info, train_texts, fit=True)
    print("[embed] Embedding test set …", flush=True)
    X_test  = embed(embedder_info, test_texts, fit=False)

    print("\n[train] Fitting Ridge regressors …")

    ridge_cr, metrics_cr, _ = train_ridge(
        X_train, creativity[train_idx],
        X_test,  creativity[test_idx],
        alpha=1.0, name="creativity",
    )
    ridge_nt, metrics_nt, _ = train_ridge(
        X_train, non_triviality[train_idx],
        X_test,  non_triviality[test_idx],
        alpha=1.0, name="non_triviality",
    )
    ridge_comb, metrics_comb, _ = train_ridge(
        X_train, combined[train_idx],
        X_test,  combined[test_idx],
        alpha=1.0, name="combined",
    )

    return (
        ridge_cr, ridge_nt, ridge_comb,
        metrics_cr, metrics_nt, metrics_comb,
        train_idx, test_idx,
    )


# ── persist ───────────────────────────────────────────────────────────────────

def save_model(ridge_cr, ridge_nt, ridge_comb, embedder_info):
    kind, model = embedder_info
    bundle = {
        "ridge_creativity":     ridge_cr,
        "ridge_non_triviality": ridge_nt,
        "ridge_combined":       ridge_comb,
        "embedder_kind": kind,
        "embedder": model,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"[save] Model saved → {MODEL_PATH}", flush=True)


def save_report(metrics_cr, metrics_nt, metrics_comb,
                embedder_kind, n_train, n_test):
    report = {
        "task": "creativity_rm",
        "embedder": embedder_kind,
        "regressor": "Ridge(alpha=1.0)",
        "n_train": n_train,
        "n_test": n_test,
        "creativity":     metrics_cr,
        "non_triviality": metrics_nt,
        "combined":       metrics_comb,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[save] Report saved → {REPORT_PATH}", flush=True)
    return report


# ── public API ────────────────────────────────────────────────────────────────

class CreativityRM:
    """
    Lightweight wrapper for inference.

    Usage:
        rm = CreativityRM.load()
        score = rm.score("CoT text")           # primary creativity head
        scores = rm.score_all("CoT text")      # all three heads
    """

    def __init__(self, bundle: dict):
        self.ridge_creativity     = bundle["ridge_creativity"]
        self.ridge_non_triviality = bundle["ridge_non_triviality"]
        self.ridge_combined       = bundle["ridge_combined"]
        self.embedder_kind        = bundle["embedder_kind"]
        self.embedder             = bundle["embedder"]

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH):
        bundle = joblib.load(path)
        return cls(bundle)

    def _embed_one(self, text: str) -> np.ndarray:
        if self.embedder_kind == "sentence-transformer":
            return self.embedder.encode([text], convert_to_numpy=True, device="cpu")
        else:
            return self.embedder.transform([text])

    def score(self, cot_text: str, researcher: str | None = None) -> float:
        """Primary RL reward: predicted opus creativity score (1-5 scale)."""
        return self.score_all(cot_text, researcher=researcher)["creativity"]

    def score_all(self, cot_text: str, researcher: str | None = None) -> dict[str, float]:
        """Return all three predicted scores."""
        text = cot_text
        if researcher is not None:
            text = mask_name(cot_text, researcher)
        X = self._embed_one(text)
        return {
            "creativity":     float(self.ridge_creativity.predict(X)[0]),
            "non_triviality": float(self.ridge_non_triviality.predict(X)[0]),
            "combined":       float(self.ridge_combined.predict(X)[0]),
        }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    texts, creativity, non_triviality, combined = load_data()
    embedder_info = load_embedder()

    (ridge_cr, ridge_nt, ridge_comb,
     metrics_cr, metrics_nt, metrics_comb,
     train_idx, test_idx) = train_and_evaluate(
        texts, creativity, non_triviality, combined, embedder_info,
        test_size=args.test_size, seed=args.seed,
    )

    save_model(ridge_cr, ridge_nt, ridge_comb, embedder_info)
    report = save_report(
        metrics_cr, metrics_nt, metrics_comb,
        embedder_kind=embedder_info[0],
        n_train=len(train_idx),
        n_test=len(test_idx),
    )

    print("\n=== CREATIVITY RM SUMMARY ===")
    print(f"  Embedder:       {report['embedder']}")
    print(f"  Train / Test:   {report['n_train']} / {report['n_test']}")
    print(f"  creativity      → Pearson={metrics_cr['pearson']:.4f}  Spearman={metrics_cr['spearman']:.4f}  MAE={metrics_cr['mae']:.4f}")
    print(f"  non_triviality  → Pearson={metrics_nt['pearson']:.4f}  Spearman={metrics_nt['spearman']:.4f}  MAE={metrics_nt['mae']:.4f}")
    print(f"  combined        → Pearson={metrics_comb['pearson']:.4f}  Spearman={metrics_comb['spearman']:.4f}  MAE={metrics_comb['mae']:.4f}")

    if args.demo:
        rm = CreativityRM({
            "ridge_creativity": ridge_cr,
            "ridge_non_triviality": ridge_nt,
            "ridge_combined": ridge_comb,
            "embedder_kind": embedder_info[0],
            "embedder": embedder_info[1],
        })
        print("\n--- Demo: predicted vs actual for 5 test samples ---")
        test_texts = [texts[i] for i in test_idx[:5]]
        for j, (text, real_cr) in enumerate(zip(test_texts, creativity[test_idx[:5]])):
            preds = rm.score_all(text)
            print(f"\n[{j}] actual creativity={real_cr:.1f}  "
                  f"predicted={preds['creativity']:.3f}  "
                  f"non_triviality_pred={preds['non_triviality']:.3f}")


if __name__ == "__main__":
    main()
