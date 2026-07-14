#!/usr/bin/env python3
"""
Fingerprint discriminator: given a name-masked CoT, predict which of 125 researchers wrote it.
Reward for RL = predicted P(target researcher).

Usage:
    python scripts/build_fingerprint_rm.py           # train + eval + save
    python scripts/build_fingerprint_rm.py --demo    # also print top-5 predictions for a few samples
"""

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
COTS_PATH = REPO_ROOT / "runs/researcher_cot/cots/full_cots.jsonl"
POOL_PATH = REPO_ROOT / "runs/researcher_cot/pool/pool_full.jsonl"
OUTPUT_DIR = REPO_ROOT / "runs/reward_models"
MODEL_PATH = OUTPUT_DIR / "fingerprint_rm.joblib"
REPORT_PATH = OUTPUT_DIR / "fingerprint_rm_report.json"

# ── sentence-transformer (CPU-safe) ──────────────────────────────────────────
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
        vec = TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        return ("tfidf", vec)


# ── name masking ──────────────────────────────────────────────────────────────

def _name_tokens(name: str) -> list[str]:
    """
    Return a list of non-trivial tokens to redact.
    For e.g. '李飞飞 Fei-Fei Li':
      - CJK block  → '李飞飞'
      - Latin parts → ['Fei-Fei', 'Li', 'Fei']   (plus split on '-')
    Skip single ASCII chars and common stop words.
    """
    STOPWORDS = {"de", "van", "von", "le", "la", "el", "di", "do", "da"}
    tokens = []

    # CJK characters as one contiguous token
    cjk_block = re.sub(r"[^一-鿿㐀-䶿 0-⩭f]", "", name)
    if cjk_block:
        tokens.append(cjk_block)

    # Latin words (handles accent chars too)
    latin = re.sub(r"[一-鿿㐀-䶿 0-⩭f]", "", name)
    # keep parenthetical aliases like "(CPMP)"
    for tok in re.split(r"[\s\-\(\)\.]+", latin):
        tok = tok.strip()
        if len(tok) > 1 and tok.lower() not in STOPWORDS:
            tokens.append(tok)

    return tokens


def mask_name(text: str, researcher: str) -> str:
    """
    1. Strip the leading **Mocking:** line if present.
    2. Redact every occurrence of the researcher's name tokens.
    """
    # Strip leading **Mocking:** or similar header lines
    text = re.sub(r"^\*\*Mocking:\*\*[^\n]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Mocking:[^\n]*\n?", "", text, flags=re.IGNORECASE)

    tokens = _name_tokens(researcher)
    for tok in tokens:
        # case-insensitive whole-word redaction
        text = re.sub(r"\b" + re.escape(tok) + r"\b", "[RESEARCHER]", text, flags=re.IGNORECASE)

    return text


# ── data loading ──────────────────────────────────────────────────────────────

def load_data():
    # Load researcher pool (canonical order → class index)
    with open(POOL_PATH) as f:
        roster = [json.loads(l)["name"] for l in f]
    researcher_to_idx = {name: i for i, name in enumerate(roster)}

    texts, labels = [], []
    missing_from_pool = set()
    with open(COTS_PATH) as f:
        for line in f:
            d = json.loads(line)
            researcher = d.get("researcher", "")
            cot = d.get("cot", "") or ""
            if not cot.strip():
                continue
            if researcher not in researcher_to_idx:
                missing_from_pool.add(researcher)
                continue
            masked = mask_name(cot, researcher)
            texts.append(masked)
            labels.append(researcher_to_idx[researcher])

    if missing_from_pool:
        print(f"[data] Warning: {len(missing_from_pool)} researcher(s) not in pool (skipped): "
              f"{list(missing_from_pool)[:5]}", flush=True)

    print(f"[data] Loaded {len(texts)} CoTs, {len(set(labels))} unique researchers.", flush=True)
    return texts, labels, roster


# ── embedding ─────────────────────────────────────────────────────────────────

def embed(embedder_info, texts: list[str], batch_size: int = 256, fit: bool = False):
    kind, model = embedder_info
    if kind == "sentence-transformer":
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            device="cpu",
        )
        return embeddings
    else:
        # TF-IDF fallback
        if fit:
            return model.fit_transform(texts)
        else:
            return model.transform(texts)


# ── training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(texts, labels, roster, embedder_info, test_size: float = 0.15, seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    labels_arr = np.array(labels)
    n = len(texts)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(n), labels_arr))

    train_texts = [texts[i] for i in train_idx]
    test_texts  = [texts[i] for i in test_idx]
    y_train = labels_arr[train_idx]
    y_test  = labels_arr[test_idx]

    print(f"[split] Train: {len(train_texts)}, Test: {len(test_texts)}", flush=True)

    # Embed
    print("[embed] Embedding training set …", flush=True)
    X_train = embed(embedder_info, train_texts, fit=True)
    print("[embed] Embedding test set …", flush=True)
    X_test  = embed(embedder_info, test_texts, fit=False)

    # Classifier
    print("[train] Fitting LogisticRegression (max_iter=2000) …", flush=True)
    clf = LogisticRegression(
        max_iter=2000,
        C=4.0,
        solver="lbfgs",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    proba = clf.predict_proba(X_test)           # (n_test, 125)
    top1_preds = np.argmax(proba, axis=1)
    top5_preds = np.argsort(proba, axis=1)[:, -5:]

    top1_acc = float(np.mean(top1_preds == y_test))
    top5_acc = float(np.mean([y in top5 for y, top5 in zip(y_test, top5_preds)]))
    random_top1 = 1.0 / len(roster)
    random_top5 = 5.0 / len(roster)

    print(f"\n[eval] Top-1 accuracy: {top1_acc:.4f}  (random baseline: {random_top1:.4f})")
    print(f"[eval] Top-5 accuracy: {top5_acc:.4f}  (random baseline: {random_top5:.4f})\n")

    # Per-class accuracy
    per_class = {}
    for idx, name in enumerate(roster):
        mask = y_test == idx
        if mask.sum() == 0:
            continue
        per_class[name] = float(np.mean(top1_preds[mask] == idx))

    return clf, proba, y_test, top1_acc, top5_acc, per_class, train_idx, test_idx


# ── persist ───────────────────────────────────────────────────────────────────

def save_model(clf, embedder_info, roster):
    kind, model = embedder_info
    bundle = {
        "clf": clf,
        "embedder_kind": kind,
        "embedder": model,
        "roster": roster,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"[save] Model saved → {MODEL_PATH}", flush=True)


def save_report(top1_acc, top5_acc, embedder_kind, n_train, n_test, n_classes, per_class):
    random_top1 = 1.0 / n_classes
    random_top5 = 5.0 / n_classes
    report = {
        "task": "fingerprint_discriminator",
        "embedder": embedder_kind,
        "classifier": "LogisticRegression(C=4, lbfgs)",
        "n_classes": n_classes,
        "n_train": n_train,
        "n_test": n_test,
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "random_top1_baseline": round(random_top1, 4),
        "random_top5_baseline": round(random_top5, 4),
        "lift_top1": round(top1_acc / random_top1, 1),
        "lift_top5": round(top5_acc / random_top5, 1),
        "per_class_top1": per_class,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[save] Report saved → {REPORT_PATH}", flush=True)
    return report


# ── public API ────────────────────────────────────────────────────────────────

class FingerprintRM:
    """
    Lightweight wrapper for inference.

    Usage:
        rm = FingerprintRM.load()
        probs = rm.predict_proba("CoT text here")
        # → dict  { "Richard Sutton": 0.82, "Geoffrey Hinton": 0.03, … }
    """

    def __init__(self, bundle: dict):
        self.clf = bundle["clf"]
        self.embedder_kind = bundle["embedder_kind"]
        self.embedder = bundle["embedder"]
        self.roster = bundle["roster"]

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH):
        bundle = joblib.load(path)
        return cls(bundle)

    def _embed_one(self, text: str) -> np.ndarray:
        if self.embedder_kind == "sentence-transformer":
            vec = self.embedder.encode([text], convert_to_numpy=True, device="cpu")
        else:
            vec = self.embedder.transform([text])
        return vec

    def predict_proba(self, cot_text: str, researcher: str | None = None) -> dict[str, float]:
        """
        Returns {researcher_name: probability} for all 125 researchers,
        sorted descending by probability.

        If `researcher` is supplied, the name is masked before featurizing
        (same as training time).
        """
        text = cot_text
        if researcher is not None:
            text = mask_name(cot_text, researcher)

        X = self._embed_one(text)
        proba = self.clf.predict_proba(X)[0]
        result = {name: float(p) for name, p in zip(self.roster, proba)}
        return dict(sorted(result.items(), key=lambda kv: -kv[1]))

    def target_prob(self, cot_text: str, researcher: str) -> float:
        """Return P(researcher | cot) — the RL reward signal."""
        probs = self.predict_proba(cot_text, researcher=researcher)
        return probs.get(researcher, 0.0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Print top-5 predictions for a few held-out samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    texts, labels, roster = load_data()
    embedder_info = load_embedder()

    clf, proba, y_test, top1_acc, top5_acc, per_class, train_idx, test_idx = train_and_evaluate(
        texts, labels, roster, embedder_info,
        test_size=args.test_size, seed=args.seed,
    )

    save_model(clf, embedder_info, roster)
    report = save_report(
        top1_acc, top5_acc,
        embedder_info[0],
        n_train=len(train_idx),
        n_test=len(test_idx),
        n_classes=len(roster),
        per_class=per_class,
    )

    print("\n=== FINGERPRINT RM SUMMARY ===")
    print(f"  Embedder:       {report['embedder']}")
    print(f"  Classes:        {report['n_classes']}")
    print(f"  Train / Test:   {report['n_train']} / {report['n_test']}")
    print(f"  Top-1 acc:      {report['top1_accuracy']:.4f}  (random: {report['random_top1_baseline']:.4f}, lift: {report['lift_top1']}x)")
    print(f"  Top-5 acc:      {report['top5_accuracy']:.4f}  (random: {report['random_top5_baseline']:.4f}, lift: {report['lift_top5']}x)")

    if args.demo:
        print("\n--- Demo: top-5 predictions for 5 held-out samples ---")
        for i in range(min(5, len(y_test))):
            true_name = roster[y_test[i]]
            top5 = np.argsort(proba[i])[-5:][::-1]
            top5_names = [(roster[j], round(float(proba[i][j]), 4)) for j in top5]
            print(f"\n[{i}] TRUE: {true_name}")
            for rank, (name, p) in enumerate(top5_names, 1):
                marker = "✓" if name == true_name else " "
                print(f"  {rank}. {marker} {name:<35} {p:.4f}")


if __name__ == "__main__":
    main()
