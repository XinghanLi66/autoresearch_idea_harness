#!/usr/bin/env python3
"""Re-fit the fingerprint + creativity reward heads and export them as slim, portable weights.

Output (runs/reward_models/slim/):
  reward_heads.npz  — fingerprint LogReg (coef/intercept/classes) + creativity Ridge (coef/intercept)
  reward_heads_meta.json — embedder id, label list, normalization, fit metrics

Rationale: the original joblib bundles pickle the whole SentenceTransformer (81MB, env-locked).
The heads themselves are tiny; in-pod we re-embed with MiniLM and apply the heads with numpy.
Fitting mirrors build_fingerprint_rm.py / build_creativity_rm.py: mask researcher names from the
CoT before embedding (so the classifier can't cheat), targets from the opus judge scores.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
COTS = ROOT / "runs/researcher_cot/cots/full_cots.jsonl"
SCORE_CACHE = ROOT / "runs/researcher_cot/score_cache"
OUT = ROOT / "runs/reward_models/slim"
CJK = re.compile(r"[一-鿿]")
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def mask_name(text: str, researcher: str) -> str:
    # strip the **Mocking:** line and redact name tokens (EN tokens + CJK run)
    text = re.sub(r"^\*\*Mocking:\*\*.*\n?", "", text)
    for t in re.split(r"[\s()·,]+", researcher):
        if len(t) >= 2 and not CJK.search(t):
            text = re.sub(re.escape(t), "[NAME]", text, flags=re.I)
    cjk = "".join(CJK.findall(researcher))
    if cjk:
        text = text.replace(cjk, "[NAME]")
    return text


def main() -> None:
    recs = [json.loads(l) for l in COTS.open()]
    rows = []
    for r in recs:
        key = SCORE_CACHE / f"{r['case_shortcut_id']}__{r['route']}.json"
        judge = json.loads(key.read_text()) if key.exists() else None
        rows.append({"researcher": r["researcher"], "cot": r["cot"], "judge": judge})
    print(f"[slim] {len(rows)} cots, {sum(1 for r in rows if r['judge'])} with judge scores", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER, device="cpu")
    texts = [mask_name(r["cot"], r["researcher"]) for r in rows]
    X = model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
    print(f"[slim] embedded: {X.shape}", flush=True)

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split

    # fingerprint head
    y = np.array([r["researcher"] for r in rows])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=3000, C=10.0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    order = np.argsort(-proba, axis=1)
    cls = np.array(clf.classes_)
    top1 = float(np.mean(cls[order[:, 0]] == yte))
    top5 = float(np.mean([yt in cls[o[:5]] for yt, o in zip(yte, order)]))
    print(f"[slim] fingerprint held-out top1={top1:.3f} top5={top5:.3f}", flush=True)

    # creativity head (mean of creativity + non_triviality, normalized later at apply time)
    jm = [(i, r["judge"]) for i, r in enumerate(rows) if r["judge"]]
    idx = np.array([i for i, _ in jm])
    t = np.array([(j.get("creativity", 0) + j.get("non_triviality", 0)) / 2 for _, j in jm], dtype=float)
    Xj = X[idx]
    Xtr2, Xte2, ttr, tte = train_test_split(Xj, t, test_size=0.15, random_state=42)
    reg = Ridge(alpha=10.0)
    reg.fit(Xtr2, ttr)
    pred = reg.predict(Xte2)
    from scipy.stats import spearmanr
    rho = float(spearmanr(pred, tte).statistic)
    print(f"[slim] creativity held-out spearman={rho:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "reward_heads.npz",
        fp_coef=clf.coef_.astype(np.float32), fp_intercept=clf.intercept_.astype(np.float32),
        cr_coef=reg.coef_.astype(np.float32), cr_intercept=np.array([reg.intercept_], dtype=np.float32),
    )
    (OUT / "reward_heads_meta.json").write_text(json.dumps({
        "embedder": EMBEDDER, "dim": int(X.shape[1]),
        "fingerprint_classes": cls.tolist(),
        "fingerprint_top1": top1, "fingerprint_top5": top5,
        "creativity_target": "mean(creativity, non_triviality) from opus judge (scale 1-5)",
        "creativity_spearman": rho,
        "mask": "strip **Mocking:** line + replace researcher name tokens with [NAME] before embedding",
        "apply": "z = emb(text_masked); fp_p = softmax(fp_coef@z + fp_intercept)[class]; cr = cr_coef@z + cr_intercept",
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "npz_bytes": (OUT / 'reward_heads.npz').stat().st_size,
                      "out": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
