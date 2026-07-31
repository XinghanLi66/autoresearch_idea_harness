#!/usr/bin/env python3
"""Build a Bradley-Terry reward head from the preference pairs (for the GRPO-BT arm).

Learns a linear scorer r(text) = w·emb(mask(text)) over MiniLM embeddings such that the chosen CoT
scores higher than the rejected one (Bradley-Terry / pairwise-logistic). This is the on-policy reward
for GRPO-BT — the same pairwise signal DPO consumes, so DPO-vs-GRPO is a clean apples-to-apples RL
comparison. Exports a slim head (numpy) matching the existing reward_heads format.

In:  runs/researcher_cot/preference/pairs_all.jsonl  ({prompt, chosen, rejected})
Out: runs/reward_models/slim/bt_reward_head.npz  (bt_coef[384], bt_intercept) + bt_reward_meta.json
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "runs/researcher_cot/preference/pairs_all.jsonl"
OUT = ROOT / "runs/reward_models/slim"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def mask_name(text: str, researcher: str | None) -> str:
    # mirror export_slim_reward_heads: strip the **Mocking:** line + blank the researcher name
    t = re.sub(r"^\s*\*\*Mocking:\*\*.*$", "", text, flags=re.M)
    if researcher:
        for tok in re.split(r"\s+", researcher):
            if len(tok) > 2:
                t = t.replace(tok, "[NAME]")
    return t.strip()


def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBEDDER, device="cpu")
        return lambda texts: np.asarray(m.encode(texts, batch_size=64, show_progress_bar=False,
                                                 normalize_embeddings=True), dtype=np.float32)
    except Exception as e:
        print(f"[embedder] sentence-transformers failed ({e}); using AutoModel mean-pool", flush=True)
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(EMBEDDER)
        mdl = AutoModel.from_pretrained(EMBEDDER).eval()

        def enc(texts):
            out = []
            for i in range(0, len(texts), 64):
                b = tok(texts[i:i+64], padding=True, truncation=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    h = mdl(**b).last_hidden_state
                    m = b["attention_mask"].unsqueeze(-1).float()
                    z = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    z = torch.nn.functional.normalize(z, dim=-1)
                out.append(z.numpy().astype(np.float32))
            return np.concatenate(out)
        return enc


def main() -> None:
    rows = [json.loads(l) for l in PAIRS.open()]
    print(f"[bt] {len(rows)} pairs")
    emb = get_embedder()
    chosen = emb([mask_name(r["chosen"], r.get("researcher")) for r in rows])
    rejected = emb([mask_name(r["rejected"], r.get("researcher")) for r in rows])
    diff = chosen - rejected                       # want w·diff > 0
    X = np.vstack([diff, -diff])                    # balanced: +diff -> win(1), -diff -> lose(0)
    y = np.concatenate([np.ones(len(diff)), np.zeros(len(diff))])
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, fit_intercept=False).fit(X, y)
    # train pairwise accuracy
    acc = float(((diff @ clf.coef_[0]) > 0).mean())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "bt_reward_head.npz",
                        bt_coef=clf.coef_.astype(np.float32),
                        bt_intercept=np.array([0.0], dtype=np.float32))
    (OUT / "bt_reward_meta.json").write_text(json.dumps({
        "embedder": EMBEDDER, "dim": int(chosen.shape[1]), "n_pairs": len(rows),
        "train_pairwise_acc": round(acc, 4),
        "apply": "r(text) = bt_coef @ emb(mask(text)); use as GRPO reward (group-relative, so intercept irrelevant)",
        "source": "pairs_all.jsonl (trivialize margin-5 + route)",
    }, indent=2))
    print(f"[bt] train pairwise acc={acc:.4f} -> wrote bt_reward_head.npz")


if __name__ == "__main__":
    main()
