#!/usr/bin/env python3
"""Build a small, diverse demo of the SFT and RL (DPO) training data for the dashboard.

Picks ~10 SFT records and ~10 DPO preference pairs via deterministic greedy coverage over
their diversity axes (SFT: reasoning route × researcher; DPO: source × researcher × margin),
truncates long fields for display, and writes data-samples/data.json into the site bundle.

Usage:
  python scripts/build_data_samples.py                # writes to the deploy bundle
  python scripts/build_data_samples.py --out <dir> --n 10
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
SFT = HARNESS / "runs/training_data/v3_researcher_cot_anchored/train.jsonl"
DPO = HARNESS / "runs/researcher_cot/preference/pairs_all.jsonl"
BUNDLE = Path("/newcpfs/lxh/autoresearch-idea-harness-dashboard/data-samples")
CAP = 12000  # per-field display cap


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def pick_diverse(recs: list[dict], keyfns: list, n: int) -> list[dict]:
    """Greedy: repeatedly take the record that introduces the most unseen key-values.
    Deterministic (first record wins ties)."""
    chosen, seen = [], [set() for _ in keyfns]
    remaining = list(recs)
    while len(chosen) < n and remaining:
        best_pos, best_score = 0, -1
        for pos, r in enumerate(remaining):
            score = sum(1 for kf, s in zip(keyfns, seen) if kf(r) not in s)
            if score > best_score:
                best_score, best_pos = score, pos
        r = remaining.pop(best_pos)
        chosen.append(r)
        for kf, s in zip(keyfns, seen):
            s.add(kf(r))
    return chosen


def cap(s):
    if not isinstance(s, str):
        return s
    return s if len(s) <= CAP else s[:CAP] + f"\n\n… [truncated, {len(s)-CAP} more chars]"


def margin_bucket(r) -> str:
    m = r.get("margin")
    if not isinstance(m, (int, float)):
        return "na"
    return "high" if m >= 4.5 else ("mid" if m >= 3.5 else "low")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BUNDLE))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sft_all = load(SFT)
    dpo_all = load(DPO)

    sft_pick = pick_diverse(sft_all, [lambda r: r.get("route"), lambda r: r.get("researcher")], args.n)
    dpo_pick = pick_diverse(dpo_all, [lambda r: r.get("source"), lambda r: r.get("researcher"), margin_bucket], args.n)

    def sft_row(r):
        msgs = {m.get("role"): m.get("content") for m in r.get("messages", [])}
        return {"researcher": r.get("researcher"), "route": r.get("route"),
                "case_title": r.get("case_title"),
                "format_reward": r.get("format_reward"),
                "system": cap(msgs.get("system")), "user": cap(msgs.get("user")),
                "assistant": cap(msgs.get("assistant"))}

    def dpo_row(r):
        return {"researcher": r.get("researcher"), "source": r.get("source"),
                "margin": r.get("margin"), "case": r.get("case"),
                "prompt": cap(r.get("prompt")), "chosen": cap(r.get("chosen")),
                "rejected": cap(r.get("rejected"))}

    data = {
        "meta": {
            "sft": {"file": str(SFT.relative_to(HARNESS.parent)), "total": len(sft_all),
                    "shown": len(sft_pick), "schema": ["messages(system/user/assistant)", "researcher", "route", "case_title", "format_reward"],
                    "routes": sorted({r.get("route") for r in sft_all}),
                    "note": "SFT: chat-format researcher-persona idea proposals. Each sample = a famous researcher's case + one reasoning route; the assistant generates a novel research idea in that researcher's voice (with anchor-based format reward)."},
            "dpo": {"file": str(DPO.relative_to(HARNESS.parent)), "total": len(dpo_all),
                    "shown": len(dpo_pick), "schema": ["prompt", "chosen", "rejected", "margin", "source", "researcher"],
                    "sources": sorted({r.get("source") for r in dpo_all}),
                    "note": "RL is preference optimization (DPO): each pair is a shared prompt with a chosen (better) and rejected (worse) proposal; margin is the reward gap (3–5)."},
            "diversity": {"sft_axes": ["route", "researcher"], "dpo_axes": ["source", "researcher", "margin-bucket"]},
        },
        "sft": [sft_row(r) for r in sft_pick],
        "dpo": [dpo_row(r) for r in dpo_pick],
    }
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"[ok] wrote {out/'data.json'}")
    print(f"[ok] SFT {len(sft_pick)} samples, routes={sorted({r['route'] for r in data['sft']})}, "
          f"researchers={len({r['researcher'] for r in data['sft']})}")
    print(f"[ok] DPO {len(dpo_pick)} samples, sources={sorted({r['source'] for r in data['dpo']})}, "
          f"researchers={len({r['researcher'] for r in data['dpo']})}, margins={sorted({r['margin'] for r in data['dpo']})}")


if __name__ == "__main__":
    main()
