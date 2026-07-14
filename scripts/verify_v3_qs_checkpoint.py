#!/usr/bin/env python3
"""Verify a trained checkpoint in-pod: generate held-out CoTs and score them (no training).

Loads the checkpoint (single GPU), samples N val prompts (researcher-conditioned system + setup),
generates at the given temperature, and scores each with the composite reward components
(fingerprint / format / creativity). Writes verify_result.json with full texts + scores so the
outputs can be inspected via qs logs / 3fs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v3_researcher_cot_grpo import (  # noqa: E402  (reuse shims + reward)
    CompositeReward,
    format_score,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter dir (post-RL verification)")
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--reward-heads-dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1400)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in open(args.val_jsonl)]
    picks, seen = [], set()
    for r in rows:  # distinct researchers
        if r["researcher"] in seen:
            continue
        seen.add(r["researcher"])
        picks.append(r)
        if len(picks) >= args.n:
            break

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel  # shims already applied via the trainer import above
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"[verify] merged adapter {args.adapter}", flush=True)
    model.eval()
    print(f"[verify] model loaded in {time.time()-t0:.0f}s", flush=True)

    reward = CompositeReward(Path(args.reward_heads_dir), 0.4, 0.3, 0.3)
    results = []
    for i, r in enumerate(picks, 1):
        msgs = r["messages"][:2]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = {k: v.to(model.device) for k, v in tok(prompt, return_tensors="pt").items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.95, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        s = reward.score([gen], [r["researcher"]])[0]
        anchors = {"mocking": "**Mocking:**" in gen, "core": "**Core idea:**" in gen,
                   "crux": "**Non-trivial crux:**" in gen}
        results.append({"researcher": r["researcher"], "system": msgs[0]["content"],
                        "setup": msgs[1]["content"], "generation": gen,
                        "scores": s, "anchors": anchors, "words": len(gen.split())})
        print(f"[verify] {i}/{len(picks)} {r['researcher']}: R={s['reward']:.3f} "
              f"fp={s['fingerprint']:.3f} fmt={s['format']:.2f} cr={s['creativity']:.2f} "
              f"anchors={sum(anchors.values())}/3 {len(gen.split())}w", flush=True)

    ok = (sum(r["scores"]["format"] for r in results) / len(results) >= 0.7
          and all(r["words"] > 150 for r in results))
    summary = {
        "status": "ok" if ok else "needs_review",
        "model": args.model, "temperature": args.temperature, "n": len(results),
        "mean": {k: round(sum(r["scores"][k] for r in results) / len(results), 4)
                 for k in ("fingerprint", "format", "creativity", "reward")},
        "anchor_rate": round(sum(sum(r["anchors"].values()) for r in results) / (3 * len(results)), 3),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("status", "mean", "anchor_rate")}), flush=True)


if __name__ == "__main__":
    main()
