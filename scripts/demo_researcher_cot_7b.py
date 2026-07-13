#!/usr/bin/env python3
"""Demo the trained 7B researcher-CoT checkpoint: mock N different researchers on held-out papers.

Picks held-out samples (default: the val split, which the model did NOT train on) from distinct
researchers, conditions the system prompt on each researcher, and generates a CoT at temp 0.8.
Writes full prompt + generated CoT + extracted 'Core idea' answer, and a readable markdown report.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "runs" / "training" / "v3_researcher_cot_7b" / "full" / "merged"


def pick_distinct(rows: list[dict], n: int) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        who = r.get("researcher")
        if who in seen:
            continue
        seen.add(who)
        out.append(r)
        if len(out) >= n:
            break
    return out


def extract(cot: str, label: str) -> str:
    m = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+?)(?:\n\*\*|\Z)", cot, re.S)
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--held-out", type=Path,
                    default=ROOT / "runs" / "training_data" / "v3_researcher_cot_prejudge" / "val.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "researcher_cot" / "demo_7b")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in args.held_out.open()]
    picks = pick_distinct(rows, args.n)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[demo] model loaded in {time.time()-t0:.0f}s on {model.device}", flush=True)

    results = []
    for i, r in enumerate(picks, 1):
        msgs = r["messages"][:2]  # system (researcher-conditioned) + user (setup); drop reference assistant
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = {k: v.to(model.device) for k, v in tok(prompt, return_tensors="pt").items()}
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        rec = {
            "researcher": r["researcher"], "case_title": r.get("case_title"),
            "system": msgs[0]["content"], "user": msgs[1]["content"],
            "generated_cot": gen,
            "answer_core_idea": extract(gen, "Core idea"),
            "answer_nontrivial_crux": extract(gen, "Non-trivial crux"),
            "gen_s": round(time.time() - t1, 1),
            "temperature": args.temperature,
        }
        results.append(rec)
        print(f"[demo] {i}/{len(picks)} {r['researcher']}: {len(gen.split())}w in {rec['gen_s']}s "
              f"| core_idea={'y' if rec['answer_core_idea'] else 'n'}", flush=True)

    (args.out_dir / "demo.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    # readable md
    md = [f"# 7B researcher-CoT demo (temp {args.temperature}, held-out papers)", "",
          f"Model: `{args.model_dir}`", ""]
    for i, r in enumerate(results, 1):
        md += [f"## {i}. Mocking {r['researcher']}", f"*held-out case: {r.get('case_title')}*", "",
               "### full prompt", "```text", f"[SYSTEM]\n{r['system']}\n\n[USER]\n{r['user']}", "```",
               "### generated CoT", "```text", r["generated_cot"], "```",
               f"### answer", f"**Core idea:** {r['answer_core_idea']}", "",
               f"**Non-trivial crux:** {r['answer_nontrivial_crux']}", "", "---", ""]
    (args.out_dir / "demo.md").write_text("\n".join(md))
    print(json.dumps({"status": "ok", "n": len(results), "researchers": [r["researcher"] for r in results],
                      "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
