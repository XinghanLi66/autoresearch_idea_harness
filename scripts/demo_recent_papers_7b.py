#!/usr/bin/env python3
"""OOD demo: run the 7B researcher-CoT checkpoint on RECENT papers (new problems it never trained on).

For each recent paper: opus-4.8 turns (title+abstract) into a leak-free `setup` and picks a best-fit
plus a contrasting researcher from the 125-roster; then the 7B checkpoint mocks each researcher on
that setup at temp 0.8. This tests generalization to new problems (vs the IID val-split demo).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autoresearch_idea_harness.io import load_config  # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

MERGED = ROOT / "runs" / "training" / "v3_researcher_cot_7b" / "full" / "merged"
DATASET = Path("/newcpfs/lxh/agentic-training/proposal_rl/runs/dataset")
POOL = ROOT / "runs" / "researcher_cot" / "pool" / "pool_full.jsonl"

DEFAULT_IDS = ["2602.11185", "2601.22947", "2510.14027", "2602.00453", "2601.19094"]

SETUP_SYSTEM = (
    "You prepare OOD test prompts for a model that mocks how famous AI researchers reason toward new "
    "ideas. Given a RECENT paper (title + abstract) and a roster of researchers, do two things:\n"
    "1) Write a `setup`: a concise, leak-free statement of the PROBLEM / situation the paper addresses "
    "(the motivation and what's unsatisfying about current approaches) AS IF the work has not been done "
    "yet, ending by asking for a genuinely novel idea. Do NOT reveal the paper's actual solution/method "
    "or results. English only.\n"
    "2) From the roster, pick `best_fit` = the researcher whose style/expertise best matches this "
    "problem, and `wildcard` = a stylistically DIFFERENT researcher who would attack it from an unusual "
    "angle. Use exact roster names.\n"
    "Output STRICT JSON only: {\"setup\":\"...\",\"best_fit\":\"...\",\"wildcard\":\"...\",\"why\":\"one sentence\"}."
)
RESEARCHER_SYSTEM = (
    "You are {name}, a world-class AI researcher. Given a research situation and the relevant prior "
    "context, reason step by step in your own distinctive style toward ONE genuinely novel, non-obvious "
    "idea. Make the creative core unmistakably clear and explain how you arrived at it. Focus on ideas "
    "and mechanisms, not implementation details."
)


def load_env(p: Path):
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def parse_json(t: str):
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        s = t.find("{")
        return json.JSONDecoder(strict=False).raw_decode(t[s:])[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, default=MERGED)
    ap.add_argument("--arxiv-ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1500)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "researcher_cot" / "demo_7b_recent")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # roster (name -> style)
    roster = {}
    for line in POOL.open():
        r = json.loads(line)
        roster[r["name"]] = r.get("style", "")
    roster_txt = "\n".join(f"- {n} — {s}" for n, s in roster.items())

    # load recent papers
    papers = {}
    for split in ["test", "val"]:
        for line in (DATASET / f"{split}.jsonl").open():
            r = json.loads(line)
            if r.get("arxiv_id") in args.arxiv_ids:
                papers[r["arxiv_id"]] = r
    picked = [papers[a] for a in args.arxiv_ids if a in papers]
    print(f"[recent] {len(picked)} recent papers loaded", flush=True)

    # 1) build setups + researcher picks via opus-4.8
    client = RunwayClient(cfg, key_env="RUNWAY_OPUS48_API_KEY")
    prepared = []
    for p in picked:
        user = (f"RECENT PAPER\ntitle: {p.get('title')}\narxiv: {p.get('arxiv_id')} ({p.get('created','')[:10]})\n"
                f"abstract: {p.get('abstract')}\n\nROSTER:\n{roster_txt}\n\nReturn strict JSON.")
        r = client.complete(endpoint="google_anthropic", model="claude-opus-4-8",
                            messages=[{"role": "system", "content": SETUP_SYSTEM}, {"role": "user", "content": user}],
                            temperature=None, max_tokens=1500, stream=False)
        obj = parse_json(r.text)
        obj["arxiv_id"] = p["arxiv_id"]; obj["title"] = p.get("title"); obj["created"] = p.get("created", "")[:10]
        obj["categories"] = p.get("categories")
        prepared.append(obj)
        print(f"[recent] setup ready {p['arxiv_id']}: best_fit={obj.get('best_fit')} wildcard={obj.get('wildcard')}", flush=True)

    # 2) 7B generate (load once), mock best_fit + wildcard per paper
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(args.model_dir), torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"[recent] 7B loaded on {model.device}", flush=True)

    def gen(name, setup):
        msgs = [{"role": "system", "content": RESEARCHER_SYSTEM.format(name=name)}, {"role": "user", "content": setup}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = {k: v.to(model.device) for k, v in tok(prompt, return_tensors="pt").items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.95, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip(), msgs[0]["content"]

    results = []
    for obj in prepared:
        entry = {**{k: obj[k] for k in ("arxiv_id", "title", "created", "categories", "setup", "best_fit", "wildcard", "why")},
                 "generations": []}
        for role_key in ("best_fit", "wildcard"):
            name = obj.get(role_key)
            if not name:
                continue
            cot, sys_prompt = gen(name, obj["setup"])
            entry["generations"].append({"role": role_key, "researcher": name, "system": sys_prompt, "cot": cot})
            print(f"[recent] {obj['arxiv_id']} {role_key}={name}: {len(cot.split())}w", flush=True)
        results.append(entry)

    (args.out_dir / "recent_demo.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    print(json.dumps({"status": "ok", "papers": len(results), "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
