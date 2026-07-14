#!/usr/bin/env python3
"""Quality-gate synthesized researcher CoTs, then assemble the SFT dataset.

Two gates:
  1. Rule gate (cheap, deterministic): both anchors present ('Core idea:' + 'Non-trivial crux:'),
     names the researcher, English-only body, no code fences, token length in range.
  2. Judge gate (claude-opus-4-8 rubric): faithfulness / clarity_of_core / non_triviality /
     creativity / fingerprint (1-5) + impl_leak / has_anchors / names_researcher flags + verdict.

Keeps only CoTs passing both gates, dedups near-identical routes per case, and writes a train/val
SFT dataset (messages format) plus a scored jsonl and a summary.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import synthesize_researcher_cot as S  # noqa: E402
from autoresearch_idea_harness.io import load_config  # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

SCORE_CACHE = ROOT / "runs" / "researcher_cot" / "score_cache"
CJK = re.compile(r"[一-鿿]")
QWEN_TOK = "/newcpfs/user/yuanqianhao/hf_models/Qwen/Qwen2.5-32B-Instruct"

JUDGE_SYSTEM = (
    "You are a STRICT grader of training chain-of-thoughts that teach a model to think like a named "
    "researcher at the moment of inventing something. Grade the given `cot` (with its `setup`).\n"
    "Score 1-5 (5=best):\n"
    "- faithfulness: consistent with real history and the researcher's actual method; NO fabricated "
    "results, NO anachronistic method names, NO hindsight sold as the original argument.\n"
    "- clarity_of_core: is the creative core stated so a careful reader cannot misunderstand it?\n"
    "- non_triviality: is the core genuinely non-obvious / against consensus, not a truism?\n"
    "- creativity: does it show inventive reasoning and a real leap, not a flat paraphrase?\n"
    "- fingerprint: does it read like THIS researcher's distinctive thinking style?\n"
    "Boolean flags:\n"
    "- impl_leak: true if it includes implementation details (code, exact hyperparameters, low-level wiring).\n"
    "- has_anchors: true if it contains BOTH '**Core idea:**' and '**Non-trivial crux:**'.\n"
    "- names_researcher: true if the researcher is named in the text.\n"
    "verdict: 'keep' | 'revise' | 'drop'.\n"
    "Output STRICT JSON only: {\"faithfulness\":n,\"clarity_of_core\":n,\"non_triviality\":n,"
    "\"creativity\":n,\"fingerprint\":n,\"impl_leak\":bool,\"has_anchors\":bool,"
    "\"names_researcher\":bool,\"verdict\":\"...\",\"reason\":\"one sentence\"}."
)


def researcher_system(name: str) -> str:
    return (
        f"You are {name}, a world-class AI researcher. Given a research situation and the relevant "
        "prior context, reason step by step in your own distinctive style toward ONE genuinely novel, "
        "non-obvious idea. Make the creative core unmistakably clear and explain how you arrived at it. "
        "Focus on ideas and mechanisms, not implementation details."
    )


def conditioned_messages(rec: dict[str, Any]) -> list[dict[str, str]]:
    """Rebuild the SFT triple with a researcher-conditioned system prompt (names the researcher in
    the input, so the model is controllable and every sample indicates whose thinking it is)."""
    setup = rec.get("setup") or (rec.get("messages", [{}, {}])[1].get("content", ""))
    cot = rec.get("cot") or (rec.get("messages", [{}, {}, {}])[2].get("content", ""))
    return [
        {"role": "system", "content": researcher_system(rec.get("researcher", "a researcher"))},
        {"role": "user", "content": setup},
        {"role": "assistant", "content": cot},
    ]


def rule_gate(rec: dict[str, Any], tok, min_tok: int, max_tok: int) -> list[str]:
    """Return list of rule failures (empty = pass)."""
    cot = rec.get("cot", "")
    fails = []
    if "**Core idea:**" not in cot:
        fails.append("missing_core_idea_anchor")
    if "**Non-trivial crux:**" not in cot:
        fails.append("missing_crux_anchor")
    if "```" in cot:
        fails.append("contains_code_fence")
    # (researcher identity is guaranteed via the conditioned system prompt, so we do not
    #  require the cot to self-name the researcher.)
    # English-only body (allow a few CJK chars for names/terms); flag if too many
    if len(CJK.findall(cot)) > 8:
        fails.append("too_much_non_english")
    n = len(tok(cot))
    if n < min_tok:
        fails.append(f"too_short({n})")
    if n > max_tok:
        fails.append(f"too_long({n})")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--cots", type=Path, required=True, help="synthesized cots jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "training_data" / "v3_researcher_cot")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--key-env", default="RUNWAY_OPUS48_API_KEY")
    ap.add_argument("--endpoint", default="google_anthropic")
    ap.add_argument("--min-tokens", type=int, default=350)
    ap.add_argument("--max-tokens", type=int, default=1800)
    ap.add_argument("--min-sub-score", type=int, default=4, help="min for faithfulness/clarity/non_triv/creativity")
    ap.add_argument("--min-fingerprint", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--no-judge", dest="judge", action="store_false", default=True)
    args = ap.parse_args()

    S.load_env(ROOT / ".env")
    cfg = load_config(args.config)
    SCORE_CACHE.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoTokenizer
        _t = AutoTokenizer.from_pretrained(QWEN_TOK, trust_remote_code=True)
        tok = lambda s: _t.encode(s or "")
    except Exception:
        tok = lambda s: (s or "").split()

    recs = [json.loads(l) for l in args.cots.open()]
    print(f"[score] {len(recs)} CoTs loaded", flush=True)

    tls = threading.local()

    def client() -> RunwayClient:
        if not hasattr(tls, "c"):
            tls.c = RunwayClient(cfg, key_env=args.key_env)
        return tls.c

    def judge(rec: dict[str, Any]) -> dict[str, Any] | None:
        key = f"{rec.get('case_shortcut_id','x')}__{rec.get('route','x')}.json"
        cache = SCORE_CACHE / key
        if cache.exists():
            return json.loads(cache.read_text())
        user = (f"Researcher: {rec['researcher']}\nCase: {rec.get('case_title','')}\n\n"
                f"--- setup ---\n{rec.get('setup','')}\n\n--- cot ---\n{rec.get('cot','')}\n\nGrade as strict JSON.")
        try:
            r = client().complete(endpoint=args.endpoint, model=args.model,
                                  messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                                  temperature=None, max_tokens=600, stream=False)
            obj = S.parse_json_obj(r.text)
        except Exception as e:
            print(f"[score] judge ERROR {rec.get('researcher')} / {rec.get('route')}: {e}", flush=True)
            return None
        cache.write_text(json.dumps(obj, ensure_ascii=False))
        return obj

    # rule gate first
    for r in recs:
        r["rule_fails"] = rule_gate(r, tok, args.min_tokens, args.max_tokens)

    # judge gate (concurrent) on rule-passing recs
    passable = [r for r in recs if not r["rule_fails"]]
    if args.judge:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for r, j in zip(passable, ex.map(judge, passable)):
                r["judge"] = j
    else:
        for r in passable:
            r["judge"] = None

    def keeps(r: dict[str, Any]) -> bool:
        if r["rule_fails"]:
            return False
        j = r.get("judge")
        if not args.judge:
            return True
        if not j:
            return False
        subs = [j.get(k, 0) for k in ("faithfulness", "clarity_of_core", "non_triviality", "creativity")]
        # NOTE: we do NOT require names_researcher — the SFT sample names the researcher via the
        # conditioned system prompt, so penalizing the cot for not self-naming would wrongly drop
        # good data. Anchors are kept (they teach the Core idea / Non-trivial crux format).
        return (j.get("verdict") == "keep" and all(s >= args.min_sub_score for s in subs)
                and j.get("fingerprint", 0) >= args.min_fingerprint and not j.get("impl_leak", True)
                and j.get("has_anchors", False))

    for r in recs:
        r["kept"] = keeps(r)
    kept = [r for r in recs if r["kept"]]

    # dedup: at most keep all routes but drop exact-duplicate cot text per case
    seen: set[tuple] = set()
    deduped = []
    for r in kept:
        sig = (r.get("case_shortcut_id"), r.get("cot", "")[:200])
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(r)

    # split
    val_n = max(1, int(len(deduped) * args.val_frac)) if deduped else 0
    val = deduped[:val_n]
    train = deduped[val_n:]

    def write_split(name: str, rows: list[dict[str, Any]]):
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({"messages": conditioned_messages(r), "researcher": r["researcher"],
                                    "case_title": r["case_title"], "route": r["route"],
                                    "source": "v3_researcher_cot"}, ensure_ascii=False) + "\n")

    write_split("train", train)
    write_split("val", val)
    with (args.out_dir / "scored.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps({k: r.get(k) for k in
                                ("researcher", "case_title", "route", "rule_fails", "judge", "kept")},
                               ensure_ascii=False) + "\n")

    summary = {
        "input": len(recs), "rule_pass": len(passable),
        "kept": len(kept), "after_dedup": len(deduped),
        "train": len(train), "val": len(val),
        "rule_fail_reasons": _count([f for r in recs for f in r["rule_fails"]]),
        "drop_rate": round(1 - len(deduped) / len(recs), 3) if recs else 0,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False))


def _count(items: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for i in items:
        d[i] = d.get(i, 0) + 1
    return dict(sorted(d.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    main()
