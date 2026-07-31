#!/usr/bin/env python3
"""Build V3 preference pairs for RL-v2 (DPO/IPO + Bradley-Terry RM).

Principle: chosen vs rejected must be FORMAT-MATCHED and differ only on the target axis
(creative / non-trivial idea), so the gradient targets novelty — not genre/format shortcuts.

Modes (SFT-independent, run now):
  route      : within-(researcher,case) route pairs from full_cots.jsonl, ranked by an opus-4.8
               PAIRWISE judge (which CoT shows more creative, non-trivial reasoning). Keep clear-margin.
  trivialize : chosen = gold anchored CoT; rejected = opus-4.8 rewrite that KEEPS persona+anchors+length
               but replaces the non-trivial crux with an obvious/incremental idea (hard negative on novelty).
  (on-policy negatives per SFT checkpoint = separate step once checkpoints land.)

Output: runs/researcher_cot/preference/pairs_<mode>.jsonl
  {prompt:[system,user], chosen:<str>, rejected:<str>, source, researcher, case, margin}
"""
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1); os.environ.setdefault(k, v.strip().strip('"').strip("'"))
from autoresearch_idea_harness.io import load_config          # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

CFG = load_config(str(ROOT / "configs/default.yaml"))
MODEL, KEY_ENV = "claude-opus-4-8", "RUNWAY_OPUS48_API_KEY"
SYS_TMPL = ("You are {r}, a world-class AI researcher. Given a research situation and the relevant prior "
            "context, reason step by step in your own distinctive style toward ONE genuinely novel, "
            "non-obvious idea. Make the creative core unmistakably clear and explain how you arrived at "
            "it. Focus on ideas and mechanisms, not implementation details.")


def client() -> RunwayClient:
    return RunwayClient(CFG, key_env=KEY_ENV)


def call(c, messages, max_tokens=1800):
    for a in range(5):
        try:
            return c.complete(endpoint="google_anthropic", model=MODEL, messages=messages,
                              temperature=None, max_tokens=max_tokens, stream=False).text.strip()
        except Exception:
            time.sleep(3 * (a + 1))
    raise RuntimeError("opus call failed after retries")


JUDGE = ("You compare two chain-of-thought reasonings that both attempt to reach ONE novel research idea "
         "in the style of {r}, for the same research situation.\n\nSITUATION:\n{setup}\n\n"
         "--- CoT A ---\n{a}\n\n--- CoT B ---\n{b}\n\n"
         "Which better exhibits CREATIVE, NON-TRIVIAL reasoning that authentically mocks {r} and conveys a "
         "genuinely novel core idea with a non-obvious crux (ignore surface formatting/length)? "
         'Reply ONLY compact JSON: {{"winner":"A"|"B"|"tie","confidence":1-5,"reason":"<12 words"}}.')


def judge_pair(c, r, setup, cot_a, cot_b):
    txt = call(c, [{"role": "user", "content": JUDGE.format(r=r, setup=setup, a=cot_a, b=cot_b)}], 200)
    s = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(s)


TRIV = ("Below is a strong first-person research chain-of-thought in the voice of {r}. Rewrite it into a "
        "WEAKER version that keeps the SAME persona/voice, the SAME structure, the '**Mocking:** {r}' header, "
        "the 'Core idea:' and 'Non-trivial crux:' anchors, and roughly the SAME length — but make the central "
        "idea OBVIOUS, incremental, and derivative (what a mediocre researcher would propose). The "
        "'Non-trivial crux' must actually be trivial. Change ONLY the substance/novelty, not the format. "
        "Output ONLY the rewritten chain-of-thought.\n\n--- ORIGINAL ---\n{gold}")


def build_route(limit, per_group, min_conf, out):
    import collections
    rows = [json.loads(l) for l in (ROOT / "runs/researcher_cot/cots/full_cots.jsonl").open()]
    by = collections.defaultdict(list)
    for x in rows:
        by[(x["researcher"], x.get("case_shortcut_id") or x["case_title"])].append(x)
    groups = [(k, v) for k, v in by.items() if len(v) >= 2]
    random.Random(0).shuffle(groups)
    if limit:
        groups = groups[:limit]
    fmt = lambda x: f"**Mocking:** {x['researcher']}\n\n{x.get('raw_cot') or x['cot']}"

    def work(item):
        (r, case), v = item
        v = v[:]; random.Random(hash(case) & 0xffff).shuffle(v)
        pairs, c = [], client()
        for i in range(min(per_group, len(v) - 1)):
            A, B = v[i], v[i + 1]
            try:
                j = judge_pair(c, r, A["setup"], fmt(A), fmt(B))
            except Exception:
                continue
            if j.get("winner") not in ("A", "B") or int(j.get("confidence", 0)) < min_conf:
                continue
            win, los = (A, B) if j["winner"] == "A" else (B, A)
            pairs.append({"prompt": [{"role": "system", "content": SYS_TMPL.format(r=r)},
                                     {"role": "user", "content": A["setup"]}],
                          "chosen": fmt(win), "rejected": fmt(los), "source": "route",
                          "researcher": r, "case": case, "margin": int(j["confidence"])})
        return pairs
    run(groups, work, out, "route")


def build_trivialize(limit, out):
    rows = [json.loads(l) for l in (ROOT / "runs/training_data/v3_researcher_cot_anchored/train.jsonl").open()]
    random.Random(0).shuffle(rows)
    if limit:
        rows = rows[:limit]

    def work(rec):
        m = rec["messages"]; r = rec["researcher"]; gold = m[-1]["content"]; c = client()
        try:
            rej = call(c, [{"role": "user", "content": TRIV.format(r=r, gold=gold)}], 1800)
        except Exception:
            return []
        if "Mocking" not in rej or len(rej) < 200:
            return []
        return [{"prompt": m[:2], "chosen": gold, "rejected": rej, "source": "trivialize",
                 "researcher": r, "case": rec.get("case_title"), "margin": 5}]
    run(rows, work, out, "trivialize")


def run(items, work, out, tag):
    out = pathlib.Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh, ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(work, it) for it in items]
        for i, f in enumerate(as_completed(futs), 1):
            for rec in (f.result() or []):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
            if i % 20 == 0:
                fh.flush(); print(f"[{tag}] {i}/{len(items)} items, {n} pairs", flush=True)
    print(f"[{tag}] DONE: {n} pairs -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["route", "trivialize"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-group", type=int, default=2)
    ap.add_argument("--min-conf", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or str(ROOT / f"runs/researcher_cot/preference/pairs_{a.mode}.jsonl")
    if a.mode == "route":
        build_route(a.limit, a.per_group, a.min_conf, out)
    else:
        build_trivialize(a.limit, out)


if __name__ == "__main__":
    main()
