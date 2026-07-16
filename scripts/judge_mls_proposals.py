#!/usr/bin/env python3
"""Judge MLS proposals with an opus-4.8 rubric — isolates proposal quality (failure Locus 1).

Scores each proposal (1-5) on the axes that determine whether a *proposal* (not the worker) is the
bottleneck: novelty_vs_baseline, mechanism_specificity, implementability (within the stated worker
constraints), expected_effectiveness (likely to beat the pass threshold), correctness. Plus a
`meaningful` verdict. Runs on any proposals_<arm>.jsonl (records: task, proposal). Cached, concurrent.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autoresearch_idea_harness.io import load_config  # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

CACHE = ROOT / "runs" / "researcher_cot" / "mls_eval" / "proposal_judge_cache"

JUDGE_SYSTEM = (
    "You are a strict ML reviewer judging a PROPOSAL for improving a specific benchmark task. You are "
    "given the task context (baseline, metric, pass threshold, worker constraints) and a proposed idea. "
    "Judge ONLY the proposal's quality as an idea to be implemented — not any implementation. Score 1-5:\n"
    "- novelty_vs_baseline: is it a genuinely different mechanism vs the stated baseline, or a trivial "
    "tweak / restatement?\n"
    "- mechanism_specificity: is the core change concrete and unambiguous (a worker could implement it "
    "exactly), or vague/hand-wavy?\n"
    "- implementability: can it be done within the stated worker constraints (edit region, no metric/"
    "epoch changes, etc.)?\n"
    "- expected_effectiveness: given ML knowledge, is it plausibly able to beat the pass threshold on "
    "this task (not just match baseline)?\n"
    "- correctness: is it free of ML errors / doomed choices?\n"
    "Also: `meaningful` (bool) = is this a real, non-trivial, on-task idea worth implementing.\n"
    "Output STRICT JSON only: {\"novelty_vs_baseline\":n,\"mechanism_specificity\":n,"
    "\"implementability\":n,\"expected_effectiveness\":n,\"correctness\":n,\"meaningful\":bool,"
    "\"one_line\":\"the proposed core change in <=20 words\",\"reason\":\"one sentence\"}."
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposals", nargs="+", required=True, help="proposals_<arm>.jsonl files")
    ap.add_argument("--packets", default=str(ROOT / "runs/researcher_cot/mls_eval/packets.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--key-env", default="RUNWAY_OPUS46_API_KEY")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    CACHE.mkdir(parents=True, exist_ok=True)

    packets = {}
    for l in open(args.packets):
        d = json.loads(l)
        # reconstruct task-context text from the user message
        ctx = d["messages"][1]["content"] if d.get("messages") else ""
        packets[d["task"]] = {"pass_metric": d.get("pass_metric"), "context": ctx[:4000]}

    jobs = []
    for pf in args.proposals:
        arm = Path(pf).stem.replace("proposals_", "")
        for l in open(pf):
            r = json.loads(l)
            jobs.append((arm, r["task"], r["proposal"], r.get("pass_metric")))

    tls = threading.local()

    def client():
        if not hasattr(tls, "c"):
            tls.c = RunwayClient(cfg, key_env=args.key_env)
        return tls.c

    def judge(job):
        arm, task, proposal, pm = job
        key = CACHE / (hashlib.sha256(f"{arm}|{task}|{proposal}".encode()).hexdigest()[:24] + ".json")
        if key.exists():
            return json.loads(key.read_text())
        ctx = packets.get(task, {}).get("context", "")
        user = f"TASK: {task} (pass threshold {pm})\n\nTASK CONTEXT:\n{ctx}\n\nPROPOSAL:\n{proposal}\n\nJudge as strict JSON."
        obj = None
        for attempt in range(4):
            try:
                res = client().complete(endpoint="google_anthropic", model=args.model,
                                        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                                        temperature=None, max_tokens=500, stream=False)
                obj = parse_json(res.text)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"[judge] ERR {arm}/{task}: {e}", flush=True)
                    return None
                import time as _t
                _t.sleep(3 * (attempt + 1))
        if obj is None:
            return None
        rec = {"arm": arm, "task": task, "pass_metric": pm, **obj}
        key.write_text(json.dumps(rec, ensure_ascii=False))
        return rec

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r in ex.map(judge, jobs):
            if r:
                results.append(r)
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))

    # summary per arm
    import statistics as st
    dims = ["novelty_vs_baseline", "mechanism_specificity", "implementability", "expected_effectiveness", "correctness"]
    summ = {}
    for arm in sorted({r["arm"] for r in results}):
        ar = [r for r in results if r["arm"] == arm]
        summ[arm] = {d: round(st.mean([x.get(d, 0) for x in ar]), 2) for d in dims}
        summ[arm]["meaningful_frac"] = round(sum(1 for x in ar if x.get("meaningful")) / len(ar), 2)
        summ[arm]["n"] = len(ar)
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "summary": summ}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
