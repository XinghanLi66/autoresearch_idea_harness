#!/usr/bin/env python3
"""Synthesize first-person "how the innovation was reached" CoTs with claude-opus-4-8.

Input: the researcher pool from build_researcher_cot_pool.py (per-researcher cases + Skills).
For each case we generate several CoTs via distinct *reasoning routes* (empirical anomaly /
first-principles / analogy). Each generated record is a ready SFT triple:
  system  = generic "reason toward one novel non-obvious idea" instruction
  user    = `setup` (the situation + prior context as of that time, answer NOT leaked)
  assistant = `cot`  (first-person trace in the researcher's voice: what was noticed → why the
              obvious approaches fail → the key leap → why non-obvious → crisp core idea;
              names the researcher; NO implementation details)

Results are cached per (case, route) so re-runs resume cheaply.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autoresearch_idea_harness.io import load_config  # noqa: E402
from autoresearch_idea_harness.runway_client import RunwayClient  # noqa: E402

CACHE_DIR = ROOT / "runs" / "researcher_cot" / "cot_cache"
PROMPT_VERSION = "v5_adaptive"  # bump to invalidate cache when the synthesis/factcheck prompt changes

ROUTES = [
    ("empirical_anomaly",
     "ENTRY = a concrete, reproducible empirical observation that others explained away. Stay "
     "observation-driven: let the measured contradiction (not a theorem) do the forcing. Your beats "
     "should read like someone staring at data that won't fit the standard story."),
    ("first_principles",
     "ENTRY = a construction / invariant / limiting argument. Stay deductive: reason from what MUST "
     "be true until the idea is almost forced, largely on paper, before appealing to any experiment. "
     "This route should feel provably-driven, not observation-driven."),
    ("analogy_transfer",
     "ENTRY = a mechanism or idea from another field/subarea. Stay transfer-driven: name the source "
     "mechanism, map it over explicitly, and be precise about WHAT TRANSFERS and WHAT BREAKS. This "
     "route should foreground the analogy, not the anomaly or the construction."),
]

SYNTH_SYSTEM = (
    "You build training data that teaches a model to THINK LIKE world-class AI researchers at the "
    "moment they invent something. Given a researcher, a documented case of one of their key "
    "innovations, and a distilled profile of their research style, write a FIRST-PERSON, "
    "STRUCTURED chain-of-thought that reconstructs how THIS researcher reasoned to the core creative "
    "idea, at the historical moment BEFORE the result was known.\n\n"
    "LANGUAGE: write in English; keep technical terms in English. No other languages.\n\n"
    "=== FACTUAL & HISTORICAL INTEGRITY (highest priority — the point of this data is rigorous "
    "reasoning, so getting facts wrong poisons it) ===\n"
    "- NEVER fabricate empirical outcomes. Do not narrate a prediction as 'confirmed', do not assert "
    "a numeric winner of an ablation/experiment, and do not claim results not yet run at that moment. "
    "Describe instead WHAT you would test and the DECISION RULE you would apply (e.g. 'if the added "
    "machinery doesn't clearly help, prefer the simpler form on parsimony/complexity grounds').\n"
    "- Prefer verified history over the provided case when they conflict on empirical facts. The case "
    "notes may contain inaccuracies; use your own knowledge to stay factually correct. Be faithful to "
    "the researcher's REASONING METHOD and STYLE, not to any wrong number in the notes.\n"
    "- NO ANACHRONISMS: only invoke prior/competing work that existed at the time. IMPORTANT: the "
    "provided case notes sometimes name a competitor that actually POSTDATES the work — treat that as "
    "an error and do NOT repeat it. Never name a specific method that did not yet exist at the moment. "
    "Replace it with the true contemporaneous competitor or a generic class (e.g. 'gated-shortcut "
    "variants' rather than any specific later architecture).\n"
    "- NO HINDSIGHT-AS-ORIGINAL: if you use a framing that only became clear later, mark it as "
    "intuition, not as the decisive contemporaneous argument.\n"
    "- KEEP NECESSARY CAVEATS: do not absolutize. Preserve the real exceptions/edge cases that a "
    "careful practitioner knows are required.\n\n"
    "=== SHAPE (adaptive — match the researcher, do NOT fill a fixed template) ===\n"
    "Write the `cot` as a first-person reasoning trace in the named researcher's authentic voice and "
    "thinking style. Its shape should differ from other researchers' — someone who hid the name should "
    "still recognize whose reasoning this is (fingerprint). Do NOT use a fixed set of section headings; "
    "let the flow follow how THIS person actually thinks (a big-picture bettor, aless-is-more minimalist, a "
    "systems debugger, and a theory-first deriver should read very differently).\n"
    "Whatever the shape, the reader must be able to extract WITHOUT ambiguity: where you started / what "
    "you noticed; why the obvious or default moves are unsatisfying; the creative leap and the reasoning "
    "that produced it; why it cut against the consensus at the time; and how you'd keep yourself honest "
    "(what you'd test + the decision rule, with NO fabricated outcome). Weave these in naturally, in "
    "whatever order and proportion fit this researcher — not as labeled boilerplate. Name yourself once, "
    "naturally.\n"
    "End with EXACTLY these two labeled one-liners on their own lines (the ONLY fixed formatting):\n"
    "**Core idea:** <1-3 sentences, crisp, unambiguous>\n"
    "**Non-trivial crux:** <1-2 sentences naming the exact subtlety a reader must not miss, incl. any "
    "necessary caveat/exception>\n\n"
    "=== OTHER RULES ===\n"
    "- No implementation details: no code, no exact hyperparameters, no low-level wiring. Stay at the "
    "level of ideas, mechanisms, and reasoning.\n"
    "- Make the assigned route genuinely distinct from the other routes in entry point and emphasis; "
    "do not converge to the same wording.\n"
    "- Self-contained; no references to 'the case above'. Target 450-750 words.\n\n"
    "You also write a `setup`: a short problem statement describing the situation and prior context AS "
    "OF THAT TIME, ending by asking for a genuinely novel idea, WITHOUT revealing the answer or the "
    "core idea. This becomes the user prompt.\n\n"
    "Output STRICT JSON only: {\"setup\": \"...\", \"cot\": \"...\"}. No text outside the JSON."
)

FACTCHECK_SYSTEM = (
    "You are a rigorous historian-of-science and ML fact-checker. You are given a FIRST-PERSON "
    "reconstructed research chain-of-thought (a `setup` problem statement and a `cot`) that mimics how "
    "a named researcher reasoned to a known innovation. Revise it ONLY as needed for factual and "
    "historical integrity, using YOUR OWN verified knowledge (the draft may have inherited errors from "
    "lower-quality notes).\n\n"
    "Fix these, and only these, categories:\n"
    "(a) Fabricated empirical outcomes: remove any narrated 'prediction confirmed', asserted numeric "
    "winner of an experiment/ablation, or result not yet run at that moment. Replace with what would be "
    "tested and the decision rule. (E.g., if history shows a variant was chosen for parsimony rather "
    "than because it won numerically, say that.)\n"
    "(b) Anachronisms: any competing/prior method named that POSTDATES this work must be removed and "
    "replaced with the true contemporaneous competitor or a generic class. (E.g., for ResNet in 2015, "
    "'fractal'/'FractalNet' is anachronistic; the era competitor is Highway Networks / gated shortcuts.)\n"
    "(c) Hindsight-as-original: if a framing only became standard later, either drop it or explicitly "
    "mark it as intuition rather than the decisive contemporaneous argument.\n"
    "(d) Over-absolutized claims missing necessary caveats: restore the real exception/edge case.\n"
    "(e) Any non-English words: rewrite in English (technical terms in English).\n\n"
    "PRESERVE everything else: the first-person voice, the researcher's name and distinctive thinking "
    "style, the adaptive reasoning flow, the final '**Core idea:**' and '**Non-trivial crux:**' "
    "one-liners, the approximate length, and any correct reasoning. Do not impose a rigid template and "
    "do not blandify or add hedging beyond what integrity requires.\n\n"
    "Output STRICT JSON only: {\"setup\": \"...\", \"cot\": \"...\", \"changes\": [\"short note\", ...]}. "
    "`changes` lists what you fixed (empty list if nothing needed). No text outside the JSON."
)

TRAIN_SYSTEM = (
    "You are a world-class AI researcher. Given a research situation and the relevant prior context, "
    "reason step by step toward ONE genuinely novel, non-obvious idea. Make the creative core "
    "unmistakably clear and explain how you arrived at it. Focus on ideas and mechanisms, not "
    "implementation details."
)


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


def parse_json_obj(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    # strict=False tolerates literal newlines/tabs inside string values (the model often emits them)
    try:
        return json.loads(t, strict=False)
    except json.JSONDecodeError:
        pass
    # raw_decode grabs the first complete JSON object and ignores any trailing "Extra data"
    s = t.find("{")
    if s >= 0:
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(t[s:])
            return obj
        except json.JSONDecodeError:
            e = t.rfind("}")
            if e > s:
                return json.loads(t[s:e + 1], strict=False)
    raise ValueError("no JSON object found")


def build_user(researcher: dict[str, Any], case: dict[str, Any], route_desc: str,
               skills_md: str, max_skills_chars: int) -> str:
    skills = (skills_md or "")[:max_skills_chars]
    return (
        f"Researcher: {researcher['name']}  (style: {researcher.get('style','')})\n"
        f"Reasoning route for THIS variant: {route_desc}\n\n"
        f"--- Distilled research style (skills) ---\n{skills}\n\n"
        f"--- Documented case ---\n{case['md']}\n\n"
        "Write the `setup` and the first-person `cot` now as strict JSON."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--pool", type=Path, default=ROOT / "runs" / "researcher_cot" / "pool" / "pilot_pool.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "researcher_cot" / "cots" / "pilot_cots.jsonl")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--key-env", default="RUNWAY_OPUS48_API_KEY")
    ap.add_argument("--endpoint", default="google_anthropic")
    ap.add_argument("--routes", type=int, default=3, help="number of reasoning routes per case (<=3)")
    ap.add_argument("--only-researcher", default=None, help="substring filter on researcher name")
    ap.add_argument("--limit-cases", type=int, default=None, help="cap total cases (smoke)")
    ap.add_argument("--max-tokens", type=int, default=4500)
    ap.add_argument("--max-skills-chars", type=int, default=6000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--factcheck", dest="factcheck", action="store_true", default=True,
                    help="run a 2nd opus pass to fix fabricated outcomes / anachronisms / hindsight (default on)")
    ap.add_argument("--no-factcheck", dest="factcheck", action="store_false")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    cfg = load_config(args.config)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    researchers = [json.loads(l) for l in args.pool.open()]
    if args.only_researcher:
        researchers = [r for r in researchers if args.only_researcher.lower() in r["name"].lower()]

    # build job list: (researcher, case, route_idx)
    jobs: list[tuple[dict, dict, int]] = []
    ncase = 0
    for r in researchers:
        for c in r.get("cases", []):
            ncase += 1
            if args.limit_cases and ncase > args.limit_cases:
                break
            for ri in range(min(args.routes, len(ROUTES))):
                jobs.append((r, c, ri))
    print(f"[synth] {len(researchers)} researchers, {len(jobs)} (case,route) jobs", flush=True)

    lock = threading.Lock()
    tls = threading.local()

    def client() -> RunwayClient:
        if not hasattr(tls, "c"):
            tls.c = RunwayClient(cfg, key_env=args.key_env)
        return tls.c

    def do_job(job: tuple[dict, dict, int]) -> dict[str, Any] | None:
        researcher, case, ri = job
        route_key, route_desc = ROUTES[ri]
        cache = CACHE_DIR / f"{case['shortcut_id']}__{route_key}__{PROMPT_VERSION}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        skills_md = (researcher.get("skills") or {}).get("md", "")
        user = build_user(researcher, case, route_desc, skills_md, args.max_skills_chars)
        try:
            res = client().complete(
                endpoint=args.endpoint, model=args.model,
                messages=[{"role": "system", "content": SYNTH_SYSTEM}, {"role": "user", "content": user}],
                temperature=None, max_tokens=args.max_tokens, stream=False,
            )
            draft = parse_json_obj(res.text)
        except Exception as e:
            print(f"[synth] ERROR {researcher['name']} / {case['title'][:30]} / {route_key}: {e}", flush=True)
            return None

        setup, cot = draft.get("setup", ""), draft.get("cot", "")
        fc_changes: list[str] = []
        raw_cot = cot
        if args.factcheck:
            fc_user = (
                f"Researcher: {researcher['name']}\nInnovation / case: {case['title']}\n\n"
                f"--- DRAFT (JSON) ---\n{json.dumps({'setup': setup, 'cot': cot}, ensure_ascii=False)}\n\n"
                "Return the corrected strict JSON."
            )
            try:
                fc = client().complete(
                    endpoint=args.endpoint, model=args.model,
                    messages=[{"role": "system", "content": FACTCHECK_SYSTEM}, {"role": "user", "content": fc_user}],
                    temperature=None, max_tokens=args.max_tokens, stream=False,
                )
                fco = parse_json_obj(fc.text)
                if fco.get("cot"):
                    setup = fco.get("setup", setup)
                    cot = fco["cot"]
                    fc_changes = fco.get("changes", []) or []
            except Exception as e:
                print(f"[synth] FACTCHECK-FAILED (keeping draft) {researcher['name']} / {route_key}: {e}", flush=True)

        rec = {
            "researcher": researcher["name"], "style": researcher.get("style", ""),
            "category": researcher.get("category", ""),
            "case_title": case["title"], "case_shortcut_id": case["shortcut_id"],
            "case_overall_score": case.get("overall_score"),
            "arxiv_ids": case.get("arxiv_ids", []),
            "route": route_key,
            "setup": setup, "cot": cot,
            "raw_cot": raw_cot, "factcheck_changes": fc_changes,
            "messages": [
                {"role": "system", "content": TRAIN_SYSTEM},
                {"role": "user", "content": setup},
                {"role": "assistant", "content": cot},
            ],
            "model": getattr(res, "model", args.model),
        }
        cache.write_text(json.dumps(rec, ensure_ascii=False))
        return rec

    results: list[dict] = []
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(do_job, jobs):
            done += 1
            if rec:
                results.append(rec)
                if done % 5 == 0 or done == len(jobs):
                    print(f"[synth] {done}/{len(jobs)} done", flush=True)

    with args.out.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {
        "jobs": len(jobs), "ok": len(results),
        "researchers": len(researchers),
        "by_researcher": {r["name"]: sum(1 for x in results if x["researcher"] == r["name"]) for r in researchers},
        "out": str(args.out),
    }
    (args.out.parent / (args.out.stem + "_summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
