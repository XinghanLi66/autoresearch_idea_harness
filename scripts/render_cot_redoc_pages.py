#!/usr/bin/env python3
"""Render the pilot CoT dataset into RedDoc page markdown files (offline).

Emits, under runs/researcher_cot/redoc_pages/:
  overview.md         - workflow of the researcher-mocking prompt + length stats (user/assistant/whole)
  raw_prompts.md      - the raw construction prompts (synth system / factcheck system / train system /
                        routes / user-template), each in a collapsible code block
  cot_0001.md ...     - one page per pilot CoT: metadata + `setup` and `cot` in collapsible code blocks
  pages_manifest.json - ordered list of {kind, title, file} for the pusher

Code blocks are how RedDoc renders collapsible content; content inside a code fence needs no escaping.
Token lengths use the Qwen2.5-32B tokenizer so they map to the training context budget.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import synthesize_researcher_cot as S  # noqa: E402

QWEN_TOK = "/newcpfs/user/yuanqianhao/hf_models/Qwen/Qwen2.5-32B-Instruct"


def fence(body: str, lang: str = "text") -> str:
    """Code fence long enough to contain any backtick run in body."""
    longest = 0
    run = 0
    for ch in body:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    bar = "`" * max(3, longest + 1)
    return f"{bar}{lang}\n{body}\n{bar}"


def get_counter():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN_TOK, trust_remote_code=True)
        return lambda s: len(tok.encode(s or "")), "qwen2.5-32b tokens"
    except Exception:
        return lambda s: len(str(s or "").split()), "whitespace words (tokenizer unavailable)"


def stats(vals: list[int]) -> dict[str, int]:
    return {"avg": round(sum(vals) / len(vals)) if vals else 0,
            "min": min(vals) if vals else 0, "max": max(vals) if vals else 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cots", type=Path, default=ROOT / "runs" / "researcher_cot" / "cots" / "pilot_cots.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "researcher_cot" / "redoc_pages")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.cots.open()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    count, unit = get_counter()

    # --- length stats over user(setup) / assistant(cot) / whole(system+user+assistant) ---
    u, a, w = [], [], []
    for r in recs:
        us, cs = count(r["setup"]), count(r["cot"])
        whole = count(S.TRAIN_SYSTEM) + us + cs
        r["_u"], r["_a"], r["_w"] = us, cs, whole
        u.append(us); a.append(cs); w.append(whole)
    su, sa, sw = stats(u), stats(a), stats(w)

    manifest: list[dict[str, str]] = []

    # --- raw prompts page ---
    routes_txt = "\n\n".join(f"[{k}]\n{d}" for k, d in S.ROUTES)
    user_tmpl = (
        "Researcher: {name}  (style: {style})\n"
        "Reasoning route for THIS variant: {route_desc}\n\n"
        "--- Distilled research style (skills) ---\n{skills (truncated to --max-skills-chars)}\n\n"
        "--- Documented case ---\n{case markdown}\n\n"
        "Write the `setup` and the first-person `cot` now as strict JSON."
    )
    factcheck_user_tmpl = (
        "Researcher: {name}\nInnovation / case: {case_title}\n\n"
        "--- DRAFT (JSON) ---\n{{\"setup\":..., \"cot\":...}}\n\n"
        "Return the corrected strict JSON."
    )
    raw = [
        f"# Raw construction prompts (`{S.PROMPT_VERSION}`, model `claude-opus-4-8`)",
        "",
        "Two opus-4.8 calls per CoT: **(1) synthesis** then **(2) fact-check**. Both use the same "
        "Runway `google_anthropic` endpoint / key. The fact-check call is given only the draft + "
        "researcher/paper name (not the source notes), so it corrects from the model's own knowledge.",
        "", "## 1a. Synthesis — system prompt", fence(S.SYNTH_SYSTEM),
        "", "## 1b. Synthesis — user template", fence(user_tmpl),
        "", "## 1c. Reasoning routes (one CoT per route)", fence(routes_txt),
        "", "## 2a. Fact-check — system prompt", fence(S.FACTCHECK_SYSTEM),
        "", "## 2b. Fact-check — user template", fence(factcheck_user_tmpl),
        "", "## 3. Training system prompt (the `system` turn in each SFT sample)", fence(S.TRAIN_SYSTEM),
    ]
    (args.out_dir / "raw_prompts.md").write_text("\n".join(raw))
    manifest.append({"kind": "raw_prompts", "title": "Raw construction prompts (synthesis + fact-check)",
                     "file": "raw_prompts.md"})

    # --- per-CoT pages ---
    for i, r in enumerate(recs, 1):
        short_case = r["case_title"]
        title = f"Pilot CoT {i:02d} — {r['researcher']} · {r['route']}"
        changes = r.get("factcheck_changes") or []
        body = [
            f"# {title}",
            "",
            f"- researcher: **{r['researcher']}**  ·  style: {r.get('style','')}",
            f"- case: {short_case}",
            f"- route: `{r['route']}`  ·  case score: {r.get('case_overall_score')}",
            f"- lengths ({unit}): user(setup) **{r['_u']}** · assistant(cot) **{r['_a']}** · whole **{r['_w']}**",
            f"- fact-check changes: {len(changes)}",
            "",
            "## user (setup)", fence(r["setup"]),
            "", "## assistant (cot)", fence(r["cot"]),
        ]
        if changes:
            body += ["", "## fact-check changes", fence("\n".join(f"- {c}" for c in changes))]
        fn = f"cot_{i:04d}.md"
        (args.out_dir / fn).write_text("\n".join(body))
        manifest.append({"kind": "cot", "title": title, "file": fn})

    # --- overview page (links filled in by the pusher after subpages exist) ---
    by_researcher: dict[str, int] = {}
    for r in recs:
        by_researcher[r["researcher"]] = by_researcher.get(r["researcher"], 0) + 1
    ov = [
        "# Researcher-Mocking CoT — Pilot Overview",
        "",
        f"**{len(recs)} chain-of-thought samples** distilled from **{len(by_researcher)} researchers** "
        "(pilot). Each sample teaches a model to reconstruct, in first person, how a top researcher "
        "reasoned to a real innovation — the creative idea and its non-trivial core stated clearly, and "
        "how they got there, with NO implementation details.",
        "",
        "## Workflow — how each researcher-mocking prompt is constructed",
        "",
        "1. **Pool crawl.** From the RedDoc 蒸馏计划 roster, for each researcher we pull their 第一步 "
        "case subdocs (①problem → ②idea → ③ablation → ④reflection, with quality scores) and their 第二步 "
        "Skills-v1 fingerprint.",
        "2. **Synthesis (opus-4.8, call 1).** For each case we generate several CoTs, one per *reasoning "
        "route* (empirical-anomaly / first-principles / analogy-transfer). The model writes a first-person "
        "trace in the researcher's voice (adaptive shape — no fixed template), plus a leak-free `setup` "
        "problem statement. It must name the researcher, omit implementation details, and end with two "
        "anchors: **Core idea** and **Non-trivial crux**.",
        "3. **Fact-check (opus-4.8, call 2).** A separate pass revises the draft for factual/historical "
        "integrity — removing fabricated outcomes, anachronistic method names, hindsight-as-original, and "
        "restoring necessary caveats — using the model's own knowledge (not the source notes). Every edit "
        "is logged per record.",
        "4. **SFT triple.** Each record becomes `[system (researcher-reasoning instruction), user (setup), "
        "assistant (cot)]`.",
        "",
        "See the **Raw construction prompts** subpage for the exact system/user prompts.",
        "",
        f"## Length stats ({unit})",
        "",
        "| segment | avg | min | max |",
        "|---|---|---|---|",
        f"| user (setup) | {su['avg']} | {su['min']} | {su['max']} |",
        f"| assistant (cot) | {sa['avg']} | {sa['min']} | {sa['max']} |",
        f"| whole (system+user+assistant) | {sw['avg']} | {sw['min']} | {sw['max']} |",
        "",
        f"All well within the Qwen2.5-32B context (32k native; SFT seq-len 8–16k).",
        "",
        "## Coverage",
        "",
        "| researcher | CoTs |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in by_researcher.items()],
        "",
        "## Pages",
        "",
        "_Subpage links are appended below after creation._",
    ]
    (args.out_dir / "overview.md").write_text("\n".join(ov))
    manifest.insert(0, {"kind": "overview", "title": "Researcher-Mocking CoT — Pilot Overview",
                        "file": "overview.md"})

    (args.out_dir / "pages_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "records": len(recs), "pages": len(manifest),
                      "unit": unit, "user": su, "assistant": sa, "whole": sw,
                      "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
