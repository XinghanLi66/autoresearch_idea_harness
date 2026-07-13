#!/usr/bin/env python3
"""Publish the 7B OOD demo (recent papers) to a NEW RedDoc page (does not touch the IID demo doc)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "4709bcf060916d61278c76e8a1b668e0"
DEMO = ROOT / "runs" / "researcher_cot" / "demo_7b_recent" / "recent_demo.jsonl"
IID_DOC = "https://docs.xiaohongshu.com/doc/26af1aca8186f2e26e4ec1ddd27e4609"


def hi_json(args, stdin=None):
    p = subprocess.run(["hi", *args], input=stdin, capture_output=True, text=True, timeout=180)
    out = p.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        s = out.find("{")
        if s >= 0:
            return json.loads(out[s:])
        raise RuntimeError(f"hi {' '.join(args[:2])} -> {out[:200]} / {p.stderr[:200]}")


def op_code():
    return hi_json(["utils:generate-operate-code"])["operateCode"]


def fence(body, lang="text"):
    longest, run = 0, 0
    for ch in body:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return f"{'`'*max(3,longest+1)}{lang}\n{body}\n{'`'*max(3,longest+1)}"


def append(sid, md):
    for attempt in range(3):
        try:
            h = hi_json(["docs:get", "--shortcut-id", sid, "--mode", "common"])["hash"]
            hi_json(["docs:edit", "--shortcut-id", sid, "--hash", h, "--append", "-"], stdin=md)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)


def main():
    demos = [json.loads(l) for l in DEMO.open() if l.strip()]
    n_gen = sum(len(d["generations"]) for d in demos)
    n_anchor = sum(1 for d in demos for g in d["generations"] if "**Core idea:**" in g["cot"])

    intro = [
        "## Overview — OOD test on recent papers",
        "",
        "This is the **out-of-distribution** companion to the IID val-split demo "
        f"([here]({IID_DOC})). Here the same 7B v1 checkpoint is fed **recent papers** (arXiv Jan-2026, "
        "new problems it never trained on). For each paper, **claude-opus-4-8** wrote a leak-free "
        "`setup` (the problem/motivation, no solution revealed) and picked a **best-fit** and a "
        "**contrasting (wildcard)** researcher from the 125-roster; then the 7B checkpoint mocks each "
        "researcher on that setup at **temperature 0.8**.",
        "",
        f"**{len(demos)} papers x 2 researchers = {n_gen} generations.**",
        "",
        "### How the checkpoint behaves (honest read)",
        "- **Topical generalization: good.** It engages the actual new problem (e.g. gradient-spectrum "
        "anisotropy, masked-diffusion positional brittleness, SSM state-as-memory, federated multi-objective "
        "RL) rather than drifting to a memorized case.",
        "- **Researcher steering: holds up.** It adopts the assigned lens — Bernstein->norm/spectrum view, "
        "Albert Gu->SSM state-as-memory, Velickovic->locality-vs-globality, Dettmers->'account for the "
        "structure, don't fight it'.",
        "- **Idea sharpness: mixed.** Proposals are coherent and on-theme but often *soft/general* rather "
        "than a crisp non-trivial mechanism; novelty vs the real paper is not verified.",
        f"- **Format caveat (as in IID): the `Core idea:`/`Non-trivial crux:` anchors appear in only "
        f"{n_anchor}/{n_gen} generations** (1 epoch on pre-judge data). Expected to sharpen with the "
        "judged dataset + more epochs.",
        "",
        "## Papers",
    ]
    op = op_code()
    res = hi_json(["docs:create", "--title", "7B Researcher-CoT v1 — OOD Demo (recent papers)",
                   "--content", "-", "--space-id", SPACE_ID, "--operate-code", op], stdin="\n".join(intro))
    sid = res["shortcutId"]
    print(f"[redoc] created {res.get('url')}", flush=True)

    for i, d in enumerate(demos, 1):
        head = [
            "", f"## Paper {i} — {d['title']}",
            f"*{d['arxiv_id']} · {d['created']} · {(d.get('categories') or ['?'])[0]}*", "",
            f"**opus researcher picks:** best-fit = **{d['best_fit']}**, wildcard = **{d['wildcard']}**  ",
            f"*why: {d.get('why','')}*", "",
            "### setup (opus, leak-free)", fence(d["setup"]),
        ]
        append(sid, "\n".join(head))
        for g in d["generations"]:
            has = "**Core idea:**" in g["cot"]
            sec = [
                "", f"### {g['role']}: {g['researcher']} — generated CoT" + ("" if has else "  *(no explicit anchor)*"),
                fence(g["cot"]),
            ]
            append(sid, "\n".join(sec))
        print(f"[redoc] appended paper {i}/{len(demos)} ({d['arxiv_id']})", flush=True)

    print(json.dumps({"status": "ok", "url": res.get("url"), "shortcutId": sid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
