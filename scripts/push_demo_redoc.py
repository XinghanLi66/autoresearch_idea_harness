#!/usr/bin/env python3
"""Publish the 7B v1 demo + recipe to a single RedDoc page (Agentic Training space).

Creates a page with Overview + Recipe (small), then appends one section per demo (full prompt,
generated CoT, answer) as collapsible code blocks — appended one at a time to avoid the 504 that
hits large single creates.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "4709bcf060916d61278c76e8a1b668e0"  # Agentic Training
DEMO = ROOT / "runs" / "researcher_cot" / "demo_7b" / "demo.jsonl"
RECIPE = ROOT / "runs" / "training" / "v3_researcher_cot_7b" / "full" / "train_summary.json"


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
    bar = "`" * max(3, longest + 1)
    return f"{bar}{lang}\n{body}\n{bar}"


def main():
    r = json.loads(RECIPE.read_text())
    demos = [json.loads(l) for l in DEMO.open() if l.strip()]
    n_anchor = sum(1 for d in demos if d.get("answer_core_idea"))

    intro = [
        "## Overview",
        "",
        "**7B researcher-mocking CoT — v1 (pre-judge).** Qwen2.5-7B-Instruct LoRA-SFT'd on "
        f"{r['num_examples']} first-person CoTs that reconstruct how top AI researchers reached their key "
        "innovations (distilled by claude-opus-4-8 from the RedDoc 蒸馏计划 cases, source-quality ≥ 8/10, "
        "**before** the opus judge gate). Each training sample is conditioned with a researcher-named "
        "system prompt, so the model can be steered to mock a chosen researcher.",
        "",
        f"Demo below: the checkpoint mocking **{len(demos)} different researchers** on held-out papers "
        f"(val split, not trained on), generated at **temperature {demos[0]['temperature']}**.",
        "",
        "**Result:** distinct, on-target researcher voices — e.g. Sutton→temporal-difference bootstrapping, "
        "Hinton→dropout (break co-adaptations), LeCun→end-to-end convnets, Ilya→the scaling bet, "
        "Hassabis→Go-as-prediction. Researcher steering works.",
        f"**v1 caveat:** the explicit `Core idea:` / `Non-trivial crux:` anchors appear in only "
        f"{n_anchor}/{len(demos)} generations (format not fully learned from 1 epoch of pre-judge data); "
        "the idea is still stated inline. Expected to improve with the judged dataset + more epochs.",
        "",
        "## Recipe",
        "",
        "| param | value |",
        "|---|---|",
        f"| base model | Qwen2.5-7B-Instruct |",
        f"| finetune | LoRA (r={r['lora_r']}, alpha={r['lora_alpha']}, target={r['lora_target_modules']}) |",
        f"| learning rate | {r['learning_rate']} |",
        f"| lr scheduler | cosine, warmup_ratio 0.03 |",
        f"| weight decay | 0.01 |",
        f"| precision | bf16 + gradient checkpointing |",
        f"| max_seq_length | {r['max_seq_length']} |",
        f"| batch (per-device x grad-accum) | {r['per_device_batch_size']} x {r['grad_accum']} = "
        f"{r['effective_global_batch']} effective |",
        f"| epochs | {r['num_epochs']} |",
        f"| dataset size | {r['num_examples']} train (+ val); pre-judge, rule-gate only |",
        f"| train time | {round(r['train_elapsed_s'])}s on 1x L20Z (80GB) |",
        f"| peak GPU memory | {r['peak_gpu_memory_gb']} GB |",
        f"| **test/inference** | **temperature 0.8**, top_p 0.95, max_new_tokens 1500 |",
        "",
        "## Demos",
        "",
        "_One section per researcher below (full prompt, generated CoT, answer)._",
    ]

    sid = None
    op = op_code()
    res = hi_json(["docs:create", "--title", "7B Researcher-CoT v1 — Demo & Recipe",
                   "--content", "-", "--space-id", SPACE_ID, "--operate-code", op], stdin="\n".join(intro))
    sid = res["shortcutId"]
    print(f"[redoc] created {res.get('url')}", flush=True)

    for i, d in enumerate(demos, 1):
        ans = d.get("answer_core_idea") or "(no explicit anchor in this generation; core idea stated inline in the CoT tail)"
        crux = d.get("answer_nontrivial_crux") or ""
        sec = [
            "", f"## Demo {i} — mocking {d['researcher']}",
            f"*held-out case: {d.get('case_title')}*", "",
            "### full prompt", fence(f"[SYSTEM]\n{d['system']}\n\n[USER]\n{d['user']}"),
            "", "### generated CoT", fence(d["generated_cot"]),
            "", "### answer", f"**Core idea:** {ans}",
        ]
        if crux:
            sec.append(f"\n**Non-trivial crux:** {crux}")
        # re-fetch hash, append
        h = hi_json(["docs:get", "--shortcut-id", sid, "--mode", "common"])["hash"]
        for attempt in range(3):
            try:
                hi_json(["docs:edit", "--shortcut-id", sid, "--hash", h, "--append", "-"], stdin="\n".join(sec))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(3)
                h = hi_json(["docs:get", "--shortcut-id", sid, "--mode", "common"])["hash"]
        print(f"[redoc] appended demo {i}/{len(demos)} ({d['researcher']})", flush=True)

    print(json.dumps({"status": "ok", "url": res.get("url"), "shortcutId": sid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
