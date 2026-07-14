#!/usr/bin/env python3
"""Publish a documentation RedDoc for Part B3 (dataset refinement + stats) and Part C (creativity eval)."""
from __future__ import annotations
import json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = "4709bcf060916d61278c76e8a1b668e0"
STATS = json.loads((ROOT / "runs/training_data/v3_researcher_cot/dataset_stats.json").read_text())
IID = json.loads((ROOT / "runs/creativity_eval/demo_7b_creativity_summary.json").read_text())
OOD = json.loads((ROOT / "runs/creativity_eval/recent_demo_creativity_summary.json").read_text())


def hi(args, stdin=None):
    p = subprocess.run(["hi", *args], input=stdin, capture_output=True, text=True, timeout=180)
    o = p.stdout.strip()
    if "{" not in o:
        raise RuntimeError(f"hi {' '.join(args[:2])} returned no JSON. stdout={o[:300]} stderr={p.stderr[:300]}")
    return json.loads(o[o.find("{"):])


def kv(d):
    """render a dict without braces (RedDoc treats {} as JSX)."""
    return " · ".join(f"{k}×{v}" for k, v in d.items())


f = STATS["funnel"]; jd = STATS["judge_dims"]; tk = STATS["tokens"]
L = []
L += ["## Overview", "",
      "Documentation of **Part B3** (judged dataset refinement + statistics) and **Part C** (creativity "
      "evaluation harness) for the V3 human-researcher CoT distillation. The dataset trains a model to "
      "reconstruct, in a named researcher's voice, how they reached a real innovation (idea + non-trivial "
      "core + how they got there, no implementation details). Everything below is on git branch `V3`.", ""]

# ---- B3 ----
L += ["## Part B3 — Dataset refinement pipeline", "",
      "opus-4.8 synthesizes CoTs (3 reasoning routes per source case) + a separate opus-4.8 fact-check "
      "pass; then this refinement funnel produces the SFT set:", "",
      "| stage | count | what it does |",
      "|---|---|---|",
      f"| synthesized CoTs | {f['synthesized']} | opus-4.8, 596 source cases (综合≥8) × 3 routes, fact-checked |",
      f"| rule gate pass | {f['rule_pass']} | drop malformed: missing `Core idea:`/`Non-trivial crux:` anchor, code-fence, wrong length, non-English |",
      f"| opus-4.8 judge | {f['judged_ok']} | rubric grade each (faithfulness/clarity/non-triviality/creativity/fingerprint + impl-leak) |",
      f"| kept (quality) | {f['kept']} | keep = verdict keep, all sub-scores ≥4, fingerprint ≥3, no impl-leak, anchors present |",
      f"| **train / val** | **{f['train']} / {f['val']}** | after dedup; researcher named via conditioned system prompt |",
      "",
      f"Rule-gate drops: {kv(STATS['rule_fail_reasons'])}. Judge verdicts: {kv(STATS['verdicts'])}. "
      "Net drop from synthesis to final ≈ 3.7% — the fact-checked synth data was already strong, so the "
      "judge mostly *confirmed* quality.", "",
      "**Key refinement decision:** each SFT sample is conditioned with a researcher-named **system prompt** "
      "(\"You are [the researcher]…\"). This (a) makes the model steerable to mock a chosen researcher, and "
      "(b) means the cot need not self-name — so we do NOT drop CoTs for not self-naming (that alone would "
      "have wrongly cut ~950 good samples).", ""]

# ---- B3 stats ----
L += ["## Part B3 — Dataset properties & statistics", "",
      "### Judge quality (1–5) over 1,751 graded CoTs",
      "| dimension | mean | median | histogram |", "|---|---|---|---|"]
for d in ["faithfulness", "clarity_of_core", "non_triviality", "creativity", "fingerprint"]:
    x = jd[d]
    L.append(f"| {d} | {x['mean']} | {x['median']} | {kv(x['hist'])} |")
L += ["",
      "### Coverage",
      f"- **Researchers:** all **{STATS['n_researchers_kept']} / 125** represented; "
      f"{STATS['cots_per_researcher']['min']}–{STATS['cots_per_researcher']['max']} CoTs each "
      f"(mean {STATS['cots_per_researcher']['mean']}).",
      f"- **Reasoning routes (balanced):** {kv(STATS['routes'])}.",
      "", "| researcher class | CoTs |", "|---|---|"]
for k, v in STATS["by_researcher_category"].items():
    L.append(f"| {k} | {v} |")
L += ["",
      "### Token lengths (Qwen2.5-32B tokenizer)",
      "| segment | mean | min | max | p90 |", "|---|---|---|---|---|",
      f"| system (researcher-conditioned) | {tk['system']['mean']} | {tk['system']['min']} | {tk['system']['max']} | — |",
      f"| user (setup) | {tk['user_setup']['mean']} | {tk['user_setup']['min']} | {tk['user_setup']['max']} | — |",
      f"| assistant (cot) | {tk['assistant_cot']['mean']} | {tk['assistant_cot']['min']} | {tk['assistant_cot']['max']} | {tk['assistant_cot']['p90']} |",
      f"| whole sample | {tk['whole']['mean']} | {tk['whole']['min']} | {tk['whole']['max']} | {tk['whole']['p90']} |",
      "", "All samples fit comfortably in the training seq-len (2048 for 7B; 8–16k available).", ""]

# ---- Part C ----
L += ["## Part C — Creativity evaluation harness", "",
      "**This is evaluation tooling, NOT training** — no RL reward or gradient uses it (yet). Two parts:", "",
      "### (i) Intrinsic scorer — `scripts/score_creativity.py`",
      "opus-4.8 rubric over generated proposals/CoTs: **novelty / non-triviality / clarity-of-core / "
      "how-arrived / feasibility** (1–5). Cached + concurrent. **Run on the 7B v1 outputs:**", "",
      "| dimension | v1 IID (val) | v1 OOD (recent papers) |", "|---|---|---|"]
for d in ["novelty", "non_triviality", "clarity_of_core", "how_arrived", "feasibility"]:
    L.append(f"| {d} | {IID['per_dimension'][d]['mean']} | {OOD['per_dimension'][d]['mean']} |")
L += [f"| **overall key-creativity** | **{IID['overall_key_creativity_mean']}** | **{OOD['overall_key_creativity_mean']}** |",
      "",
      "**Read:** v1 is strongest on **how_arrived** (path-tracing, ~3.2–3.5) and weak on **novelty / "
      "clarity-of-core** (~2.2–2.8) — it learned the *format* of creative reasoning, not yet the *substance*. "
      "These low dimensions are the actionable targets for a v2 retrain (judged data + more epochs) and later RL.", "",
      "### (ii) Freer worker (extrinsic eval) — implemented, not yet run",
      "The extrinsic eval has a worker (Claude Code) implement a proposal and returns a signed metric. It was "
      "over-constrained; Part C relaxes it:",
      "- **`--free-hparams`**: the worker prompt's scope rule now lets it tune its own hyperparameters "
      "(lr/batch/epochs/warmup/optimizer) instead of being pinned by the proposal; it still may not change the "
      "metric, split, or reporting.",
      "- **`--max-turns` / `--worker-timeout`**: already existed — just pass larger values for a bigger budget.",
      "- Wired through `prompts.py`, `end_to_end.py`, `formal_sweep.py` + the 3 run scripts. The full extrinsic "
      "run needs a GPU worker node and is not executed yet.", "",
      "### Is Part C training or eval?",
      "**Eval.** The intrinsic scorer measures generated outputs; the freer worker is the evaluation harness. "
      "Neither trains the model. (The creativity dimensions could later become an RL reward, but that is not "
      "wired up now.)", ""]

content = "\n".join(L)
op = hi(["utils:generate-operate-code"])["operateCode"]
res = hi(["docs:create", "--title", "V3 — Dataset (B3) & Creativity Eval (Part C)",
          "--content", "-", "--space-id", SPACE, "--operate-code", op], stdin=content)
print(json.dumps({"status": "ok", "url": res.get("url"), "chars": len(content)}, ensure_ascii=False))


if __name__ == "__main__":
    pass
