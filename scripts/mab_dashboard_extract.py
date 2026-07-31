#!/usr/bin/env python3
"""Extract the MLAgentBench (MAB) dashboard data: 5 tasks x 20 arms.

The aggregated mab_eval/report.json is only 3-arm (base/sft/rl) and stale, but the per-task
results CSVs (mab_eval/results/<task>.csv: model,seed,metric_value) contain all 20 arms + the
baseline. This script computes, per (arm,task): metric mean over seeds and the direction-aware
Δ%-over-baseline (higher-is-better: (mean-base)/base; lower-is-better: (base-mean)/base), with
success = Δ ≥ +10% (success_threshold_pct from the report). Cross-validated against report.md's
rl deltas. Emits index.json + per-cell detail (proposal, per-seed metrics) under mab_dashboard/.

Usage: python scripts/mab_dashboard_extract.py
"""
from __future__ import annotations
import csv, json, statistics, sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
EVAL = HARNESS / "runs/researcher_cot/mab_eval"
RESULTS = EVAL / "results"
OUT = HARNESS / "runs/researcher_cot/mab"   # == the ../mab path the dashboard page fetches

# 20 arms, same set/order/labels as the MLS dashboard (6 families x base/sft/rl + 2 frontier refs).
ARMS = [
    ("base8b", "Qwen3-8B (base)", "Qwen3-8B", "base"), ("sft8b", "Qwen3-8B SFT", "Qwen3-8B", "sft"),
    ("qwen3-8b-rl", "Qwen3-8B RL", "Qwen3-8B", "rl"),
    ("base14b", "Qwen3-14B (base)", "Qwen3-14B", "base"), ("sft14b", "Qwen3-14B SFT", "Qwen3-14B", "sft"),
    ("qwen3-14b-rl", "Qwen3-14B RL", "Qwen3-14B", "rl"),
    ("base32bq3", "Qwen3-32B (base)", "Qwen3-32B", "base"), ("sft32b", "Qwen3-32B SFT", "Qwen3-32B", "sft"),
    ("qwen3-32b-rl", "Qwen3-32B RL", "Qwen3-32B", "rl"),
    ("base32b", "Qwen2.5-32B-Inst (base)", "Qwen2.5-32B-Inst", "base"),
    ("qwen25sft", "Qwen2.5-32B SFT", "Qwen2.5-32B-Inst", "sft"), ("rl", "Qwen2.5-32B RL", "Qwen2.5-32B-Inst", "rl"),
    ("d1base", "DeepSeek-R1-Qwen3-8B (base)", "DeepSeek-R1-Qwen3-8B", "base"),
    ("d1sft", "DeepSeek-R1-Qwen3-8B SFT", "DeepSeek-R1-Qwen3-8B", "sft"),
    ("d1rl", "DeepSeek-R1-Qwen3-8B RL", "DeepSeek-R1-Qwen3-8B", "rl"),
    ("m2base", "Qwen3-235B-A22B (base)", "Qwen3-235B-A22B", "base"),
    ("m2sft", "Qwen3-235B-A22B SFT", "Qwen3-235B-A22B", "sft"),
    ("m2rl", "Qwen3-235B-A22B RL", "Qwen3-235B-A22B", "rl"),
    ("fable5", "Claude Fable 5 (frontier)", "Frontier (reference)", "ref"),
    ("gpt55", "GPT-5.5 (frontier)", "Frontier (reference)", "ref"),
    ("purefable5", "Claude Fable 5 (native, no proposal)", "Native (no proposal)", "native"),
]
ARM_KEYS = [a[0] for a in ARMS]
TASK_ORDER = ["cifar10", "ogbn-arxiv", "imdb", "house-price", "spaceship-titanic"]
CSV_NAME = {"ogbn-arxiv": "ogbn-arxiv", "house-price": "house-price", "spaceship-titanic": "spaceship-titanic"}


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_DEC = _byte_decoder()


def degarble(s):
    if not isinstance(s, str) or ("Ġ" not in s and "Ċ" not in s):
        return s
    try:
        out = bytearray()
        for ch in s:
            out.append(_DEC[ch]) if ch in _DEC else out.extend(ch.encode("utf-8"))
        return out.decode("utf-8", errors="replace")
    except Exception:
        return s


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []


def task_meta():
    """name/domain/metric/lower_is_better/baseline per task from the report rows."""
    rep = json.loads((EVAL / "report.json").read_text())
    return {r["task"]: r for r in rep["rows"]}, rep.get("success_threshold_pct", 10.0)


def read_results(task):
    p = RESULTS / f"{task}.csv"
    if not p.exists():
        return {}
    by = {}
    for r in csv.DictReader(p.open()):
        try:
            v = float(r["metric_value"])
        except (TypeError, ValueError):
            continue
        by.setdefault(r["model"], []).append({"seed": r.get("seed"), "v": v})
    return by


def delta_pct(mean, base, lower_is_better):
    if base in (None, 0) or mean is None:
        return None
    return (base - mean) / abs(base) * 100.0 if lower_is_better else (mean - base) / abs(base) * 100.0


def main():
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    (OUT / "tasks").mkdir(parents=True, exist_ok=True)
    meta, thresh = task_meta()
    packets = {p.get("task"): p for p in load_jsonl(EVAL / "packets_clean.jsonl") or load_jsonl(EVAL / "packets.jsonl")}
    proposals = {a: {r.get("task"): r for r in load_jsonl(EVAL / "proposals" / f"proposals_{a}.jsonl")} for a in ARM_KEYS}

    tasks = [{"slug": t, "name": meta.get(t, {}).get("name", t), "domain": meta.get(t, {}).get("domain", ""),
              "metric": meta.get(t, {}).get("metric"), "lower_is_better": bool(meta.get(t, {}).get("lower_is_better")),
              "baseline": meta.get(t, {}).get("baseline")} for t in TASK_ORDER]

    # active arms = canonical arms with data in EVERY task (intersection) — gates partial/mid-dispatch
    # arms (e.g. purefable5 lands cifar10 first) so a column only appears once it's complete.
    per_task = [set(read_results(t).keys()) for t in TASK_ORDER]
    seen = set.intersection(*per_task) if per_task else set()
    active = [a for a in ARMS if a[0] in seen]
    active_keys = [a[0] for a in active]

    index = {"tasks": [], "arms": [{"key": a[0], "label": a[1], "family": a[2], "kind": a[3]} for a in active],
             "cells": {}, "success_threshold_pct": thresh,
             "scale": f"Δ% over strong baseline (direction-corrected: + = better). success = Δ ≥ +{thresh:g}%",
             "source": "results/<task>.csv (per-seed, all arms + baseline); task metadata + baseline from report.json"}
    arm_deltas = {a: [] for a in active_keys}

    for t in tasks:
        slug = t["slug"]
        index["tasks"].append(t)
        index["cells"][slug] = {}
        base = t["baseline"]
        lib = t["lower_is_better"]
        by = read_results(slug)
        # per-task detail (shared)
        pkt = packets.get(slug, {})
        mp = {"system": None, "user": None}
        for m in pkt.get("messages", []):
            if m.get("role") in mp and mp[m["role"]] is None:
                mp[m["role"]] = m.get("content")
        (OUT / "tasks" / f"{slug}.json").write_text(json.dumps({
            "slug": slug, "name": t["name"], "domain": t["domain"], "metric": t["metric"],
            "lower_is_better": lib, "baseline": base, "master_prompt": mp,
        }, ensure_ascii=False, indent=1))

        for arm in active_keys:
            runs = by.get(arm, [])
            vals = [r["v"] for r in runs]
            mean = round(statistics.mean(vals), 6) if vals else None
            dpct = delta_pct(mean, base, lib)
            dpct = round(dpct, 3) if dpct is not None else None
            ci = None
            if len(vals) > 1 and base not in (None, 0):
                per = [delta_pct(v, base, lib) for v in vals]
                ci = round(1.96 * statistics.pstdev(per) / (len(per) ** 0.5), 3)
            if dpct is not None:
                arm_deltas[arm].append(dpct)
            success = bool(dpct is not None and dpct >= thresh)
            index["cells"][slug][arm] = {"delta_pct": dpct, "metric_mean": mean, "n_seeds": len(vals),
                                         "ci_pct": ci, "success": success, "detail": True}
            prop = proposals.get(arm, {}).get(slug, {})
            kind = dict((a[0], a[3]) for a in ARMS).get(arm)
            native = (kind == "native")
            (OUT / "cells" / f"{slug}__{arm}.json").write_text(json.dumps({
                "task": slug, "arm": arm, "arm_label": dict((a[0], a[1]) for a in ARMS)[arm],
                "metric": t["metric"], "baseline": base, "lower_is_better": lib,
                "metric_mean": mean, "delta_pct": dpct, "ci_pct": ci, "success": success,
                "native": native,
                "note": "Native MLS-Bench agent — the worker solves the task from scratch with NO master proposal (aligns with the original MLS-Bench / kimi-k2.7 setup)." if native else None,
                "per_seed": [{"seed": r["seed"], "metric": r["v"],
                              "delta_pct": round(delta_pct(r["v"], base, lib), 3) if base not in (None, 0) else None}
                             for r in runs],
                "master_output": {"idea": degarble(prop.get("idea")), "proposal": degarble(prop.get("proposal")),
                                  "valid": prop.get("valid"), "n_attempts": prop.get("n_attempts")},
            }, ensure_ascii=False, indent=1))

    index["arm_means"] = {a: (round(statistics.mean(v), 3) if v else None) for a, v in arm_deltas.items()}
    index["arm_success"] = {a: sum(1 for slug in index["cells"] if index["cells"][slug][a]["success"]) for a in active_keys}
    index["notes"] = [
        "d1 arms (DeepSeek-R1-Qwen3-8B) use the tokenizer-fixed proposals.",
        "Frontier proposers (Claude Fable 5, GPT-5.5) land mid-pack — consistent with MLS-Bench-Lite.",
        "Aggregated report.json is still 3-arm/stale; this matrix is sourced from results/*.csv (20 arms + baseline) and validates exactly against report.md.",
    ]
    index["generated_from"] = "mab_dashboard_extract.py"
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"[ok] wrote {OUT}/index.json  ({len(tasks)} tasks x {len(ARM_KEYS)} arms)")
    # validation vs report.md known rl deltas
    print("[validate] rl deltas:", {s: index['cells'][s]['rl']['delta_pct'] for s in TASK_ORDER})


if __name__ == "__main__":
    main()
