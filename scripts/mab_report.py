#!/usr/bin/env python3
"""Generate the MLAgentBench (MAB) eval report from the results/<task>.csv sink.

Mirrors mls_lite_report.py, but reads OUR own sink (MAB has no leaderboard). For each task we take the
baseline metric (mean over baseline seeds) as the reference, then for each arm compute the direction-signed
relative Δ-over-baseline (positive = improvement; mae is lower-is-better) per seed, and aggregate to
mean ± 95% CI across the arm's seeds. We also emit MAB's binary success flag: Δ >= +10% over baseline.

Usage:
  python scripts/mab_report.py --arms base,sft,rl \
    --out-md runs/researcher_cot/mab_eval/report.md --out-json runs/researcher_cot/mab_eval/report.json
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _mab_common as mab  # noqa: E402

SUCCESS_THRESHOLD_PCT = 10.0


def _vals(rows: list[dict], model: str) -> list[float]:
    out = []
    for r in rows:
        if r.get("model") == model:
            v = mab._f(r.get("metric_value"))
            if v is not None:
                out.append(v)
    return out


def _ci(deltas: list[float]) -> float | None:
    if len(deltas) < 2:
        return None
    return round(1.96 * statistics.pstdev(deltas) / (len(deltas) ** 0.5), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-json", default=str(ROOT / "docs/eval/mab_tasks.json"))
    ap.add_argument("--arms", default="base,sft,rl")
    ap.add_argument("--results-dir", default=None, help="default: runs/researcher_cot/mab_eval/results")
    ap.add_argument("--out-md", default=str(ROOT / "runs/researcher_cot/mab_eval/report.md"))
    ap.add_argument("--out-json", default=str(ROOT / "runs/researcher_cot/mab_eval/report.json"))
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks_json).read_text())
    arms = args.arms.split(",")
    slug2name = {t["slug"]: t["name"] for d in tasks["domains"].values() for t in d}
    dom_of = {t["slug"]: dom for dom, ts in tasks["domains"].items() for t in ts}

    rows, missing = [], 0
    for slug in tasks["slugs"]:
        res = mab.read_results(slug, args.results_dir)
        base_vals = _vals(res, "baseline")
        base = statistics.mean(base_vals) if base_vals else None
        row = {"task": slug, "name": slug2name.get(slug, slug), "domain": dom_of.get(slug, ""),
               "metric": mab.metric_name(slug), "lower_is_better": mab.lower_is_better(slug),
               "baseline": round(base, 5) if base is not None else None, "arms": {}}
        for arm in arms:
            avals = _vals(res, arm)
            deltas = [d for v in avals if (d := mab.signed_delta_pct(slug, base, v)) is not None] \
                if base is not None else []
            mean_d = round(statistics.mean(deltas), 3) if deltas else None
            row["arms"][arm] = {
                "metric_mean": round(statistics.mean(avals), 5) if avals else None,
                "n_seeds": len(avals),
                "delta_mean_pct": mean_d,
                "delta_ci_pct": _ci(deltas),
                "success": (mean_d is not None and mean_d >= SUCCESS_THRESHOLD_PCT),
            }
            if base is None or not avals:
                missing += 1
        rows.append(row)

    def summary(arm: str) -> dict:
        ds = [r["arms"][arm]["delta_mean_pct"] for r in rows
              if r["arms"][arm]["delta_mean_pct"] is not None]
        ns = sum(1 for r in rows if r["arms"][arm]["success"])
        return {"mean_delta_pct": round(statistics.mean(ds), 3) if ds else None,
                "n_success": ns, "n": len(rows)}

    summ = {arm: summary(arm) for arm in arms}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(
        {"rows": rows, "summary": summ, "arms": arms,
         "success_threshold_pct": SUCCESS_THRESHOLD_PCT, "missing_cells": missing},
        indent=2, ensure_ascii=False))

    # ---- markdown ----
    L = []
    L.append("<redoc-highlight emoji=\"dengpao\" fillColor=\"orange\">")
    L.append("**MLAgentBench (MAB) 评测：Δ-over-baseline（方向修正后的相对提升 %，正=更好；"
             f"success = Δ ≥ +{SUCCESS_THRESHOLD_PCT:.0f}%）。** 5 tasks，worker (fable-5) 忠实实现每个 arm 的 proposal。")
    means = " · ".join(f"{a} **{summ[a]['mean_delta_pct']}%** ({summ[a]['n_success']}/{summ[a]['n']} success)"
                       for a in arms)
    L.append(f"均值 Δ：{means}")
    L.append("</redoc-highlight>\n")
    L.append("## 每任务 Δ-over-baseline (%)\n")
    header = "| domain | task | metric | baseline | " + " | ".join(f"Δ{a}%" for a in arms) + " |"
    L.append(header)
    L.append("|---|---|---|--:|" + "|".join(["--:"] * len(arms)) + "|")
    for r in rows:
        cells = []
        for a in arms:
            d = r["arms"][a]["delta_mean_pct"]
            ci = r["arms"][a]["delta_ci_pct"]
            flag = " ✓" if r["arms"][a]["success"] else ""
            cells.append(f"{d}{'±'+str(ci) if ci is not None else ''}{flag}" if d is not None else "—")
        L.append(f"| {r['domain']} | {r['name']} | {r['metric']}"
                 f"{' ↓' if r['lower_is_better'] else ''} | {r['baseline']} | " + " | ".join(cells) + " |")
    mean_row = " | ".join(f"**{summ[a]['mean_delta_pct']}%**" for a in arms)
    succ_row = " | ".join(f"{summ[a]['n_success']}/{summ[a]['n']}" for a in arms)
    L.append(f"| | **均值 mean Δ** | | | {mean_row} |")
    L.append(f"| | **success (Δ≥{SUCCESS_THRESHOLD_PCT:.0f}%)** | | | {succ_row} |\n")
    if missing:
        L.append(f"<redoc-highlight emoji=\"gantanhao\" fillColor=\"red\">缺 {missing} 个 (arm,task) "
                 "结果（评测未跑完或缺 baseline）。</redoc-highlight>")
    Path(args.out_md).write_text("\n".join(L))
    print(f"wrote {args.out_md} + {args.out_json} | " +
          " ".join(f"{a}={summ[a]['mean_delta_pct']}%" for a in arms) + f" | missing={missing}")


if __name__ == "__main__":
    main()
