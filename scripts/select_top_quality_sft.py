#!/usr/bin/env python3
"""Select the top-N highest-quality SFT rows by v3_target_audit.score.

Reads the train-all CoT SFT dataset (all 928 rows, each with a `messages` field)
and the per-article target audit, ranks by audit score (desc) then chronological
rank (asc, for stable ties), and writes a new sft-dir whose train.jsonl holds the
top-N rows. The output layout matches what scripts/prepare_v3_sft_run.py expects
(train.jsonl / val.jsonl / test.jsonl + summary.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "runs" / "training_data" / "v3_0_sft_qwen25_32b_v1sem_cot_strict929_schema_all_train_all"
DEFAULT_AUDIT = ROOT / "runs" / "training_data" / "v3_0_targets_qwen25_32b_strict_batch1000_audit" / "tex_targets.accepted.jsonl"
DEFAULT_OUT = ROOT / "runs" / "training_data" / "v3_0_sft_qwen25_32b_top500_audit"


def load_audit_scores(audit_path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with audit_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            aid = r.get("arxiv_id")
            score = (r.get("v3_target_audit") or {}).get("score")
            if aid is not None and score is not None:
                scores[aid] = float(score)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-n", type=int, default=500)
    args = ap.parse_args()

    src_all = args.src_dir / "all.jsonl"
    rows: list[dict[str, Any]] = [json.loads(l) for l in src_all.open()]
    audit = load_audit_scores(args.audit)

    missing = [r["arxiv_id"] for r in rows if r["arxiv_id"] not in audit]
    if missing:
        print(f"[select] WARNING: {len(missing)} rows lack an audit score (treated as 0): {missing[:5]}")

    rows.sort(key=lambda r: (-audit.get(r["arxiv_id"], 0.0), r.get("chronological_rank", 1 << 30)))
    top = rows[: args.top_n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "all.jsonl").open("w") as f, (args.out_dir / "train.jsonl").open("w") as g:
        for r in top:
            line = json.dumps(r, ensure_ascii=False)
            f.write(line + "\n")
            g.write(line + "\n")
    # phase-by none uses only train.jsonl; write empty val/test so prepare can stat them.
    (args.out_dir / "val.jsonl").write_text("")
    (args.out_dir / "test.jsonl").write_text("")

    scores_sel = [audit.get(r["arxiv_id"], 0.0) for r in top]
    summary = {
        "source_sft_dir": str(args.src_dir),
        "audit": str(args.audit),
        "top_n": args.top_n,
        "selected_count": len(top),
        "audit_score_min": min(scores_sel) if scores_sel else None,
        "audit_score_max": max(scores_sel) if scores_sel else None,
        "audit_score_mean": round(sum(scores_sel) / len(scores_sel), 3) if scores_sel else None,
        "score_histogram": {
            s: sum(1 for x in scores_sel if x == s) for s in sorted(set(scores_sel), reverse=True)
        },
        "selection": "sort by v3_target_audit.score desc, chronological_rank asc; top-N into train.jsonl",
        "note": "train_all style: all selected rows go to train.jsonl; val/test empty (use --phase-by none).",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", **{k: summary[k] for k in (
        "selected_count", "audit_score_min", "audit_score_max", "audit_score_mean")},
        "out_dir": str(args.out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
