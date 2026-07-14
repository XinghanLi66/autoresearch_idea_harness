#!/usr/bin/env python3
"""Re-organize the judged CoT dataset into the two-anchor format for RL/format-reward.

Every assistant target is guaranteed to carry TWO explicit anchors:
  1. researcher anchor  -> a leading "**Mocking:** <researcher>" line (which researcher it mocks)
  2. novelty/crux anchor -> trailing "**Core idea:** ..." and "**Non-trivial crux:** ..." (already
     present in the judged set; the novelty statement lives here)

Also defines `format_score(text, researcher)` — the rule-based FORMAT REWARD used both to validate
this dataset and (later) inside the RL reward: it checks all anchors are present and the mocked
researcher matches the target.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[一-鿿]")
MOCK_RE = re.compile(r"\*\*Mocking:\*\*\s*(.+)")


def name_tokens(name: str) -> list[str]:
    toks = [t for t in re.split(r"[\s()·,]+", name) if len(t) >= 2 and not CJK.search(t)]
    cjk = "".join(CJK.findall(name))
    if cjk:
        toks.append(cjk)
    return toks


def format_score(text: str, researcher: str) -> dict:
    """Rule-based format reward in [0,1] with a breakdown. All three anchors + name match = 1.0."""
    has_mock = bool(MOCK_RE.search(text))
    has_core = "**Core idea:**" in text
    has_crux = "**Non-trivial crux:**" in text
    # mocked name matches target?
    name_ok = False
    m = MOCK_RE.search(text)
    if m:
        line = m.group(1)
        name_ok = any(t in line for t in name_tokens(researcher))
    parts = {"mocking_anchor": has_mock, "name_match": name_ok,
             "core_idea_anchor": has_core, "crux_anchor": has_crux}
    score = (0.3 * has_mock + 0.2 * name_ok + 0.25 * has_core + 0.25 * has_crux)
    return {"score": round(score, 3), **parts}


def anchor_assistant(cot: str, researcher: str) -> str:
    """Prepend the researcher anchor if absent; the core/crux anchors are already in the judged cot."""
    if MOCK_RE.search(cot.splitlines()[0] if cot else ""):
        return cot
    return f"**Mocking:** {researcher}\n\n{cot}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT / "runs/training_data/v3_researcher_cot")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/training_data/v3_researcher_cot_anchored")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"splits": {}, "format_reward_coverage": {}}
    all_scores = []
    for split in ["train", "val"]:
        src = args.src / f"{split}.jsonl"
        if not src.exists():
            continue
        rows = [json.loads(l) for l in src.open()]
        out_rows = []
        for r in rows:
            msgs = r["messages"]
            researcher = r["researcher"]
            msgs[2]["content"] = anchor_assistant(msgs[2]["content"], researcher)
            fs = format_score(msgs[2]["content"], researcher)
            all_scores.append(fs["score"])
            r["format_reward"] = fs
            out_rows.append(r)
        with (args.out / f"{split}.jsonl").open("w") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["splits"][split] = len(out_rows)

    n = len(all_scores)
    perfect = sum(1 for s in all_scores if s >= 0.999)
    report["format_reward_coverage"] = {
        "total": n, "mean_format_score": round(sum(all_scores) / n, 4) if n else 0,
        "all_anchors_present_and_matched": perfect, "pct_perfect": round(perfect / n * 100, 1) if n else 0,
    }
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
