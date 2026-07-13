#!/usr/bin/env python3
"""Crawl the RedDoc "人类researcher 蒸馏计划" roster into a local CoT-synthesis pool.

For each researcher (parent doc) it collects:
  - the 第二步 Skills-v1 doc (methodology fingerprint)
  - every 第一步 case subdoc (①problem → ②idea → ③ablation → ④reflection, with quality scores)
  - arXiv IDs referenced by the cases (the highest-signal "signature pubs" for the pilot)

Output: runs/researcher_cot/pool/pool.jsonl (one record per researcher) + a summary.
All RedDoc fetches go through the `hi` CLI and are cached on disk so re-runs are cheap.

Usage:
  python scripts/build_researcher_cot_pool.py --parent-ids <id> <id> ...   # explicit subset (pilot)
  python scripts/build_researcher_cot_pool.py --limit 5                    # first N from roster
  python scripts/build_researcher_cot_pool.py --all                        # all 125
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ROSTER_ID = "f96e8ad56b7a7c9353c08749e8e1cf89"
CACHE_DIR = ROOT / "runs" / "researcher_cot" / "cache"
OUT_DIR = ROOT / "runs" / "researcher_cot" / "pool"
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def run_hi(args: list[str], retries: int = 4) -> dict[str, Any]:
    """Call the hi CLI and parse its JSON stdout, retrying transient failures
    (the node/nvm CLI intermittently errors with 'Cannot find package ...')."""
    last = ""
    for attempt in range(retries):
        try:
            proc = subprocess.run(["hi", *args], capture_output=True, text=True, timeout=120)
            out = proc.stdout.strip()
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                start = out.find("{")
                if start >= 0:
                    return json.loads(out[start:])
                last = f"non-JSON: {out[:200]} / err {proc.stderr[:200]}"
        except Exception as e:
            last = str(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"hi {' '.join(args[:2])} failed after {retries} tries: {last}")


def get_doc_md(shortcut_id: str) -> str:
    """docs:get a doc, return its markdown (cached under CACHE_DIR/<id>.md)."""
    cache = CACHE_DIR / f"{shortcut_id}.md"
    if cache.exists():
        return cache.read_text()
    res = run_hi(["docs:get", "--shortcut-id", shortcut_id, "--mode", "common"])
    md = Path(res["mdPath"]).read_text()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(md)
    return md


def menu(shortcut_id: str) -> list[dict[str, Any]]:
    cache = CACHE_DIR / f"{shortcut_id}.menu.json"
    if cache.exists():
        return json.loads(cache.read_text())
    res = run_hi(["docs:menu-list", "--shortcut-id", shortcut_id, "--all"])
    lst = res.get("list", [])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(lst, ensure_ascii=False))
    return lst


def parse_roster(md: str) -> list[dict[str, str]]:
    """Return [{name, style, parent_id, category}] tracking the current section header."""
    out: list[dict[str, str]] = []
    category = ""
    for line in md.splitlines():
        h = re.match(r"^###\s+(.*)$", line)
        if h:
            category = h.group(1).strip()
            continue
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*\[https://docs\.xiaohongshu\.com/doc/([0-9a-f]+)\]", line)
        if m and m.group(1) not in ("人物",):
            out.append({"name": m.group(1).strip(), "style": m.group(2).strip(),
                        "parent_id": m.group(3), "category": category})
    return out


def extract_arxiv_ids(text: str) -> list[str]:
    ids = []
    for m in ARXIV_RE.findall(text):
        if m not in ids:
            ids.append(m)
    return ids


def parse_overall_score(case_md: str) -> tuple[str | None, float | None]:
    """Parse the case's overall quality score. Formats seen:
       '... ｜ 综合 **10/10**'  and  '> 质量分 **9/10** ...'  and bare '**8/10**' in the header."""
    head = case_md[:800]
    pats = [
        r"综合\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"质量分\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"\*\*\s*(\d+(?:\.\d+)?)\s*/\s*10\s*\*\*",  # any bolded X/10 near the top
        r"(\d+(?:\.\d+)?)\s*/\s*10",                  # last resort: first X/10 in the header
    ]
    for p in pats:
        m = re.search(p, head)
        if m:
            val = float(m.group(1))
            return f"{m.group(1)}/10", val
    return None, None


def crawl_researcher(r: dict[str, str]) -> dict[str, Any]:
    children = menu(r["parent_id"])
    case_index = next((c for c in children if "第一步" in c["title"] or "案例索引" in c["title"]), None)
    skills = next((c for c in children if "第二步" in c["title"] or "Skills" in c["title"]), None)

    skills_rec = None
    if skills:
        skills_rec = {"shortcut_id": skills["shortcutId"], "title": skills["title"],
                      "md": get_doc_md(skills["shortcutId"])}

    cases: list[dict[str, Any]] = []
    if case_index:
        for c in menu(case_index["shortcutId"]):
            md = get_doc_md(c["shortcutId"])
            score_str, score_num = parse_overall_score(md)
            cases.append({
                "shortcut_id": c["shortcutId"], "title": c["title"], "md": md,
                "overall_score": score_str, "overall_score_num": score_num,
                "arxiv_ids": extract_arxiv_ids(md),
            })

    all_arxiv = []
    for c in cases:
        for a in c["arxiv_ids"]:
            if a not in all_arxiv:
                all_arxiv.append(a)

    return {
        "name": r["name"], "style": r["style"], "category": r["category"],
        "parent_id": r["parent_id"],
        "skills": skills_rec,
        "cases": cases,
        "case_arxiv_ids": all_arxiv,
        "n_cases": len(cases),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster-id", default=ROSTER_ID)
    ap.add_argument("--parent-ids", nargs="*", default=None, help="explicit researcher parent doc ids (pilot)")
    ap.add_argument("--limit", type=int, default=None, help="first N researchers from the roster")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--out-name", default="pool.jsonl")
    args = ap.parse_args()

    roster = parse_roster(get_doc_md(args.roster_id))
    print(f"[pool] roster parsed: {len(roster)} researchers", flush=True)

    if args.parent_ids:
        sel = [r for r in roster if r["parent_id"] in set(args.parent_ids)]
        # keep any explicitly-requested ids even if absent from roster parse
        found = {r["parent_id"] for r in sel}
        for pid in args.parent_ids:
            if pid not in found:
                sel.append({"name": pid, "style": "", "category": "", "parent_id": pid})
    elif args.all:
        sel = roster
    else:
        sel = roster[: (args.limit or 5)]
    print(f"[pool] selected {len(sel)} researchers", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.out_name
    records = []
    with out_path.open("w") as f:
        for i, r in enumerate(sel, 1):
            try:
                rec = crawl_researcher(r)
            except Exception as e:
                print(f"[pool] {i}/{len(sel)} {r['name']}: ERROR {e}", flush=True)
                continue
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[pool] {i}/{len(sel)} {rec['name']}: {rec['n_cases']} cases, "
                  f"{len(rec['case_arxiv_ids'])} arxiv ids, skills={'y' if rec['skills'] else 'n'}", flush=True)

    summary = {
        "researchers": len(records),
        "total_cases": sum(r["n_cases"] for r in records),
        "total_case_arxiv_ids": sum(len(r["case_arxiv_ids"]) for r in records),
        "researchers_missing_skills": [r["name"] for r in records if not r["skills"]],
        "researchers_zero_cases": [r["name"] for r in records if r["n_cases"] == 0],
        "out_path": str(out_path),
    }
    (args.out_dir / "pool_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
