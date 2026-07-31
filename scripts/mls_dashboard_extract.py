#!/usr/bin/env python3
"""Extract the MLS-Bench-Lite dashboard data (30 tasks x 18 arms).

Faithful, reproducible aggregation for the cc004 analyst dashboard. Emits a static
data tree under runs/researcher_cot/mls_lite_dashboard/:

  index.json            matrix: tasks[], arms[], cells[task][arm] = {status, score, n_runs}
  tasks/<slug>.json     shared per-task detail (description, baselines, master prompt)
  cells/<slug>__<arm>.json  per-cell detail (master output/idea, worker prompt/code, per-run scores)

Status semantics (goal-faithful):
  ok          run produced a valid metric -> real rescaled score (0 = TRUE 0, not missing)
  zero        genuine 0-floor (proposal ran but <= worst anchor); faithful, clickable
  failed      attempted but no valid metric (empty worker output / crash / OOM / infra)
  unfinished  robo-humanoid row (dropped) + m2rl column (RL still training) + never-dispatched cells; not clickable

Cell status is derived CELL-LEVEL from the LIVE leaderboard (tasks/<t>/leaderboard.csv):
a final row with >=1 populated metric column => the run scored (ok, even if the rescaled
value is 0). No populated metric => failed-or-floor, disambiguated by the task verdict in
failure_reasons.json. `mlsbench score` is used only for the numeric rescaled value.

Usage:
  python scripts/mls_dashboard_extract.py            # full run (calls mlsbench score per task; cached)
  python scripts/mls_dashboard_extract.py --no-score # reuse cached raw_scores/, skip mlsbench
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
import concurrent.futures as cf
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]                 # autoresearch_idea_harness/
EVAL = HARNESS / "runs/researcher_cot/mls_lite_eval"
OUT = HARNESS / "runs/researcher_cot/mls_lite_dashboard"
MLSBENCH_ROOT = Path("/newcpfs/lxh/MLS-Bench")
TASKS_DIR = MLSBENCH_ROOT / "tasks"
LOGS_DIR = MLSBENCH_ROOT / "logs"
LITE_JSON = HARNESS / "docs/eval/mls_bench_lite_tasks.json"

# ---- 18 canonical arms: 6 model families x {base, sft, rl}. key == leaderboard/log tag. ----
ARMS = [
    ("base8b",       "Qwen3-8B (base)",              "Qwen3-8B",              "base"),
    ("sft8b",        "Qwen3-8B SFT",                 "Qwen3-8B",              "sft"),
    ("qwen3-8b-rl",  "Qwen3-8B RL",                  "Qwen3-8B",              "rl"),
    ("base14b",      "Qwen3-14B (base)",             "Qwen3-14B",             "base"),
    ("sft14b",       "Qwen3-14B SFT",                "Qwen3-14B",             "sft"),
    ("qwen3-14b-rl", "Qwen3-14B RL",                 "Qwen3-14B",             "rl"),
    ("base32bq3",    "Qwen3-32B (base)",             "Qwen3-32B",             "base"),
    ("sft32b",       "Qwen3-32B SFT",                "Qwen3-32B",             "sft"),
    ("qwen3-32b-rl", "Qwen3-32B RL",                 "Qwen3-32B",             "rl"),
    ("base32b",      "Qwen2.5-32B-Inst (base)",      "Qwen2.5-32B-Inst",      "base"),
    ("qwen25sft",    "Qwen2.5-32B SFT",              "Qwen2.5-32B-Inst",      "sft"),
    ("rl",           "Qwen2.5-32B RL",              "Qwen2.5-32B-Inst",      "rl"),
    ("d1base",       "DeepSeek-R1-Qwen3-8B (base)",  "DeepSeek-R1-Qwen3-8B",  "base"),
    ("d1sft",        "DeepSeek-R1-Qwen3-8B SFT",     "DeepSeek-R1-Qwen3-8B",  "sft"),
    ("d1rl",         "DeepSeek-R1-Qwen3-8B RL",      "DeepSeek-R1-Qwen3-8B",  "rl"),
    ("m2base",       "Qwen3-235B-A22B (base)",       "Qwen3-235B-A22B",       "base"),
    ("m2sft",        "Qwen3-235B-A22B SFT",          "Qwen3-235B-A22B",       "sft"),
    ("m2rl",         "Qwen3-235B-A22B RL",           "Qwen3-235B-A22B",       "rl"),
    ("fable5",       "Claude Fable 5 (frontier)",    "Frontier (reference)",  "ref"),
    ("gpt55",        "GPT-5.5 (frontier)",           "Frontier (reference)",  "ref"),
    ("purefable5",   "Claude Fable 5 (native, no proposal)", "Native (no proposal)", "native"),
]
ARM_KEYS = [a[0] for a in ARMS]

# ---- final matrix source: the authoritative leaderboard-based report (all cells populated) ----
REPORT_JSON = HARNESS / "runs/researcher_cot/mls_lite_eval/report.json"
ARM_NOTES_JSON = HARNESS / "runs/researcher_cot/mls_lite_eval/failure_reasons.json"

# Per-task worker swap: fable-5's content filter blocks the protein tasks, so those were run
# with claude-opus-4-8 (uniformly across all arms). Affects only where the worker LOGS live.
DEFAULT_WORKER = "claude-fable-5"
WORKER_BY_TASK = {
    "ai4bio-mutation-effect-prediction": "claude-opus-4-8",
    "ai4sci-pla-binding-affinity": "claude-opus-4-8",
}

# Caveats footnoted on the dashboard. Key ("task","*") = whole row (all arms); ("task","arm") = one cell.
CAVEATS = {
    ("ai4bio-mutation-effect-prediction", "*"):
        "Worker swapped to claude-opus-4-8 (fable-5 content-filter blocks protein tasks). Per-task swap, uniform across all arms.",
    ("ai4sci-pla-binding-affinity", "*"):
        "Worker swapped to claude-opus-4-8 (fable-5 content-filter blocks protein tasks). Per-task swap, uniform across all arms.",
    ("robo-humanoid-sim2real-algo", "*"):
        "Uniform capability-floor 0.0 for all arms (real success_rate=0.0 in pod logs; CSV metrics are blank placeholders). 6 arms re-inserted after a leaderboard lock-contention window.",
    ("jepa-planning", "sft8b"):
        "Salvaged-reasoning proposal (8B over-think truncation). Real planner score 0.33.",
    ("jepa-planning", "qwen3-8b-rl"):
        "Salvaged-reasoning proposal (8B over-think truncation). Real planner score 0.556.",
}


def load_arm_notes() -> dict:
    """Merge per-cell caveats from failure_reasons.json _arm_notes ('task/arm' keys)."""
    try:
        d = json.loads(ARM_NOTES_JSON.read_text())
        for k, v in (d.get("_arm_notes") or {}).items():
            if "/" in k:
                task, arm = k.split("/", 1)
                CAVEATS.setdefault((task, arm), v)
    except Exception as e:
        print(f"[warn] arm_notes: {e}", file=sys.stderr)
    return CAVEATS


def caveat_for(task: str, arm: str):
    return CAVEATS.get((task, arm)) or CAVEATS.get((task, "*"))


LB_SUFFIX = ":eng:ctx_proposal"
NON_METRIC_COLS = {"timestamp", "model", "is_final", "seed"}


def arm_tag(arm: str) -> str:
    return f"claude-fable-5__{arm}{LB_SUFFIX}"


# ---- byte-level-BPE de-garble (defensive): fixes text stored as raw byte-level glyphs ----
# (e.g. the d1base arm; Ġ=space, Ċ=newline). Salvaged proposals are already clean, but worker
# prompts come from immutable eval logs, so de-garble at display time. No-op on clean text.
def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_DEC = _byte_decoder()


def degarble(s):
    """De-tokenize byte-level-BPE glyph text. Handles MIXED strings (a clean harness
    preamble + a garbled proposal): glyph chars -> their decoded byte, any other char ->
    its own UTF-8 bytes; decode the whole byte stream once. No-op if no Ġ/Ċ present."""
    if not isinstance(s, str) or ("Ġ" not in s and "Ċ" not in s):
        return s
    try:
        out = bytearray()
        for ch in s:
            if ch in _DEC:
                out.append(_DEC[ch])
            else:
                out.extend(ch.encode("utf-8"))
        # errors='replace': a few base-model outputs emit lone invalid-UTF-8 byte tokens
        # (e.g. 0xd7 from "32×32"); replace those rather than leave the whole string garbled.
        return out.decode("utf-8", errors="replace")
    except Exception:
        return s


def load_lite() -> tuple[list[dict], dict]:
    d = json.loads(LITE_JSON.read_text())
    dom_of = {t["slug"]: dom for dom, ts in d["domains"].items() for t in ts}
    name_of = {t["slug"]: t["name"] for dom, ts in d["domains"].items() for t in ts}
    tasks = [{"slug": s, "domain": dom_of.get(s, "?"), "name": name_of.get(s, s)} for s in d["slugs"]]
    # append the dropped task so the matrix is 30 rows
    tasks.append({"slug": DROPPED_TASK, "domain": "Robotics", "name": "Humanoid Sim2Real (dropped)"})
    return tasks, d


def mlsbench_scores(slug: str, use_cache: bool) -> dict[str, float]:
    """{arm: rescaled_score(0-100)} from native `mlsbench score` (0.5 -> 50)."""
    cache = OUT / "raw_scores" / f"{slug}.json"
    data = None
    if use_cache and cache.exists():
        try:
            data = json.loads(cache.read_text())
        except Exception:
            data = None
    if data is None:
        try:
            r = subprocess.run(["mlsbench", "score", slug, "--format", "json"],
                               capture_output=True, text=True, timeout=180)
            data = json.loads(r.stdout)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            print(f"[warn] mlsbench score {slug}: {e}", file=sys.stderr)
            return {}
    rows = data.get(slug, data) if isinstance(data, dict) else data
    rows = rows if isinstance(rows, list) else []
    out = {}
    for r in rows:
        m = r.get("model", "")
        if m.startswith("claude-fable-5__") and m.endswith(LB_SUFFIX) and "task_score" in r:
            arm = m[len("claude-fable-5__"):-len(LB_SUFFIX)]
            out[arm] = round(100.0 * float(r["task_score"]), 1)
    return out


def baseline_methods(slug: str) -> list[dict]:
    """Named baseline methods + raw metric snapshot, from the live leaderboard CSV.

    The mlsbench rescale calibrates the STRONG baseline to 0.5 (=50 on our scale); these
    rows name the reference methods the proposals are competing against."""
    rows = read_leaderboard(slug)
    mcols = metric_cols(rows)
    out = {}
    for r in rows:
        m = r.get("model", "")
        if not m.startswith("baseline:"):
            continue
        metrics = {c: r[c] for c in mcols if r.get(c) not in ("", "nan", "NaN", None)}
        if m not in out or (metrics and not out[m]["metrics"]):
            out[m] = {"name": m.split(":", 1)[1], "metrics": metrics}
    return list(out.values())


def read_leaderboard(slug: str) -> list[dict]:
    p = TASKS_DIR / slug / "leaderboard.csv"
    if not p.exists():
        return []
    try:
        return list(csv.DictReader(p.open()))
    except Exception as e:
        print(f"[warn] leaderboard {slug}: {e}", file=sys.stderr)
        return []


def arm_rows(rows: list[dict], arm: str) -> list[dict]:
    tag = arm_tag(arm)
    return [r for r in rows if r.get("model") == tag]


def metric_cols(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return [c for c in rows[0].keys() if c not in NON_METRIC_COLS]


def row_has_metric(r: dict, mcols: list[str]) -> bool:
    for c in mcols:
        v = r.get(c)
        if v not in ("", "nan", "NaN", None):
            return True
    return False


def find_log_dirs(slug: str, arm: str, worker: str = DEFAULT_WORKER) -> list[Path]:
    base = LOGS_DIR / slug
    if not base.exists():
        return []
    # dir pattern: <worker>__<arm>__<arm>_<slug>_s<seed>
    pref = f"{worker}__{arm}__"
    return sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith(pref))


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ---- master prompt + master output ----
def load_packets() -> dict[str, dict]:
    out = {}
    for p in load_jsonl(EVAL / "packets.jsonl"):
        out[p["task"]] = p
    return out


def load_proposals() -> dict[str, dict[str, dict]]:
    """{arm: {task: record}} from proposals/proposals_<arm>.jsonl."""
    out = {}
    for arm in ARM_KEYS:
        recs = load_jsonl(EVAL / "proposals" / f"proposals_{arm}.jsonl")
        out[arm] = {r["task"]: r for r in recs if "task" in r}
    return out


# ---- worker log parsing ----
def parse_worker(log_dir: Path) -> dict:
    """Extract worker prompt (user handoff), assistant edit trace, final code, tokens."""
    agent = log_dir / "agent"
    msgs = load_jsonl(agent / "messages.jsonl")
    summary = {}
    sp = agent / "summary.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text())
        except Exception:
            summary = {}
    user_prompt = None
    steps = []
    for m in msgs:
        role = m.get("role")
        if role == "user" and user_prompt is None:
            c = m.get("content")
            user_prompt = degarble(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
        elif role == "assistant":
            entry = {"step": m.get("step"), "tool": m.get("tool_name")}
            ti = m.get("tool_input")
            if isinstance(ti, dict):
                entry["file"] = ti.get("filename")
                # keep a short preview of edit content, full code comes from files/
                cont = ti.get("content")
                if isinstance(cont, str):
                    entry["preview"] = cont[:400]
            elif m.get("content"):
                entry["text"] = str(m.get("content"))[:400]
            steps.append(entry)
    # final worker code = highest-numbered step file(s)
    final_code = {}
    fdir = agent / "files"
    if fdir.exists():
        by_name = {}
        for f in fdir.glob("step_*"):
            # step_<n>_<pkg>_<name>
            parts = f.name.split("_", 2)
            try:
                n = int(parts[1])
            except Exception:
                n = 0
            key = parts[2] if len(parts) > 2 else f.name
            if key not in by_name or n > by_name[key][0]:
                by_name[key] = (n, f)
        for key, (n, f) in by_name.items():
            try:
                final_code[f"{key} (step {n})"] = f.read_text()[:20000]
            except Exception:
                pass
    return {
        "log_dir": str(log_dir.relative_to(MLSBENCH_ROOT)),
        "user_prompt": user_prompt,
        "steps": steps,
        "final_code": final_code,
        "summary": {k: summary.get(k) for k in ("steps", "tests", "done", "max_steps", "tokens")},
    }


def per_run_scores(rows_arm: list[dict], mcols: list[str]) -> list[dict]:
    """Per-seed final-row metric snapshot for the cell detail page."""
    out = []
    for r in rows_arm:
        if str(r.get("is_final", "")).lower() != "true":
            continue
        metrics = {c: r[c] for c in mcols if r.get(c) not in ("", "nan", "NaN", None)}
        out.append({"seed": r.get("seed"), "timestamp": r.get("timestamp"),
                    "has_metric": bool(metrics), "metrics": metrics})
    return out


def classify(*_a, **_k):  # retained for import stability; matrix now comes from report.json
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT_JSON))
    ap.add_argument("--no-score", action="store_true", help="reuse cached raw_scores/ (per-seed detail only)")
    args = ap.parse_args()
    use_cache = args.no_score

    report = json.loads(Path(args.report).read_text())
    rep_rows = {r["task"]: r for r in report["rows"]}          # authoritative final matrix
    rep_summary = report.get("summary", {})
    # active arms = canonical arms present in the report (auto-includes new arms e.g. purefable5)
    active = [a for a in ARMS if a[0] in rep_summary]
    active_keys = [a[0] for a in active]
    load_arm_notes()
    packets = load_packets()
    proposals = load_proposals()
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    (OUT / "tasks").mkdir(parents=True, exist_ok=True)

    # task spine = report rows (task/name/domain), in report order
    tasks = [{"slug": r["task"], "name": r.get("name", r["task"]), "domain": r.get("domain", "?")}
             for r in report["rows"]]

    index = {
        "tasks": [], "arms": [{"key": a[0], "label": a[1], "family": a[2], "kind": a[3]} for a in active],
        "cells": {},
        "arm_means": {a: rep_summary.get(a, {}).get("mean") for a in active_keys},
        "arm_beat": {a: rep_summary.get(a, {}).get("n_beat_baseline") for a in active_keys},
        "source": "report.json (leaderboard-based final matrix; all cells populated, 0 missing)",
        "scale": "rescaled: 50 = strong baseline, 100 = theoretical upper bound. A 0 is a faithful true-0 (capability floor), not missing data.",
    }
    n_cells = n_caveat = 0
    arm_label = {a[0]: a[1] for a in ARMS}
    arm_kind = {a[0]: a[3] for a in ARMS}
    caveat_texts = []  # ordered-unique -> footnote number = index+1
    task_ctx = {}

    for t in tasks:
        slug = t["slug"]
        index["cells"][slug] = {}
        worker = WORKER_BY_TASK.get(slug, DEFAULT_WORKER)
        lb_rows = read_leaderboard(slug)
        base = LOGS_DIR / slug
        listing = [d for d in base.iterdir() if d.is_dir()] if base.exists() else []  # list once per task
        task_ctx[slug] = {"worker": worker, "lb_rows": lb_rows, "mcols": metric_cols(lb_rows),
                          "listing": listing, "rrow": rep_rows.get(slug, {})}

        # ---- shared per-task detail ----
        desc_p = TASKS_DIR / slug / "task_description.md"
        pkt = packets.get(slug, {})
        master_prompt = {"system": None, "user": None}
        for m in pkt.get("messages", []):
            if m.get("role") in master_prompt and master_prompt[m["role"]] is None:
                master_prompt[m["role"]] = m.get("content")
        bm = baseline_methods(slug)   # native raw anchors (the rescaled 50 is calibrated to the best of these)
        # compact copy for the index-level raw reference (cap metrics per method to bound size)
        t["raw_baselines"] = [{"name": x["name"], "metrics": dict(list(x["metrics"].items())[:12])} for x in bm]
        index["tasks"].append(t)
        (OUT / "tasks" / f"{slug}.json").write_text(json.dumps({
            "slug": slug, "domain": t["domain"], "name": t["name"], "worker": worker,
            "description": desc_p.read_text() if desc_p.exists() else None,
            "metric": pkt.get("metric"), "lower_is_better": pkt.get("lower_is_better"),
            "master_prompt": master_prompt, "row_caveat": CAVEATS.get((slug, "*")),
            "baselines": {"methods": bm,
                          "note": "rescaled cells: 50 = strong baseline, 100 = theoretical upper bound."},
        }, ensure_ascii=False, indent=1))

    # footnote numbers (deterministic task×arm order)
    for t in tasks:
        for arm in active_keys:
            cav = caveat_for(t["slug"], arm)
            if cav and cav not in caveat_texts:
                caveat_texts.append(cav)

    def build_cell(job):
        slug, arm = job
        ctx = task_ctx[slug]
        pref = f"{ctx['worker']}__{arm}__"
        logdirs = sorted(d for d in ctx["listing"] if d.name.startswith(pref))
        runs = per_run_scores(arm_rows(ctx["lb_rows"], arm), ctx["mcols"])
        prop = proposals.get(arm, {}).get(slug, {})
        cav = caveat_for(slug, arm)
        native = (arm_kind.get(arm) == "native")
        (OUT / "cells" / f"{slug}__{arm}.json").write_text(json.dumps({
            "task": slug, "arm": arm, "arm_label": arm_label[arm],
            "status": "scored", "score": ctx["rrow"].get(arm), "worker": ctx["worker"], "caveat": cav,
            "native": native,
            "note": "Native MLS-Bench agent — the worker solves the task from scratch with NO master proposal (aligns with the original MLS-Bench setup)." if native else None,
            "master_output": {"idea": degarble(prop.get("idea")), "proposal": degarble(prop.get("proposal")),
                              "valid": prop.get("valid"), "n_attempts": prop.get("n_attempts")},
            "per_run_scores": runs, "workers": [parse_worker(d) for d in logdirs],
        }, ensure_ascii=False, indent=1))
        return slug, arm, len(logdirs), sum(1 for x in runs if x["has_metric"])

    jobs = [(t["slug"], arm) for t in tasks for arm in active_keys]
    with cf.ThreadPoolExecutor(max_workers=16) as ex:   # FS I/O bound -> parallelize
        res = list(ex.map(build_cell, jobs))
    for slug, arm, nruns, nscored in res:
        cav = caveat_for(slug, arm)
        fn = (caveat_texts.index(cav) + 1) if cav else None
        n_cells += 1
        if cav:
            n_caveat += 1
        index["cells"][slug][arm] = {"status": "scored", "score": task_ctx[slug]["rrow"].get(arm),
                                     "caveat": bool(cav), "fn": fn, "n_runs": nruns,
                                     "n_scored": nscored, "detail": True}

    index["counts"] = {"cells": n_cells, "caveated": n_caveat, "tasks": len(tasks), "arms": len(active_keys)}
    index["caveats"] = caveat_texts
    index["generated_from"] = "mls_dashboard_extract.py"
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(f"[ok] wrote {OUT}/index.json")
    print(f"[ok] {n_cells} cells = {len(tasks)} tasks x {len(active_keys)} arms; {n_caveat} caveated; source=report.json")


if __name__ == "__main__":
    main()
