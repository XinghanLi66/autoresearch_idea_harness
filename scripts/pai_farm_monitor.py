#!/usr/bin/env python3
"""PAI DLC idle/waste monitor for the L20Z farm (workspace 262162, quota quota1shcr2h7uae).

Read-only. Flags three classes of compute waste and prints a report + a copy-pasteable stop
list for a human to confirm. Does NOT auto-kill (shared workloads; auto-mode classifier blocks
unattended mass-stops anyway). Requested by cc001 after a vLLM serve job pinned 4 GPUs at 0%
util for >42h on a GPU-quota-bound farm.

Checks (all derived from `list-jobs`, plus a cheap per-job log tail for idle):
  1. idle-GPU     — Running job whose work has stopped:
                      * serve jobs: vLLM tail shows sustained "0.0 tokens/s" / "Running: 0 reqs"
                      * any job:    log tail unchanged for >= STALL_MIN (progress stalled proxy;
                                    get-job-metrics PodMetrics come back empty, so log-diff is the
                                    reliable cheap signal — see cc001 plumbing notes).
  2. over-runtime — Running longer than its class ceiling (serve/cheap 6h, heavy 15h).
  3. duplicate    — same DisplayName in >1 non-terminal job (re-launched drivers).

Plumbing gotchas handled (all per cc001, verified this session):
  * cred server flaky (ECS-metadata 404 fallback) -> every pai call retried 4x with backoff.
  * PageSize=500 errors -> page with PageSize=100 until a short page.
  * StatusIn filter broken -> filter on j['Status'] in Python.
  * get-job-metrics returns empty PodMetrics -> idle relies on the log-tail signal, not metrics.

Usage:
  export ALIBABA_CLOUD_CREDENTIALS_URI=http://localhost:7002/api/v1/credentials/0
  python scripts/pai_farm_monitor.py                 # one pass, print report
  python scripts/pai_farm_monitor.py --quiet         # only print if there are hits
  python scripts/pai_farm_monitor.py --no-checkin    # don't append to checkins.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
PAI_MANAGE = "/root/.claude/skills/pai/scripts/pai_manage.py"
CRED_URI = "http://localhost:7002/api/v1/credentials/0"

# Farm identity (mirrors scripts/mls_cluster_dispatch.py).
WORKSPACE_ID = "262162"
QUOTA_ID = "quota1shcr2h7uae"

# Terminal statuses; everything else counts as non-terminal (Running/Queuing/Dequeued/
# Creating/Restarting/EnvPreparing/...).
TERMINAL = {"Succeeded", "Failed", "Stopped", "Untracked"}

# Heavy MLS tasks (mirrors HEAVY_TASKS in mls_cluster_dispatch.py) — matched as substrings of
# DisplayName since names are `mls-<arm>-<task-slug>-s<seed>`.
HEAVY_TASKS = {
    "llm-pretrain-optimizer", "robomimic-bc-loss", "jepa-planning",
    "robo-diffusion-policy", "llm-rl-importance-sampling", "robo-diffusion-guidance",
}

# Runtime ceilings (hours) by job class.
CEIL_SERVE_H = 6.0    # serve idle since last request
CEIL_HEAVY_H = 15.0   # heavy eval tasks (est. 4-12h)
CEIL_CHEAP_H = 6.0    # everything else

MIN_AGE_H_FOR_IDLE = 0.5   # don't flag idle before a job has had 30 min to warm up
IDLE_PROBE_MIN_AGE_H = 3.0  # only spend a log fetch on non-serve jobs once they're this old
                            # (fresh eval jobs aren't "waste" yet; keeps the pass fast)
STALL_MIN = 30.0           # log tail unchanged this long => progress-stalled
LOG_TAIL_LINES = 40        # how many recent log lines to inspect
PAI_TIMEOUT_S = 45         # per pai_manage subprocess; a hung call must not stall the pass

STATE_PATH = HARNESS_ROOT / "runs" / "pai_farm_monitor" / "state.json"
# Canonical cross-agent channel is the PARENT-level agent-memory/checkins.md (the harness subdir
# has no checkins.md); fall back to the harness copy if the layout ever changes.
_CHECKIN_CANDIDATES = [
    HARNESS_ROOT.parent / "agent-memory" / "checkins.md",
    HARNESS_ROOT / "agent-memory" / "checkins.md",
]
CHECKINS = next((p for p in _CHECKIN_CANDIDATES if p.exists()), _CHECKIN_CANDIDATES[0])


# ---------------------------------------------------------------------------
# PAI plumbing (retry-wrapped for the flaky cred server)
# ---------------------------------------------------------------------------

def _pai(args: list[str], *, retries: int = 4, parse: bool = True):
    env = dict(os.environ)
    env["ALIBABA_CLOUD_CREDENTIALS_URI"] = CRED_URI
    env.pop("PAI_WORKSPACE_ID", None)  # DSW leaks its own workspace; must not override --set
    last_err = ""
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                [sys.executable, PAI_MANAGE, *args],
                capture_output=True, text=True, env=env, timeout=PAI_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            last_err = f"timeout after {PAI_TIMEOUT_S}s"
            time.sleep(1.5 * (attempt + 1))
            continue
        out = proc.stdout.strip()
        if out:
            if not parse:
                return proc
            try:
                return json.loads(out)
            except json.JSONDecodeError:
                last_err = f"non-JSON stdout: {out[:200]}"
        else:
            last_err = proc.stderr.strip()[-200:]
        time.sleep(1.5 * (attempt + 1))  # backoff over the cred flake
    raise RuntimeError(f"pai_manage {args[0]} failed after {retries} tries: {last_err}")


def list_all_jobs() -> list[dict]:
    """All jobs in the workspace, paged at PageSize=100 (500 errors)."""
    jobs: list[dict] = []
    page = 1
    while True:
        body = _pai(["list-jobs", "--set", f"WorkspaceId={WORKSPACE_ID}",
                     "--set", "PageSize=100", "--set", f"PageNumber={page}",
                     "--compact"]).get("body", {})
        batch = body.get("Jobs", []) or []
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        if page > 50:  # hard stop; 5000 jobs is far more than this farm holds
            break
    return jobs


def pod_log_tail(job_id: str, lines: int = LOG_TAIL_LINES) -> list[str]:
    """Recent log lines from the master pod (pod-id convention <jobid>-master-0)."""
    try:
        body = _pai(["get-pod-logs", "--job-id", job_id,
                     "--pod-id", f"{job_id}-master-0", "--compact"], retries=3).get("body", {})
    except RuntimeError:
        return []
    logs = body.get("Logs", []) or []
    return [ln for ln in logs if ln.strip()][-lines:]


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def job_class(name: str) -> str:
    n = (name or "").lower()
    if "serve" in n:
        return "serve"
    if any(h in n for h in HEAVY_TASKS):
        return "heavy"
    return "cheap"


def ceiling_h(cls: str) -> float:
    return {"serve": CEIL_SERVE_H, "heavy": CEIL_HEAVY_H}.get(cls, CEIL_CHEAP_H)


def serve_idle(tail: list[str]) -> bool:
    """vLLM serve is idle if its recent throughput lines are all 0.0 tokens/s with 0 running reqs
    and no nonzero generation throughput appears in the tail."""
    thru = [ln for ln in tail if "tokens/s" in ln.lower()]
    if not thru:
        return False
    # any nonzero generation throughput in the tail => actively serving, not idle
    for ln in thru:
        low = ln.lower()
        if "generation throughput" in low and "0.0 tokens/s" not in low:
            return False
    last = thru[-1].lower()
    return "0.0 tokens/s" in last and ("running: 0" in last or "running:0" in last)


def tail_sig(tail: list[str]) -> str:
    return hashlib.sha1(("\n".join(tail[-8:])).encode("utf-8", "replace")).hexdigest()[:16]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def age_hours(job: dict) -> float:
    dur = job.get("Duration")
    if isinstance(dur, (int, float)) and dur > 0:
        return dur / 3600.0
    gr = job.get("GmtRunningTime") or job.get("GmtCreateTime")
    if gr:
        try:
            t = datetime.fromisoformat(gr.replace("Z", "+00:00"))
            return (now_utc() - t).total_seconds() / 3600.0
        except ValueError:
            pass
    return 0.0


# ---------------------------------------------------------------------------
# State (for the log-progress-stall diff across runs)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def run_once() -> tuple[list[dict], list[dict]]:
    """Returns (flags, running) where flags is a list of {name,id,status,dur_h,reason,detail}."""
    jobs = list_all_jobs()
    nonterminal = [j for j in jobs if j.get("Status") not in TERMINAL]
    running = [j for j in nonterminal if j.get("Status") == "Running"]

    state = load_state()
    new_state: dict = {}
    ts = now_utc().isoformat()
    flags: list[dict] = []

    # --- Check 3: duplicate DisplayName among non-terminal jobs ---
    by_name: dict[str, list[dict]] = {}
    for j in nonterminal:
        by_name.setdefault(j.get("DisplayName", "?"), []).append(j)
    for name, group in by_name.items():
        if len(group) > 1:
            ids = ", ".join(j["JobId"] for j in group)
            # flag every duplicate but the oldest (keep the longest-running instance)
            group_sorted = sorted(group, key=age_hours, reverse=True)
            for j in group_sorted[1:]:
                flags.append({
                    "name": name, "id": j["JobId"], "status": j.get("Status"),
                    "dur_h": round(age_hours(j), 1), "gpu": j.get("RequestGPU"),
                    "reason": "duplicate", "detail": f"{len(group)}x non-terminal ({ids})",
                })

    # --- Checks 1 & 2 on Running jobs ---
    for j in running:
        jid = j["JobId"]
        name = j.get("DisplayName", "?")
        cls = job_class(name)
        dur_h = age_hours(j)
        gpu = j.get("RequestGPU")
        reasons: list[str] = []
        detail_bits: list[str] = []

        # Check 2 first (free, from Duration): over-runtime by class ceiling.
        ceil = ceiling_h(cls)
        over_runtime = dur_h > ceil
        if over_runtime:
            reasons.append("over-runtime")
            detail_bits.append(f"{dur_h:.1f}h > {ceil:.0f}h ({cls})")

        # Log tail (single SDK fetch) only when it can ADD signal: a job that's already
        # over-runtime is flagged regardless, so skip its fetch. Probe healthy-age jobs to catch
        # idle/stall EARLY (serve at any age >=30min; others once >= IDLE_PROBE_MIN_AGE_H).
        probe = (not over_runtime and dur_h >= MIN_AGE_H_FOR_IDLE
                 and (cls == "serve" or dur_h >= IDLE_PROBE_MIN_AGE_H))
        tail = pod_log_tail(jid) if probe else []
        sig = tail_sig(tail) if tail else ""

        # log-progress stall tracking (persisted across runs)
        prev = state.get(jid)
        if sig:
            if prev and prev.get("sig") == sig:
                since = prev.get("since", ts)
                new_state[jid] = {"sig": sig, "since": since}
                try:
                    stalled_min = (now_utc() - datetime.fromisoformat(since)).total_seconds() / 60.0
                except ValueError:
                    stalled_min = 0.0
                if stalled_min >= STALL_MIN:
                    reasons.append("idle:log-stalled")
                    detail_bits.append(f"log unchanged {stalled_min:.0f}min")
            else:
                new_state[jid] = {"sig": sig, "since": ts}
        elif prev:
            new_state[jid] = prev  # preserve stall clock for over-runtime jobs we skipped probing

        # Check 1: serve idle (strong signal, when probed)
        if cls == "serve" and serve_idle(tail):
            reasons.append("idle:serve-0-throughput")
            detail_bits.append("vLLM 0.0 tokens/s, 0 reqs")

        if reasons:
            flags.append({
                "name": name, "id": jid, "status": j.get("Status"),
                "dur_h": round(dur_h, 1), "gpu": gpu,
                "reason": "+".join(reasons), "detail": "; ".join(detail_bits),
            })

    save_state(new_state)
    return flags, running


REPORTED_PATH = HARNESS_ROOT / "runs" / "pai_farm_monitor" / "reported.json"


def load_reported() -> set[str]:
    if REPORTED_PATH.exists():
        try:
            return set(json.loads(REPORTED_PATH.read_text()))
        except json.JSONDecodeError:
            return set()
    return set()


def save_reported(ids: list[str]) -> None:
    REPORTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTED_PATH.write_text(json.dumps(sorted(ids)))


def render(flags: list[dict], running: list[dict]) -> str:
    lines = []
    ts = now_utc().strftime("%Y-%m-%d %H:%M UTC")
    tot_gpu = sum(int(j.get("RequestGPU") or 0) for j in running)
    lines.append(f"[pai-farm-monitor {ts}] ws {WORKSPACE_ID} · running={len(running)} "
                 f"(~{tot_gpu} GPU) · flagged={len(flags)}")
    if not flags:
        lines.append("  no waste flagged (idle / over-runtime / duplicate all clear)")
        return "\n".join(lines)
    wasted = sum(int(f.get("gpu") or 0) for f in flags)
    lines.append(f"  ⚠ {len(flags)} job(s) flagged, ~{wasted} GPU potentially reclaimable:")
    for f in sorted(flags, key=lambda x: x.get("dur_h", 0), reverse=True):
        lines.append(f"    • {f['name'][:52]:52s} {f['id']}  "
                     f"{f['status']:8s} {f['dur_h']:>6.1f}h  gpu={f.get('gpu')}  "
                     f"[{f['reason']}]  {f['detail']}")
    lines.append("  --- stop list (human-confirm; run these to reclaim GPUs) ---")
    for f in flags:
        lines.append(f"    python {PAI_MANAGE} stop-job --job-id {f['id']}   "
                     f"# {f['name'][:48]} [{f['reason']}]")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="PAI DLC L20Z farm idle/waste monitor (read-only).")
    ap.add_argument("--quiet", action="store_true", help="only print/append when there are hits")
    ap.add_argument("--no-checkin", action="store_true", help="do not append to agent-memory/checkins.md")
    ap.add_argument("--always-checkin", action="store_true",
                    help="append even if the flagged set is unchanged since last run")
    args = ap.parse_args()

    prev_reported = load_reported()
    try:
        flags, running = run_once()
    except Exception as e:  # never crash a cron loop
        print(f"[pai-farm-monitor] ERROR: {type(e).__name__}: {e}", flush=True)
        return 1

    cur_ids = {f["id"] for f in flags}
    save_reported(list(cur_ids))
    new_ids = cur_ids - prev_reported

    report = render(flags, running)
    if flags or not args.quiet:
        print(report, flush=True)
    if new_ids:
        print(f"  ({len(new_ids)} NEW since last run: "
              f"{', '.join(sorted(new_ids))})", flush=True)

    # Append to checkins only when the flagged set changed (new hits) — avoids spamming the
    # channel every cycle with the same standing backlog.
    changed = bool(cur_ids != prev_reported)
    if flags and not args.no_checkin and CHECKINS.exists() and (changed or args.always_checkin):
        with CHECKINS.open("a") as fh:
            tag = "NEW hits" if new_ids else "flagged set changed"
            fh.write(f"\n- [pai-farm-monitor {now_utc().strftime('%Y-%m-%d %H:%M UTC')}] "
                     f"@cc001 @cc000 {len(flags)} farm job(s) flagged ({tag}; read-only, confirm before stop). "
                     f"Full report + stop list: `python scripts/pai_farm_monitor.py`:\n")
            for f in flags:
                star = " *NEW*" if f["id"] in new_ids else ""
                fh.write(f"    - {f['name']} ({f['id']}) {f['dur_h']}h gpu={f.get('gpu')} "
                         f"[{f['reason']}] {f['detail']}{star}\n")
    # exit 2 signals "hits found" (cron/loop can branch on it); 0 = clean
    return 2 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
