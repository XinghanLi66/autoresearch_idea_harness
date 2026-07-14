#!/usr/bin/env python3
"""Drive a prepared QS chunk-staging run: submit chunk trials in concurrent waves,
poll to completion, retry failures, then run the finalize trial.

Chunk trials write independent part_NNNN.b64 files on /mnt/3fs (order-independent),
so waves are safe; finalize validates all parts + sha256 before assembling the file.

Usage:
  CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK=1 python scripts/run_qs_chunk_stage.py \
      --run-dir runs/qs_data_stage_chunks/<run_id> [--wave 8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd: list[str] | str, *, shell=False, timeout=300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)


def submit(step: dict) -> dict:
    """Run the step's .submit.sh; parse trial_id from the tee'd submission json or stdout."""
    p = sh(["bash", step["submit"]], timeout=600)
    out = p.stdout + p.stderr
    trial = None
    sub_json = Path(step["submit"]).with_suffix("").with_suffix("")  # step_xxx.submit.sh -> step_xxx
    m = re.search(r'"trialId"\s*:\s*(\d+)', out) or re.search(r'"trial_id"\s*:\s*(\d+)', out)
    if m:
        trial = int(m.group(1))
    else:  # fall back to the submission.json the script tees
        cand = sorted(Path(step["submit"]).parent.glob(Path(step["submit"]).name.replace(".submit.sh", "*.submission.json")))
        for c in cand:
            try:
                d = json.loads(c.read_text())
                trial = int(d.get("trialId") or d.get("trial_id"))
                break
            except Exception:
                continue
    return {"ok": p.returncode == 0 and trial is not None, "trial_id": trial,
            "returncode": p.returncode, "tail": out[-300:]}


def trial_state(trial_id: int) -> str:
    p = sh(["qs", "training", "get", str(trial_id), "-q"], timeout=120)
    m = re.search(r"状态\s+(\S+)", p.stdout)
    return m.group(1) if m else "Unknown"


def wait_all(trials: dict[int, str], poll_s: int = 20, timeout_s: int = 3600) -> dict[int, str]:
    """Poll until every trial reaches a terminal state."""
    t0 = time.time()
    pending = set(trials)
    while pending and time.time() - t0 < timeout_s:
        for t in list(pending):
            st = trial_state(t)
            trials[t] = st
            if st in ("Complete", "Failed", "Stopped", "Error", "Killed"):
                pending.discard(t)
        if pending:
            print(f"[stage] waiting on {len(pending)} trials … states={dict(list(trials.items())[:3])}", flush=True)
            time.sleep(poll_s)
    return trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--wave", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()

    if os.environ.get("CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK") != "1":
        raise SystemExit("set CONFIRM_SUBMIT_V3_QS_DATA_STAGE_CHUNK=1 to submit")

    plan = json.loads((args.run_dir / "run_plan.json").read_text())
    steps = plan["steps"]
    chunks = [s for s in steps if s["kind"] == "chunk"]
    finals = [s for s in steps if s["kind"] == "finalize"]
    log_path = args.run_dir / "chunk_submissions.jsonl"
    done_keys = set()
    if log_path.exists():  # resume: skip chunks already Complete
        for line in log_path.open():
            try:
                r = json.loads(line)
                if r.get("final_state") == "Complete":
                    done_keys.add(r["submit_path"])
            except Exception:
                pass
    todo = [s for s in chunks if s["submit"] not in done_keys]
    print(f"[stage] {len(chunks)} chunks total, {len(done_keys)} already done, {len(todo)} to submit", flush=True)

    def run_wave(batch: list[dict], attempt: int) -> list[dict]:
        failed = []
        subs = {}
        for s in batch:
            r = submit(s)
            print(f"[stage] submit {Path(s['submit']).name}: trial={r['trial_id']} ok={r['ok']}", flush=True)
            if r["ok"]:
                subs[r["trial_id"]] = ("Submitted", s)
            else:
                failed.append(s)
        states = wait_all({t: "Submitted" for t in subs})
        with log_path.open("a") as f:
            for t, st in states.items():
                s = subs[t][1]
                f.write(json.dumps({"submit_path": s["submit"], "trial_id": t,
                                    "final_state": st, "attempt": attempt, "kind": "chunk"}) + "\n")
                if st != "Complete":
                    failed.append(s)
        return failed

    remaining = todo
    for attempt in range(args.max_retries + 1):
        if not remaining:
            break
        nxt = []
        for i in range(0, len(remaining), args.wave):
            nxt += run_wave(remaining[i:i + args.wave], attempt)
        remaining = nxt
        if remaining:
            print(f"[stage] retrying {len(remaining)} failed chunks (attempt {attempt+1})", flush=True)
    if remaining:
        raise SystemExit(f"[stage] {len(remaining)} chunks failed after retries; not finalizing")

    # finalize
    for s in finals:
        r = submit(s)
        print(f"[stage] finalize submit: trial={r['trial_id']} ok={r['ok']}", flush=True)
        if not r["ok"]:
            raise SystemExit("finalize submission failed")
        states = wait_all({r["trial_id"]: "Submitted"})
        st = states[r["trial_id"]]
        with log_path.open("a") as f:
            f.write(json.dumps({"submit_path": s["submit"], "trial_id": r["trial_id"],
                                "final_state": st, "kind": "finalize"}) + "\n")
        if st != "Complete":
            raise SystemExit(f"finalize trial ended {st} — check qs logs {r['trial_id']}")
    print(json.dumps({"status": "ok", "chunks": len(chunks), "finalized": True}), flush=True)


if __name__ == "__main__":
    main()
