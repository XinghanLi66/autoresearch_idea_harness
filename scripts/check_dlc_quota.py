#!/usr/bin/env python3
"""Fail-closed PAI-DLC quota availability checker.

This replaces ad-hoc quota polling that treated API failures as empty queues.
If a workspace cannot be queried or the response is malformed, the pool is
marked UNKNOWN and the command exits non-zero unless --allow-partial is set.

Important: PAI list-jobs may only expose jobs visible to the current credential,
not the global quota scheduler queue. By default this script reports visible
jobs only and refuses to select a "best" pool. Use --accept-visible-jobs-only
only when an external quota-level source is unavailable and the risk is explicit.

The installed DLC SDK does not expose a quota-global queue API. Its
FromAllWorkspaces flag is documented as current-user job search when combined
with ShowOwn=true, so it is not an authoritative quota-depth source either.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "pai-dlc.ap-southeast-1.aliyuncs.com"
DEFAULT_CREDENTIAL_URI = "http://localhost:7002/api/v1/credentials/0"
PAI = Path("/root/.claude/skills/pai/scripts/pai_manage.py")

POOLS = [
    {"workspace": "224239", "quota": "quota1ecrg95m4n9", "label": "M0-Dots_V3"},
    {"workspace": "137902", "quota": "quotadbz1mvpy1v5", "label": "ws137902"},
    {"workspace": "238626", "quota": "quota1d8xmvdw5tb", "label": "ws238626"},
    {"workspace": "262162", "quota": "quota1shcr2h7uae", "label": "ws262162"},
]

WAITING_STATUSES = {"Creating", "Queuing", "EnvPreparing"}
RUNNING_STATUSES = {"Running"}
ACTIVE_STATUSES = WAITING_STATUSES | RUNNING_STATUSES | {"Stopping"}


class QueryError(RuntimeError):
    pass


def _load_json_response(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise QueryError("empty stdout from pai_manage.py")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise QueryError(f"stdout is not JSON: {text[:300]}") from exc
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise QueryError(f"expected JSON object, got {type(data).__name__}")
    return data


def _body(data: dict[str, Any]) -> dict[str, Any]:
    body = data.get("body", data)
    if not isinstance(body, dict):
        raise QueryError("response body is not an object")
    return body


def _jobs_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = body.get("Jobs", [])
    if jobs is None:
        return []
    if not isinstance(jobs, list):
        raise QueryError("body.Jobs is not a list")
    return [job for job in jobs if isinstance(job, dict)]


def _total_count(body: dict[str, Any], fallback: int) -> int:
    for key in ("TotalCount", "Total", "TotalNumber"):
        value = body.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


def list_workspace_jobs(
    workspace_id: str,
    *,
    endpoint: str,
    credential_uri: str,
    page_size: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    if not PAI.is_file():
        raise QueryError(f"PAI helper not found: {PAI}")
    env = os.environ.copy()
    env["ALIBABA_CLOUD_CREDENTIALS_URI"] = credential_uri
    jobs: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        request = {"WorkspaceId": workspace_id, "PageNumber": page, "PageSize": page_size}
        cmd = [
            sys.executable,
            str(PAI),
            "list-jobs",
            "--endpoint",
            endpoint,
            "--request-json",
            json.dumps(request, separators=(",", ":")),
            "--compact",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise QueryError(f"list-jobs failed for workspace {workspace_id}: {msg[:1000]}")
        body = _body(_load_json_response(result.stdout))
        page_jobs = _jobs_from_body(body)
        jobs.extend(page_jobs)
        total = _total_count(body, fallback=len(jobs))
        if len(jobs) >= total or len(page_jobs) < page_size:
            return jobs
    raise QueryError(f"list-jobs exceeded max_pages={max_pages} for workspace {workspace_id}")


def summarize_pool(pool: dict[str, str], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    quota_jobs = [job for job in jobs if job.get("ResourceId") == pool["quota"]]
    active_jobs = [job for job in quota_jobs if job.get("Status") in ACTIVE_STATUSES]
    status_counts = Counter(str(job.get("Status") or "UNKNOWN") for job in quota_jobs)
    waiting_jobs = [job for job in active_jobs if job.get("Status") in WAITING_STATUSES]
    running_jobs = [job for job in active_jobs if job.get("Status") in RUNNING_STATUSES]
    stopping_jobs = [job for job in active_jobs if job.get("Status") == "Stopping"]
    score = -10_000 * len(waiting_jobs) - 100 * len(running_jobs) - len(stopping_jobs)
    return {
        "query_ok": True,
        "workspace": pool["workspace"],
        "quota": pool["quota"],
        "label": pool["label"],
        "workspace_job_count": len(jobs),
        "quota_job_count": len(quota_jobs),
        "waiting": len(waiting_jobs),
        "running": len(running_jobs),
        "stopping": len(stopping_jobs),
        "active": len(active_jobs),
        "score": score,
        "global_queue_source": None,
        "global_waiting": None,
        "global_running": None,
        "global_score": None,
        "status_counts": dict(sorted(status_counts.items())),
        "waiting_jobs": [_job_name(job) for job in waiting_jobs],
        "running_jobs": [_job_name(job) for job in running_jobs],
        "active_jobs": [_job_name(job) for job in active_jobs],
    }


def _job_name(job: dict[str, Any]) -> str:
    job_id = str(job.get("JobId") or "?")
    name = str(job.get("DisplayName") or "?")
    status = str(job.get("Status") or "?")
    return f"{job_id}:{name}:{status}"


def summarize_unknown(pool: dict[str, str], error: str) -> dict[str, Any]:
    return {
        "query_ok": False,
        "workspace": pool["workspace"],
        "quota": pool["quota"],
        "label": pool["label"],
        "waiting": None,
        "running": None,
        "stopping": None,
        "active": None,
        "score": -10**12,
        "global_queue_source": None,
        "global_waiting": None,
        "global_running": None,
        "global_score": None,
        "status_counts": {},
        "waiting_jobs": [],
        "running_jobs": [],
        "active_jobs": [],
        "error": error,
    }


def load_mock(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise QueryError("mock file must be an object keyed by workspace id")
    out: dict[str, list[dict[str, Any]]] = {}
    for workspace, jobs in data.items():
        if not isinstance(jobs, list):
            raise QueryError(f"mock workspace {workspace} is not a list")
        out[str(workspace)] = [job for job in jobs if isinstance(job, dict)]
    return out


def _pool_key(workspace: str, quota: str) -> str:
    return f"{workspace}/{quota}"


def _parse_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise QueryError(f"{field} must be an integer, got bool")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        raise QueryError(f"{field} must be a non-negative integer")
    if number < 0:
        raise QueryError(f"{field} must be non-negative")
    return number


def _parse_queue_counts(value: Any, *, source: str) -> dict[str, int]:
    if isinstance(value, dict):
        waiting = _parse_nonnegative_int(value.get("waiting", value.get("queuing", 0)), field=f"{source}.waiting")
        running = _parse_nonnegative_int(value.get("running", 0), field=f"{source}.running")
        return {"waiting": waiting, "running": running}
    if isinstance(value, int):
        return {"waiting": _parse_nonnegative_int(value, field=f"{source}.waiting"), "running": 0}
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,/]", value) if part.strip()]
        if not parts:
            raise QueryError(f"{source} has empty queue counts")
        waiting = _parse_nonnegative_int(parts[0], field=f"{source}.waiting")
        running = _parse_nonnegative_int(parts[1], field=f"{source}.running") if len(parts) > 1 else 0
        return {"waiting": waiting, "running": running}
    raise QueryError(f"{source} must be object, integer, or string")


def load_queue_snapshot(path: Path | None, manual_entries: list[str]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    if path:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            for key, value in data.items():
                snapshot[str(key)] = {
                    **_parse_queue_counts(value, source=str(key)),
                    "source": str(path),
                }
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    raise QueryError("queue snapshot list entries must be objects")
                workspace = str(item.get("workspace") or item.get("workspace_id") or "")
                quota = str(item.get("quota") or item.get("resource_id") or "")
                if not workspace or not quota:
                    raise QueryError("queue snapshot entry missing workspace/quota")
                snapshot[_pool_key(workspace, quota)] = {
                    **_parse_queue_counts(item, source=_pool_key(workspace, quota)),
                    "source": str(path),
                }
        else:
            raise QueryError("queue snapshot must be an object or list")
    for entry in manual_entries:
        if "=" not in entry:
            raise QueryError(f"--manual-queue must be WORKSPACE/QUOTA=WAITING[,RUNNING], got {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        if "/" not in key:
            raise QueryError(f"--manual-queue key must be WORKSPACE/QUOTA, got {key!r}")
        snapshot[key] = {
            **_parse_queue_counts(value, source=key),
            "source": "--manual-queue",
        }
    return snapshot


def apply_global_queue_snapshot(results: list[dict[str, Any]], snapshot: dict[str, dict[str, Any]]) -> set[str]:
    covered: set[str] = set()
    for row in results:
        key = _pool_key(row["workspace"], row["quota"])
        counts = snapshot.get(key)
        if not counts:
            continue
        waiting = int(counts["waiting"])
        running = int(counts["running"])
        row["global_queue_source"] = counts.get("source")
        row["global_waiting"] = waiting
        row["global_running"] = running
        row["global_score"] = -10_000 * waiting - 100 * running - int(row.get("stopping") or 0)
        covered.add(key)
    return covered


def rank_pools(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    mock = load_mock(Path(args.mock_file)) if args.mock_file else None
    results = []
    errors = []
    for pool in POOLS:
        try:
            if mock is not None:
                jobs = mock.get(pool["workspace"], [])
            else:
                jobs = list_workspace_jobs(
                    pool["workspace"],
                    endpoint=args.endpoint,
                    credential_uri=args.credential_uri,
                    page_size=args.page_size,
                    max_pages=args.max_pages,
                )
            results.append(summarize_pool(pool, jobs))
        except Exception as exc:  # noqa: BLE001 - fail-closed with error surface
            errors.append(f"{pool['label']} {pool['workspace']}/{pool['quota']}: {exc}")
            results.append(summarize_unknown(pool, str(exc)))
    snapshot = load_queue_snapshot(Path(args.queue_snapshot) if args.queue_snapshot else None, args.manual_queue or [])
    covered = apply_global_queue_snapshot(results, snapshot)
    expected = {_pool_key(pool["workspace"], pool["quota"]) for pool in POOLS}
    if snapshot and not args.allow_partial_snapshot and covered != expected:
        missing = ", ".join(sorted(expected - covered))
        errors.append(f"queue snapshot does not cover all known pools; missing: {missing}")
    if snapshot and (args.allow_partial_snapshot or covered == expected):
        results.sort(
            key=lambda r: (
                r["query_ok"],
                r.get("global_score") is not None,
                r.get("global_score") if r.get("global_score") is not None else -10**12,
            ),
            reverse=True,
        )
    else:
        results.sort(key=lambda r: (r["query_ok"], r["score"]), reverse=True)
    return results, errors


def print_table(results: list[dict[str, Any]]) -> None:
    print(f"{'RANK':<5} {'LABEL':<14} {'WORKSPACE':<10} {'QUOTA':<22} {'WAIT':<6} {'RUN':<5} {'STOP':<5} STATUS")
    print("-" * 96)
    for i, row in enumerate(results, 1):
        if not row["query_ok"]:
            status = "UNKNOWN"
            wait = run = stop = "?"
        elif row.get("global_score") is not None:
            wait = row["global_waiting"]
            run = row["global_running"]
            stop = row["stopping"]
            if row["global_waiting"]:
                status = "QUEUE*"
            elif row["global_running"]:
                status = "BUSY*"
            else:
                status = "IDLE*"
        else:
            wait = row["waiting"]
            run = row["running"]
            stop = row["stopping"]
            if row["waiting"]:
                status = "QUEUE"
            elif row["running"]:
                status = "BUSY"
            else:
                status = "IDLE"
        print(f"{i:<5} {row['label']:<14} {row['workspace']:<10} {row['quota']:<22} {wait!s:<6} {run!s:<5} {stop!s:<5} {status}")
        if row.get("waiting_jobs"):
            print(f"      waiting: {', '.join(row['waiting_jobs'][:8])}")
            if len(row["waiting_jobs"]) > 8:
                print(f"      ... {len(row['waiting_jobs']) - 8} more waiting jobs")
        if row.get("running_jobs"):
            print(f"      running: {', '.join(row['running_jobs'][:8])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--credential-uri", default=os.getenv("ALIBABA_CLOUD_CREDENTIALS_URI", DEFAULT_CREDENTIAL_URI))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Return 0 even if some pools failed to query.")
    parser.add_argument(
        "--accept-visible-jobs-only",
        action="store_true",
        help="Allow ranking by list-jobs output even though it may miss quota-global queues.",
    )
    parser.add_argument(
        "--queue-snapshot",
        help="Authoritative console/quota-level queue snapshot JSON. Covers WORKSPACE/QUOTA keys or list entries.",
    )
    parser.add_argument(
        "--manual-queue",
        action="append",
        default=[],
        help="Manual global queue count: WORKSPACE/QUOTA=WAITING[,RUNNING]. Repeat for all pools.",
    )
    parser.add_argument(
        "--allow-partial-snapshot",
        action="store_true",
        help="Allow selecting from a snapshot that does not cover every known pool.",
    )
    parser.add_argument("--mock-file", help="Offline fixture: JSON object keyed by workspace id with job lists.")
    args = parser.parse_args()

    results, errors = rank_pools(args)
    has_global_snapshot = any(row.get("global_score") is not None for row in results)
    full_global_snapshot = has_global_snapshot and not any("queue snapshot does not cover" in err for err in errors)
    visible_best = results[0] if results and results[0].get("query_ok") else None
    if full_global_snapshot or (has_global_snapshot and args.allow_partial_snapshot):
        best = next((row for row in results if row.get("query_ok") and row.get("global_score") is not None), None)
    elif args.accept_visible_jobs_only:
        best = visible_best
    else:
        best = None
    output = {
        "endpoint": args.endpoint,
        "strict": not args.allow_partial,
        "queue_source": "manual_global_snapshot" if has_global_snapshot else "visible_jobs_only",
        "can_prove_global_queue": bool(full_global_snapshot),
        "limitations": [
            "PAI DLC list-jobs is visible-jobs-only for this credential and may miss quota-global queues.",
            "The installed alibabacloud_pai_dlc20201203 SDK exposes no quota-global queue/list-quota operation.",
            "ListJobsRequest.FromAllWorkspaces is documented for current-user job search with ShowOwn=true, not global quota depth.",
            "If console/quota UI shows queued jobs that list-jobs misses, pass a full --manual-queue snapshot before selecting a pool.",
        ],
        "selection_enabled": bool(best),
        "manual_snapshot_covered": [
            _pool_key(row["workspace"], row["quota"])
            for row in results
            if row.get("global_score") is not None
        ],
        "results": results,
        "errors": errors,
        "visible_best": visible_best,
        "best": best,
    }
    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_table(results)
        if output["best"]:
            selected = output["best"]
            if selected.get("global_score") is not None:
                print(f"\nBest global-snapshot pool: --workspace-id {selected['workspace']} --resource-id {selected['quota']} ({selected['label']})")
            else:
                print(f"\nBest visible-jobs pool: --workspace-id {selected['workspace']} --resource-id {selected['quota']} ({selected['label']})")
        else:
            print("\nNo verified best pool selected: list-jobs is visible-jobs-only and cannot prove global quota queue.")
            if visible_best:
                print(
                    "Visible-jobs-only best would be: "
                    f"--workspace-id {visible_best['workspace']} --resource-id {visible_best['quota']} ({visible_best['label']})"
                )
        if errors:
            print("\nQuery errors (not treated as idle):", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
    if errors and not args.allow_partial:
        return 2
    if not output["best"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
