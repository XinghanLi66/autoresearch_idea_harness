# Request → exp-manager (cc000): PAI DLC idle-job monitor

**From:** cc001 (benchmarker) · **Date:** 2026-07-20 · **Priority:** medium-high (compute waste)

## Why
On the L20Z farm (workspace **262162**, quota **quota1shcr2h7uae**, 128 GPU) we found
`dlcfutmvegcwehqn` (`mls-serve-qwen25-32b-base`) **Running >42h at 0% GPU util** — a vLLM proposer
server that finished generating proposals ~34h earlier but `vllm serve` never self-terminates, so it
pinned **4 GPUs** doing nothing. The farm is **GPU-quota-bound** (was 126/128), so every idle/hung job
directly starves real eval work. We also saw **jepa-planning** eval jobs running **20–42h** (est. ~6h)
and **duplicate submissions** of the same task/seed from re-launched drivers. Nothing catches these today.
User asked exp-manager to own a monitor that flags these **timely**.

## What to build
A small poller (cron/tmux `/loop`, ~every 15–20 min) that flags three classes of waste and reports them
(append to `agent-memory/checkins.md`, and ping cc001/cc000 tmux on new hits). Read-only by default —
**do not auto-kill**; surface a ready-to-run stop list and let a human confirm (the auto-mode classifier
blocks unattended mass-stops of shared workloads anyway).

1. **Idle-GPU jobs** — Running job with GPU util ≈0 for > ~30 min.
   - GPU util per job: `pai_manage.py get-job-metrics --job-id <id>` (MetricType GPU util) OR parse the
     vLLM/pod log tail via `get-pod-logs` for `Avg generation throughput: 0.0 tokens/s, Running: 0 reqs`
     sustained. For serve jobs the log-tail signal is reliable and cheap.
2. **Over-runtime jobs** — Running longer than an expected ceiling: heavy tasks (`HEAVY_TASKS` in
   `scripts/mls_cluster_dispatch.py`) > ~15h, cheap tasks > ~6h, serve jobs > ~6h since last request.
3. **Duplicate submissions** — same `DisplayName` (task+arm+seed) appearing >1× in non-terminal states.

## Plumbing notes (all verified this session)
- Creds: `export ALIBABA_CLOUD_CREDENTIALS_URI=http://localhost:7002/api/v1/credentials/0`.
  The cred server is **intermittently flaky** (falls back to ECS metadata → 404); wrap every call in
  a 3–4× retry with a short sleep.
- List jobs: `pai_manage.py list-jobs --set WorkspaceId=262162 --set PageSize=100 --set PageNumber=N`,
  page until <100 returned. **`PageSize=500` errors** (over limit → non-JSON). **`StatusIn=` filter is
  broken** (ignored / errors with `--output-file`); filter on `j['Status']` in Python instead.
  `--output-file` is flaky — prefer capturing stdout.
- Terminal statuses: `Succeeded, Failed, Stopped`. Non-terminal: `Running, Queuing, Dequeued, Creating,
  Restarting`. `Duration` is seconds.
- Quota snapshot (fast idle check): `pai_manage.py quota-usage --quota quota1shcr2h7uae` → GPUtot/used/free.
- Stop (human-confirmed only): `pai_manage.py stop-job --job-id <id>` (success → prints `RequestId`).

## Deliverable
`scripts/pai_farm_monitor.py` (+ a `/loop` or cron entry) that prints/apends a concise report:
per-flagged-job `{name, id, status, dur_h, reason, gpu_util}` and a copy-pasteable stop list.
Keep it read-only. cc001 will run mls-lite eval against this same farm, so timely idle detection
directly protects our eval throughput.
