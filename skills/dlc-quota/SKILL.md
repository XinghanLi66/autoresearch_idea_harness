---
name: dlc-quota
description: Use before selecting a PAI-DLC workspace/quota, launching a DLC job, or diagnosing DLC queue wait. Requires fail-closed quota checks: API failures or malformed responses must never be treated as idle capacity.
---

# DLC Quota Selection

> **INTERNAL:** This skill targets the lab's Alibaba PAI-DLC quota pools and is
> not needed for external reproduction (see REPRODUCE.md).

Use this skill before any PAI-DLC launch, retry, or quota decision.

## Required Command

Run the fail-closed checker from the repository root:

```bash
python autoresearch_idea_harness/scripts/check_dlc_quota.py
```

For machine-readable output:

```bash
python autoresearch_idea_harness/scripts/check_dlc_quota.py --json
```

If sandbox blocks the local credential endpoint or DLC API, rerun the same command with required approval. Do not infer availability from a failed command.

## Decision Rules

- A pool is selectable only when `query_ok=true`.
- API failure, credential failure, timeout, malformed JSON, or missing `Jobs` is `UNKNOWN`, not idle.
- `list-jobs` output is **visible-jobs-only**. It may miss quota-global queues from jobs the current credential cannot see.
- The installed `alibabacloud_pai_dlc20201203` SDK has no quota-global queue/list-quota operation. `ListJobsRequest.FromAllWorkspaces` is documented for current-user job search with `ShowOwn=true`, not quota depth.
- Default checker output must not be treated as proof of zero queue. It intentionally returns no `best` unless `--accept-visible-jobs-only` is passed.
- If the console or quota page shows a queue count such as 17 but `list-jobs` shows 0, trust the quota page for wait-time decisions and mark script output as insufficient.
- Prefer the pool with the lowest `waiting` count.
- Break ties by lower `running`, then lower `stopping`.
- If every pool is `UNKNOWN`, do not launch.
- If the selected job command depends on the shared cluster filesystem, the dry-run payload must include `DataSources` with the matching `MountPath`.
- Always inspect the dry-run payload before real submit and confirm `WorkspaceId`, `ResourceId`, `Priority`, `Image`, `DataSources`, and command file.

## Manual Console Snapshot

When the PAI console/quota page shows queue counts, pass all four pool counts explicitly:

```bash
python autoresearch_idea_harness/scripts/check_dlc_quota.py \
  --manual-queue 224239/quota1ecrg95m4n9=17 \
  --manual-queue 137902/quotadbz1mvpy1v5=0 \
  --manual-queue 238626/quota1d8xmvdw5tb=0,1 \
  --manual-queue 262162/quota1shcr2h7uae=0
```

The format is `WORKSPACE/QUOTA=WAITING[,RUNNING]`. By default the snapshot must cover every known pool before the checker selects `best`.

## Current Known Pools

- `224239 / quota1ecrg95m4n9` (`M0-Dots_V3`)
- `137902 / quotadbz1mvpy1v5`
- `238626 / quota1d8xmvdw5tb`
- `262162 / quota1shcr2h7uae`

## Memory Update

After choosing or rejecting a pool, append concise status to:

- `agent-memory/exp-manager/jobs.md`
- `agent-memory/checkins.md`

Record command, chosen workspace/quota, queue counts, dry-run result, submitted job id if any, and next monitor action.
