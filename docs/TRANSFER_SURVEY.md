# Project Transfer Survey & Artifact Index — idea-proposal-training

_Owner: lixinghan (HF account `coder66` = yours). Compiled by cc001, last updated 2026-07-31.
This is the canonical index of everything transferred for continuing the project off-machine.
Status legend: ✅ done/verified · ⏳ in progress (QS side, cc000) · ⚠️ note._

## Quick verdict
Everything reachable from D0 is uploaded & verified. The QS-3FS-only trained weights + the
QS-ported benchmark are being pushed by cc000 (separate machine); their rows are marked ⏳ with
the target repo names. To rebuild on a new machine: clone the 4 GitHub repos, pull the HF
datasets + model repos, recreate conda envs from the env specs, restore Claude sessions, `git
apply` the MLS-Bench patch (or clone `mlsbench-qs-code`).

---
## 1. CODE — GitHub (all pushed ✅)
| Repo | Branch | Commit | Contents |
|------|--------|--------|----------|
| `XinghanLi66/autoresearch_idea_harness` | V3 | latest | core code (MLS-lite + MAB pipelines, proposal-gen, V3 training), 108 RedDoc MD exports, MLS-Bench harness patch, findings, HF upload scripts, this survey |
| `XinghanLi66/proposal_rl` | main | `cc5dc27` | V1 RL pipeline (+ WIP preserved on branch `local-wip-20260731`) |
| `XinghanLi66/idea-proposal-training-paper` | main | `919538a` | arXiv paper (method/experiments/takeaways + tables), `main.pdf` tracked |
| `XinghanLi66/idea-proposal-training-dashboard` | main | `12c2910` | results dashboard (MLS 21×30 + raw arm + MAB), GitHub Pages |

## 2. MODELS — HF (private under `coder66`)
### V1/V2 (public, pre-existing) ✅
- `coder66/proposal-rl-qwen2.5-7b-ppl-grpo` (V1 7B RL) · `coder66/proposal-cot-sft-qwen2.5-32b-lora` (V2 32B LoRA)

### V3 dense — uploaded from D0 (private) ✅
- `coder66/proposal-qwen3-8b-v3-sft` — `sft8b` (S1; base for `qwen3-8b-rl`)
- `coder66/proposal-qwen3-14b-v3-sft` — `sft14b` (S2; base for `qwen3-14b-rl`)
- `coder66/proposal-qwen3-32b-v3-sft` — `sft32b` (S3; base for `qwen3-32b-rl`)
- `coder66/proposal-qwen2.5-32b-v3-dpo` — `rl` arm (SFT parent = qwen25sft)

### V3 QS-3FS-only — cc000 uploading (private) ⏳
- `d1sft` (DeepSeek-R1-8B full, flagship) · `d1rl` (LoRA→d1sft)
- `m2sft` (235B LoRA adapter, 813MB) · `m2rl` (235B LoRA→m2sft) — 438G merged blobs intentionally skipped
- `qwen3-8b-rl` / `qwen3-14b-rl` / `qwen3-32b-rl` (LoRAs → S1/S2/S3)
- `qwen25sft` (V3 full SFT Qwen2.5-32B, the `rl` arm's SFT parent)
- ⚠️ `qwen3-30b-a3b`: not on /newcpfs and not a final eval arm — cc000 confirming whether a 30B-A3B MoE was trained on 3FS.
- Base models (DeepSeek-R1-0528-Qwen3-8B, Qwen3-235B-A22B, Qwen3-8B/14B/32B, Qwen2.5-32B-Instruct) are public HF → reference, not uploaded.

## 3. DATA — HF (private under `coder66`)
- `coder66/idea-proposal-training-v3-data` ✅ — V3 researcher-CoT SFT datasets (4 variants) + 31 conda env specs (`mab` + 30 `mlsbench-*`, incl. `mab.environment.yml`)
- `coder66/arxiv-research-proposals-v1` ✅ (public) · `coder66/arxiv-proposal-cot-sft-32b-v2` ✅ (public)

## 4. EVALUATION — HF (private) ✅
`coder66/idea-proposal-training-v3-eval`:
- 21-arm proposals (MLS + MAB) · `report_mls.json` + `report_mab.json` · dashboard index snapshots · findings
- `eval_traces_final.tar.gz` — 1,572 raw worker run dirs
- **`eval_traces_complete.tar.gz`** — 735 complete per-cell bundles, each containing **problem + master_prompt + master_proposal + worker_prompt + worker_trace + generated_code + result** (630 MLS + 105 MAB; worker traces for the 625 cells with a scoring run). See `EVAL_TRACES_COMPLETE.md` in the repo.

## 5. DOCS — RedDoc → Markdown ✅
- 108 pages exported (all 37 root project write-ups + 详细报告 + 70 Pilot CoTs; 66 ICML external-paper reviews excluded per lixinghan) → `autoresearch_idea_harness/docs/redoc_exports/` (in GitHub).
- Enumerated authoritatively via `docs:menu-list` on space `4709bcf0…` (174 total pages in space).

## 6. CLAUDE SESSIONS + AGENT MEMORY — HF (private) ✅
`coder66/agentic-training-claude-sessions`:
- 16 transcripts + 31 memory files + 15 skills + `agent-memory/` (50M) + config, with `RESTORE.md` (resume via `claude --resume`; keep cwd path or rename the project-key dir).

## 7. BENCHMARK HARNESS
- **MLS-Bench mods** ✅ — eval-fix patch (17 files) + base ref at `autoresearch_idea_harness/docs/eval/mls_bench_harness_mods/`.
- **`mlsbench-qs-code`** ⏳ — QS-ported modified MLS-Bench at `/mnt/3fs/lxh/mlsbench`; cc000 to confirm/push the branch to GitHub.
- MLAgentBench — upstream `snap-stanford/MLAgentBench` (re-clonable); our adapter is in the harness repo (`scripts/mab_*.py`).

## Open QS-side items (cc000) — the only things not yet confirmed
1. Upload the 9 QS-3FS weights/adapters (§2) → private HF; report URLs.
2. Confirm/upload `qwen3-30b-a3b` (or confirm it was never trained).
3. Push `mlsbench-qs-code` to GitHub.
_(Per lixinghan, QS/PAI infra itself is not needed going forward — only these artifacts.)_
