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

### V3 QS-3FS-only — uploaded by cc000 (private) ✅ (verified live 2026-07-31)
- `coder66/idea-proposal-training-d1sft` — DeepSeek-R1-8B full (flagship), 17 files
- `coder66/idea-proposal-training-qwen25sft` — V3 full SFT Qwen2.5-32B (`rl` arm's SFT parent), 41 files
- `coder66/idea-proposal-training-d1rl-lora` — LoRA → d1sft
- `coder66/idea-proposal-training-m2sft-lora` — 235B LoRA adapter (813MB) → Qwen3-235B-A22B
- `coder66/idea-proposal-training-m2rl-lora` — 235B RL LoRA (53G) → m2sft
- `coder66/idea-proposal-training-qwen3-8b-rl-lora` / `-14b-rl-lora` / `-32b-rl-lora` — LoRAs → S1/S2/S3
- `coder66/idea-proposal-training-m1-qwen3-30b-a3b-sft-lora` — **Qwen3-30B-A3B MoE SFT LoRA** (M1, ckpt-814, base Qwen3-30B-A3B) — the "qwen3 30b" arm, confirmed trained & uploaded
- (438G merged 235B blobs intentionally skipped — adapter+base+recipe instead. Base models are public HF → reference, not uploaded. LoRA cards carry valid public `base_model` + lineage in body.)

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
- **`mlsbench-qs-code`** ⏳ — QS-ported modified MLS-Bench at `/mnt/3fs/lxh/mlsbench` (1.9G, 72k files; not a git repo). cc000 bundling as tar.gz → private `coder66/mlsbench-qs-code` (GitHub would need LFS for 1.9G). URL pending.
- MLAgentBench — upstream `snap-stanford/MLAgentBench` (re-clonable); our adapter is in the harness repo (`scripts/mab_*.py`).

## Status — everything confirmed except one QS bundle
All D0-side artifacts ✅ uploaded & verified. All 9 QS model repos ✅ verified live (private).
`qwen3-30b-a3b` ✅ confirmed trained & uploaded. **Only remaining:** the `mlsbench-qs-code`
tar.gz bundle → `coder66/mlsbench-qs-code` (cc000 finishing; URL to be appended here).
_(Per lixinghan, QS/PAI infra itself is not needed going forward — only these artifacts.)_
