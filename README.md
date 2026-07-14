# Autoresearch Idea Harness

Standalone V2 harness for evidence-conditioned research idea proposal.

This project is intentionally separate from `proposal_rl`.  It treats the idea
proposal model as one replaceable component in a larger loop:

```text
paper bank -> evidence packets -> proposal generators -> expert forecasts -> reports
```

The MVP is train-ready but does not train a model.  It builds auditable JSONL
artifacts that can later feed SFT, reward modeling, or RL.

## Setup

```bash
git clone https://github.com/XinghanLi66/autoresearch_idea_harness.git
cd autoresearch_idea_harness

python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt

cp .env.example .env   # then fill in ANTHROPIC_API_KEY (see below)
```

All paths in `configs/default.yaml` are repo-relative by default and are
resolved against the repository root (env vars like `${MODEL_DIR}` and `~` are
expanded). Point `data/`, `external/`, and `models/` at your own copies, or
edit the config.

To use a local copy of the 32B base model, download it from the HuggingFace
hub and set `MODEL_DIR`:

```bash
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
  --local-dir "$MODEL_DIR/Qwen/Qwen2.5-32B-Instruct"
```

If `MODEL_DIR` is unset, the registry entry falls back to the hub id
`Qwen/Qwen2.5-32B-Instruct` and `transformers` downloads it on demand.

## External reproducibility scope

Everything you need for the external reproduction path is public:

- **External (reproducible with this repo + your own hardware/keys):**
  collate the released V2 SFT dataset -> prepare the chronological SFT run
  (`scripts/prepare_v3_sft_run.py`) -> train the 32B LoRA SFT with `torchrun`
  on your own 8-GPU node (training entry point lives in the sibling
  [`proposal_rl`](../proposal_rl) repo, `train/sft.py`) -> evaluate proposals
  with your own `ANTHROPIC_API_KEY`. See [REPRODUCE.md](REPRODUCE.md) for the
  step-by-step guide and the honest expected outcome.
- **Internal-only (lab infrastructure; released artifacts replace them):**
  - *Runway LLM proxy* (`src/autoresearch_idea_harness/runway_client.py`,
    the `runway_*` generators, and the `target_synthesis` step): an internal
    gateway to Anthropic/OpenAI models. It has no public endpoint; the CoT
    target synthesis it powered is superseded by the released dataset.
    The public path for LLM calls is the standard Anthropic API — use the
    `claude` generator and `ANTHROPIC_API_KEY`.
  - *PAI-DLC / cluster submission* (`scripts/check_dlc_quota.py`,
    `scripts/prepare_v3_*_run.py`, `scripts/watch_v3_*.py`,
    `scripts/monitor_v3_sft_job.py`, `skills/dlc-quota/`): Alibaba PAI-DLC
    job submission helpers for the lab cluster, marked with an internal
    header. Externally, run the generated `run_sft_curriculum.sh` directly
    with `torchrun` instead of submitting a DLC job.

## Quick Smoke

```bash
python scripts/build_paper_bank.py --limit 20
python scripts/build_evidence_packets.py --limit 20
python scripts/generate_proposal_batches.py --limit 20 \
  --generators evidence_synthesis,abstract_ablation,conservative_baseline,local_checkpoint_placeholder
python scripts/aggregate_market.py --make-fixture
python scripts/render_report.py
```

Outputs are written under `runs/`.

## LLM Pipeline

Create a local `.env` from `.env.example` and fill in `ANTHROPIC_API_KEY`
(public path). The `runway_*` generator/expert entries additionally need the
internal Runway gateway (`RUNWAY_BASE_URL` + keys) and only work inside the
lab network; external users should use the `claude` generator instead.
The `.env` file is ignored and should not be committed or copied into reports.

```bash
python scripts/build_paper_bank.py --limit 100
python scripts/build_evidence_packets.py --limit 20
python scripts/generate_proposal_batches.py --limit 20
python scripts/generate_expert_forecasts.py --limit-packets 20
python scripts/aggregate_market.py --forecasts runs/expert_forecasts/llm.jsonl
python scripts/render_report.py
```

Default LLM roles:

- Proposal generator: `runway_opus47_proposal` (internal Runway route;
  externally pass `--generators claude`)
- Expert judges: `opus47`, `gpt55` (internal Runway routes)
- `gpt55` uses the Runway Responses API; `opus47` uses the Google Anthropic
  `rawPredict` API documented under the Gemini service docs.
  For debugging a partially unavailable market, pass `--skip-errors` to
  `scripts/generate_expert_forecasts.py`.

## End-to-End Pipeline

Run an offline self-contained smoke first:

```bash
python scripts/run_end_to_end.py \
  --task dl_activation_function \
  --subtask resnet20-cifar10 \
  --n-proposals 3 \
  --mock-llm \
  --worker-mode fixture
```

Run the live master/scorer stage but stop before worker execution:

```bash
python scripts/run_end_to_end.py \
  --task dl_activation_function \
  --subtask resnet20-cifar10 \
  --n-proposals 3 \
  --worker-mode skip
```

Run the read-only dashboard:

```bash
python scripts/dashboard_tui.py --runs-root runs/end_to_end
```

End-to-end runs write detailed artifacts under
`runs/end_to_end/<task>_<subtask>_<run_id>/`, including `events.jsonl`,
`llm_calls/`, proposal and market files, worker logs, `summary.json`, and
`report.md`.

## V3 Training Manifest

Build the chronological, leakage-guarded manifest for the next 32B training
stage:

```bash
python scripts/discover_v3_base_models.py
python scripts/validate_v3_base_model.py --base-model-id <registered_32b_model>

python scripts/build_v3_training_manifest.py \
  --base-model-id <registered_32b_model>
```

`discover_v3_base_models.py` scans only the configured local model roots with a
bounded depth and writes `runs/reports/v3_base_model_candidates.md/json`. It is
read-only: it never registers a model and intentionally leaves `release_date`
blank, because that date must come from a trusted model card or release record
before leakage-safe training.

This writes `runs/training_data/v3_0_manifest/`:

- `samples.jsonl`: all selected papers, sorted by target-paper `created` date.
- `splits/{train,val,test}.jsonl`: chronological splits, not random splits.
- `synthesis_queue.jsonl`: papers that still need TeX-grounded target synthesis.
- `ready_sft.jsonl`: samples whose target proposal is already cached.
- `ref_evidence_queue.jsonl`: selected references that should receive TeX evidence.
- `manifest.json` and `target_schema.json`: filter counts and target granularity.

The default V3 filter starts with high-quality `method_algorithm` papers whose
route is `tex_target_with_ref_detail`, whose TeX has method/implementation/eval
signals, and whose category is relevant to MLS-style CV/DL work. The manifest
requires an explicit base-model release date so target papers before that date
are excluded rather than accidentally leaking base-model pretraining knowledge.
For formal runs, add the 32B base model to `configs/default.yaml` under
`base_models.registry` with `path`, `release_date`, and `size_b`; direct
`--base-model-release-date` is kept only for planning/smoke usage.

After the manifest exists, synthesize missing TeX-grounded targets into a
resumable cache. **Internal-only:** this step calls Anthropic models through
the internal Runway gateway (`target_synthesis.provider:
runway_google_anthropic`) and also imports legacy synthesis helpers from the
sibling `proposal_rl` checkout, so it cannot be run externally. External
users should skip it and use the released V2 SFT dataset instead (see
REPRODUCE.md).

```bash
python scripts/synthesize_v3_targets.py \
  --manifest-dir runs/training_data/v3_0_manifest \
  --output runs/training_data/v3_0_targets/tex_targets.jsonl \
  --limit 10 \
  --dry-run

python scripts/synthesize_v3_targets.py \
  --manifest-dir runs/training_data/v3_0_manifest \
  --output runs/training_data/v3_0_targets/tex_targets.jsonl \
  --concurrency 2
```

Collate the manifest plus target cache into message-format SFT JSONL:

```bash
python scripts/collate_v3_sft.py \
  --manifest-dir runs/training_data/v3_0_manifest \
  --target-cache runs/training_data/v3_0_targets/tex_targets.jsonl \
  --output-dir runs/training_data/v3_0_sft
```

`synthesize_v3_targets.py --mock` is available for offline plumbing tests only;
do not use mock output for training.

Prepare a chronological SFT run directory after real targets are collated:

```bash
python scripts/prepare_v3_sft_run.py \
  --sft-dir runs/training_data/v3_0_sft \
  --output-dir runs/training/v3_sft \
  --run-id v3_sft_32b_first \
  --base-model-id <registered_32b_model> \
  --phase-by month
```

This writes `run_plan.json`, phase-local `sft_config.yaml` files, prebuilt
message Parquets, `run_sft_curriculum.sh`, and a `dlc_command_skeleton.sh`
(the DLC skeleton is for internal PAI-DLC submission only — externally just
run `run_sft_curriculum.sh` with `torchrun` available; see REPRODUCE.md).
The current default is conservative low-LR LoRA for the first cycle so the
master keeps general debugging/advice ability; move to fuller 32B tuning only
after regression checks.

Inspect the V3 path without mutating runs:

```bash
python scripts/render_v3_training_report.py
python scripts/dashboard_v3_tui.py
```

## Main Artifacts

- `runs/paper_bank/papers.jsonl`: classified target papers joined with original refs.
- `runs/paper_bank/ref_evidence.jsonl`: reference-side TeX evidence cache.
- `runs/evidence_packets/v1/{train,val,test}.jsonl`: expert/eval task packets.
- `runs/training_data/v3_0_manifest/`: chronological training manifest for V3.
- `runs/training_data/v3_0_targets/tex_targets.jsonl`: TeX-grounded target cache.
- `runs/training_data/v3_0_sft/{train,val,test}.jsonl`: message-format SFT dataset.
- `runs/training/v3_sft/<run_id>/`: chronological SFT run plan and launcher.
- `runs/reports/v3_training_status.md`: local V3 status report.
- `runs/proposal_batches/private.jsonl`: proposal batches with model identities.
- `runs/proposal_batches/expert.jsonl`: anonymized expert-facing proposal batches.
- `runs/expert_forecasts/*.jsonl`: expert probability forecasts, fixture or LLM.
- `runs/eval_summary/summary.json`: model/proposal aggregation.
- `runs/reports/market_preview.md`: human-readable preview/report.

## Design Defaults

- JSONL first; no database or vector DB in MVP.
- First route only: `tex_target_with_ref_detail`.
- First paper types: `method_algorithm` and `system_tooling`.
- Evidence packets require at least one reference-side TeX snippet by default;
  use `--allow-metadata-only-refs` only for coverage/debug checks.
- Success definition: expert probability that implementation would beat the
  stated baseline/metric.
- Expert market is holdout evaluation only; no reward training yet.
- Secrets are read only from environment variables or `.env`; generated artifacts
  record model IDs and usage, not API keys.
- The end-to-end runner copies small task/worker interfaces into this project
  and only references large external data/checkouts through config paths.
