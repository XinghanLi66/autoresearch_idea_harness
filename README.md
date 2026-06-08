# Autoresearch Idea Harness

Standalone V2 harness for evidence-conditioned research idea proposal.

This project is intentionally separate from `proposal_rl`.  It treats the idea
proposal model as one replaceable component in a larger loop:

```text
paper bank -> evidence packets -> proposal generators -> expert forecasts -> reports
```

The MVP is train-ready but does not train a model.  It builds auditable JSONL
artifacts that can later feed SFT, reward modeling, or RL.

## Quick Smoke

```bash
cd /newcpfs/lxh/agentic-training/autoresearch_idea_harness

python scripts/build_paper_bank.py --limit 20
python scripts/build_evidence_packets.py --limit 20
python scripts/generate_proposal_batches.py --limit 20 \
  --generators evidence_synthesis,abstract_ablation,conservative_baseline,local_checkpoint_placeholder
python scripts/aggregate_market.py --make-fixture
python scripts/render_report.py
```

Outputs are written under `runs/`.

## LLM Pipeline

Create a local `.env` from `.env.example` and fill Runway keys/model names there.
The `.env` file is ignored and should not be committed or copied into reports.

```bash
cd /newcpfs/lxh/agentic-training/autoresearch_idea_harness

python scripts/build_paper_bank.py --limit 100
python scripts/build_evidence_packets.py --limit 20
python scripts/generate_proposal_batches.py --limit 20
python scripts/generate_expert_forecasts.py --limit-packets 20
python scripts/aggregate_market.py --forecasts runs/expert_forecasts/llm.jsonl
python scripts/render_report.py
```

Default LLM roles:

- Proposal generator: `runway_opus47_proposal`
- Expert judges: `opus47`, `gpt55`
- `gpt55` uses the Runway Responses API; `opus47` uses the Google Anthropic
  `rawPredict` API documented under the Gemini service docs.
  For debugging a partially unavailable market, pass `--skip-errors` to
  `scripts/generate_expert_forecasts.py`.

## End-to-End Pipeline

Run an offline self-contained smoke first:

```bash
cd /newcpfs/lxh/agentic-training/autoresearch_idea_harness

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

## Main Artifacts

- `runs/paper_bank/papers.jsonl`: classified target papers joined with original refs.
- `runs/paper_bank/ref_evidence.jsonl`: reference-side TeX evidence cache.
- `runs/evidence_packets/v1/{train,val,test}.jsonl`: expert/eval task packets.
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
