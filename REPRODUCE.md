# Reproducing the V2 32B CoT-SFT Result

This guide covers the **external** reproduction path: training the 32B LoRA
SFT model on the released V2 dataset with your own hardware and evaluating it
with your own Anthropic API key. Internal-only infrastructure (the Runway LLM
gateway used for CoT target synthesis, and Alibaba PAI-DLC cluster submission)
is **not** required: the released dataset replaces the synthesis step, and a
direct `torchrun` invocation replaces cluster submission.

## Honest expected outcome (read this first)

This is a documented **negative result**. The pipeline and metrics reproduce,
but the V2 32B CoT-SFT checkpoint did **not** beat the base
Qwen2.5-32B-Instruct model on the downstream worker pass-rate benchmark.
Reproducing this run should give you:

- a training run that completes cleanly (~49 minutes on 8x H800 for the
  928-row dataset at 16k sequence length, single epoch, LoRA r=64), with the
  usual loss decrease and checkpoints per chronological phase;
- proposal outputs that follow the trained XML schema;
- a worker-eval pass rate that is **not better** than the untrained base
  model's pass rate on the same tasks.

If your pass rate roughly matches base (rather than exceeding it), you have
successfully reproduced the finding.

## 0. Cross-repo dependency: `proposal_rl`

The SFT **training entry point does not live in this repo**. It lives in the
sibling `proposal_rl` repository at `train/sft.py`, and the launcher generated
by this repo runs:

```bash
cd <proposal_rl_root> && torchrun --nproc_per_node=$NGPU train/sft.py --config <phase>/sft_config.yaml
```

Clone `proposal_rl` next to this repo (the default
`proposal_rl_root: ../proposal_rl` in `configs/default.yaml` then works
as-is), or set `proposal_rl_root` to wherever you put it. The training
environment additionally needs `peft>=0.15`, `accelerate>=1.10` (1.14.0 verified), and `datasets`
(see `proposal_rl`'s requirements; pins verified by independent-cluster replication 2026-07-15 —
older peft/accelerate fail at LoraConfig/32B-load with transformers 5.5 + verl 0.7.1; remove torchao if present).

## 1. Environment

```bash
git clone https://github.com/XinghanLi66/autoresearch_idea_harness.git
cd autoresearch_idea_harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # fill in ANTHROPIC_API_KEY for the eval step
```

Base model — either predownload it:

```bash
export MODEL_DIR=/path/to/hf_models
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
  --local-dir "$MODEL_DIR/Qwen/Qwen2.5-32B-Instruct"
```

or leave `MODEL_DIR` unset and let `transformers` pull
`Qwen/Qwen2.5-32B-Instruct` from the hub on first use.

## 2. Get the released V2 SFT dataset

Download the released dataset (message-format SFT JSONL, chronological
train/val/test splits, 928 usable training rows):

```bash
# https://huggingface.co/datasets/<HF_NAMESPACE>/arxiv-proposal-cot-sft-32b-v2 (fill <HF_NAMESPACE> after upload)
huggingface-cli download <HF_NAMESPACE>/arxiv-proposal-cot-sft-32b-v2 --repo-type dataset \
  --local-dir runs/training_data/v3_0_sft
```

You should end up with `runs/training_data/v3_0_sft/{train,val,test}.jsonl`.
This replaces the internal-only steps (quality cache -> V1-semantics prompts
-> Runway CoT target synthesis -> collate). If you want to re-collate from a
manifest + target cache of your own, use `scripts/collate_v3_sft.py`; the
synthesis step itself (`scripts/synthesize_v3_targets.py`) is internal-only.

## 3. Prepare the SFT run

```bash
python scripts/prepare_v3_sft_run.py \
  --sft-dir runs/training_data/v3_0_sft \
  --output-dir runs/training/v3_sft \
  --run-id v2_32b_repro \
  --base-model-id qwen25_32b_instruct \
  --phase-by month \
  --max-seq-length 16384
```

This writes `runs/training/v3_sft/v2_32b_repro/` containing `run_plan.json`,
per-phase `sft_config.yaml` + prebuilt message Parquets, and the launcher
`run_sft_curriculum.sh`. (It also emits a `dlc_command_skeleton.sh` /
`pai_create_job*.sh` — those are internal PAI-DLC submission artifacts;
ignore them.)

Alternatively, skip the curriculum tooling and point `proposal_rl/train/sft.py`
at a single hand-written `sft_config.yaml` referencing the train Parquet.

## 4. Train (direct torchrun, 8-GPU node)

```bash
export NGPU=8
bash runs/training/v3_sft/v2_32b_repro/run_sft_curriculum.sh
```

The launcher iterates the chronological phases; each phase runs

```bash
cd ../proposal_rl && torchrun --nproc_per_node=$NGPU train/sft.py \
  --config runs/training/v3_sft/v2_32b_repro/curriculum/<phase>/sft_config.yaml
```

Reference wall-clock: **~49 minutes on 8x H800** for 928 rows at 16k max
sequence length (LoRA r=64, alpha=128, lr 1e-5, 1 epoch). Any 8-GPU node with
~80 GB per GPU should work; adjust `per_device_train_batch_size` /
`gradient_accumulation_steps` for smaller GPUs.

## 5. Evaluate

Generation-side sanity checks on the trained checkpoint:

```bash
python scripts/generate_v3_checkpoint_proposal.py --model-dir runs/training/v3_sft/v2_32b_repro/checkpoints/<last_phase>/final ...
python scripts/demo_held_out_proposals.py --model-dir runs/training/v3_sft/v2_32b_repro/checkpoints/<last_phase>/final
```

Worker-eval (proposal -> Claude Code worker implements it -> benchmark
pass/fail) needs `ANTHROPIC_API_KEY` in your environment or `.env`, the
Claude Code CLI on PATH (`CLAUDE_CMD` env var to override), plus the MLS-Bench
checkout configured at `mls_bench_root`:

```bash
python scripts/run_end_to_end.py \
  --task dl_activation_function \
  --subtask resnet20-cifar10 \
  --n-proposals 3
python scripts/render_v3_eval_comparison.py
```

Compare the trained checkpoint's pass rate against the base
`Qwen/Qwen2.5-32B-Instruct` under the same worker/task settings. Expected:
**no improvement over base** (the documented V2 negative result).

## Internal-only components (for transparency)

| Component | Where | External replacement |
|---|---|---|
| Runway LLM gateway | `src/autoresearch_idea_harness/runway_client.py`, `runway_*` generators/experts | Standard Anthropic API (`claude` generator, `ANTHROPIC_API_KEY`) |
| CoT target synthesis | `scripts/synthesize_v3_targets.py` (`target_synthesis` config) | Released V2 dataset (step 2) |
| PAI-DLC submission | `scripts/check_dlc_quota.py`, `scripts/prepare_v3_*_run.py`, `scripts/watch_v3_*.py`, `scripts/monitor_v3_sft_job.py`, `pai_create_job*.sh` artifacts | Direct `torchrun` (step 4) |
| Multi-host tmux sweep launcher | `scripts/launch_v2_3_formal_shards.sh` | Run `scripts/run_formal_sweep.py` directly on one node |
| proposal_rl experiment checkpoints (V2.3 sweep modules) | `formal_sweep.py` module table (`PROPOSAL_RL_RUNS_ROOT`) | Not released; external sweep covers worker-only and API-master modules |

## Replication receipts (independent cluster, 2026-07-15)

This branch's training path was replicated on an independent 4-GPU (aarch64 GB200) cluster,
cloning THIS branch (plus `proposal_rl@opensource-prep` for `train/sft.py`) over HTTPS:

| Step | Result | Evidence |
|---|---|---|
| Clone + `py_compile` on fresh image | PASS | trial 1697055 |
| Smoke (256 rows, 32B LoRA, 4 GPUs) | PASS — losses 1.87–2.00, sharded checkpoint saved | trial 1699582 |
| FULL run (928 rows, 16k seq, LoRA r64/α128, lr 1e-5, 1 epoch) | PASS — loss 2.033 → 1.837 (min, s7) → 1.880 (s14), checkpoint `global_step_14` saved | trial 1699595 |

This matches the documented expectation: training reproduces cleanly (the recorded result is a
no-pass-rate-gain vs base — see the top of this file). Environment notes confirmed:
`peft>=0.15`, `accelerate>=1.10` (1.14.0 verified) with `transformers==5.5.0` + `verl==0.7.1`;
remove old `torchao` if present. Known cosmetic issue: verl's post-training FSDP→HF merge can fail
with a backend-enum `AttributeError` — the sharded checkpoint is intact; merge offline via
`python -m verl.model_merger`.
