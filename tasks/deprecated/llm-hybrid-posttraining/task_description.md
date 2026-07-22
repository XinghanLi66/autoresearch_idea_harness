# LLM Hybrid Post-Training: Shared-Train Question Router

## Research Question
Implement a better **question-level hybrid router** for `Unify-Post-Training`'s HPT scaffold.

This task keeps the repo's original HPT-style training path much more intact:

- `algorithm.adv_estimator=grpo`
- `trainer.unify_strategy=switch`
- `actor.offline_loss_type=sft`

and opens the trainer-side routing logic that decides whether a question should
keep its on-policy RL samples or switch to an off-policy SFT target.

The editable code is all in:

- `Unify-Post-Training/hpt/verl/verl/mix_src/mix_trainer.py`

The editable regions are:

- `select_on_off_ada_balance()` at lines `394-414`
- per-question statistics and routing state at lines `599-620`
- off-policy sample construction at lines `759-799`
- on-policy retention/deletion at lines `942-963`

The controller still returns:

```python
(on_remove_num, on_add_num, off_add_num)
```

Current semantics are:

- `on_remove_num`: whether / how much on-policy data for this question is removed
- `on_add_num`: how many additional on-policy rollouts are generated
- `off_add_num > 0`: add off-policy SFT samples
- `off_add_num < 0`: add off-policy RL samples (`whether_off=True`)

## What Is Fixed
The rest of the HPT scaffold is intentionally fixed and read-only:

- `mix_actor.py` uses `offline_loss_type=sft`
- `mix_trainer.py` uses `algorithm.adv_estimator=grpo`
- `mix_vllm_rollout.py` and `mix_hf_rollout.py` define how on-policy and off-policy sequences are built
- `rl_dataset_with_target.py` provides each prompt's demonstration target

So the benchmark question is not “invent a whole new post-training stack”. It is:

**Given a fixed HPT training stack, what is the best per-question routing rule between on-policy RL and off-policy SFT?**

## Evaluation Setup
The task trains **once** on a shared mixed OpenR1 subset and validates on three benchmark splits.

### Reproduction notes

Run the full task through MLS-Bench rather than launching the relay commands
alone. The `AIME24` command performs the shared training run and writes
`final_metrics.txt`; the `AMC23` and `MATH-500` commands only relay the
corresponding scores from that file. If you run scripts manually, use the same
`OUTPUT_DIR` for the training and relay commands.

Baseline training requires a machine with:

- `4` visible CUDA GPUs with at least H100/H200-class memory for the default
  `rollout.name=vllm`, `tensor_model_parallel_size=2`, and 16K-context run.
- Python `3.10` in a conda environment.
- CUDA-compatible PyTorch, vLLM, Ray, flash-attn, xFormers, pandas/pyarrow,
  `tensordict`, `codetiming`, `pyvers`, `math-verify==0.6.0`, and
  `latex2sympy2_extended==1.0.9`.
- Local copies of `Qwen2.5-Math-1.5B` and `upt-data/openr1.parquet` prepared
  by the package `data_deps`.

For local conda execution, first put `conda` on `PATH`, then run:

```bash
PYTHONPATH=src python3 -m mlsbench build Unify-Post-Training --config configs/config.gpublaze.local.yaml
PYTHONPATH=src python3 -m mlsbench baseline llm-hybrid-posttraining --name hpt --seed 42 --config configs/config.gpublaze.local.yaml
```

If an older `mlsbench-Unify-Post-Training` environment already exists, rebuild
with `--force` so the package `local_install_cmds` install vLLM and the pinned
training dependencies.

MLS-Bench applies package-level `pre_edit.py` infrastructure patches before the
task edit ops. Those patches keep the HPT controller as the only task-level
research surface while making the upstream UPT checkout runnable in the
benchmark harness.

### Model and runtime

- Model: `Qwen/Qwen2.5-Math-1.5B`
- Backend: `rollout.name=vllm`, `tensor_model_parallel_size=2`
- GPUs: `4 x H200`
- Training objective scaffold: fixed `offline_loss_type=sft`
- Advantage estimator: fixed `grpo`
- Reward implementation: fixed `reward_impl_version=6`

### Training settings

- `train_batch_size=8`
- `max_response_length=8192`
- `ppo_mini_batch_size=4`
- `ppo_micro_batch_size=1`
- `ppo_max_token_len_per_gpu=32768`
- `rollout.n=8`
- `rollout.n_verify=8`
- `rollout.n_val=4`
- `total_training_steps=25`
- `actor.use_dynamic_bsz=False`
- `actor.model.use_remove_padding=True`
- model override: `max_position_embeddings=16384`, `rope_theta=40000`

### Data split

The task builds:

- one deterministic mixed OpenR1 train subset with `144` requested rows
- source mix: `amc_aime:72`, `cn_contest:36`, `olympiads:36`
- within each source bucket, rows are chosen by shortest `(prompt + target)` proxy length
- in the current repo path, prompt-length filtering inside `RLHFDataset` reduces the effective
  trainable set to `90` rows for seed `42`

and validates on:

1. `AIME24`: `20` fixed held-out examples
2. `AMC23`: `20` fixed held-out examples
3. `MATH-500`: `30` fixed held-out examples (`hidden`)

These eval subsets are fixed by committed parquet files under
`data/eval_subsets_20260426/` as `AIME24_eval.parquet`, `AMC23_eval.parquet`,
and `MATH-500_eval.parquet`, with sizes `20 / 20 / 30` (`AIME24` / `AMC23` / `MATH-500`).
`train_shared_hpt.sh` points directly to these files, so evaluation is always
performed on the exact same fixed split without any runtime subset-selection logic.

## Metric
The metric is the **final validation score** for each benchmark.

The underlying repo metrics come from:

```text
val/test_score/<benchmark>
```

Higher is better.

## Baselines On This Shared Scaffold
All visible baselines use the same fixed script and differ only inside the
opened trainer routing regions:

- `sft`: always remove all on-policy samples for a question and add one off-policy demonstration
- `grpo`: never switch; keep pure on-policy updates
- `hpt`: the repo's original HPT switch logic

## Guidance
Useful read-only files:

- `mix_actor.py`
- `mix_core_alg.py`
- `mix_vllm_rollout.py`
- `mix_hf_rollout.py`
- `rl_dataset_with_target.py`
- `exp_scripts/train.sh`
- `exp_scripts/train_luffy.sh`

You should optimize the router in a way that still makes sense as a
post-training algorithm, not as a benchmark-specific hack. Good solutions
usually use:

- current pass rate / reward statistics
- training stage (`global_steps`)
- historical per-question signals
- explicit routing between RL and SFT sample types

without rewriting the fixed actor loss implementation.
