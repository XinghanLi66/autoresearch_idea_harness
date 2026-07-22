# Flow Maps: Self-Distillation Loss Design

## Objective

Improve the **self-distillation training objective** for flow maps (NeurIPS 2025, [flow-maps](https://github.com/nmboffi/flow-maps)) so that, under a **fixed training budget** on CIFAR-10, **Fréchet Inception Distance (FID)** is **lower** than strong baselines.

## Composition

- **Three baselines** (reference loss implementations from the paper codebase): **lsd**, **psd_uniform**, **psd_midpoint**.
- **Three benchmarks** (model scale): **`train_small`**, **`train_medium`**, **`train_large`** — scripts set ``FLOWMAPS_UNET_SIZE`` (small / medium / large EDM2 presets in ``configs.cifar10_bench``).

Each baseline run applies the corresponding reference ``losses.py`` and sets ``scripts/bench_env.sh`` so ``learn.py`` uses the right **``slurm_id``** (0 = LSD, 1 = PSD uniform, 2 = PSD midpoint). The same three **train_*.sh** scripts are used for every baseline; only the baseline edit and ``bench_env.sh`` differ.

## Background

Flow maps combine a **diagonal** (interpolant / velocity) term and **off-diagonal** self-distillation. The codebase provides:

- **LSD** — Lagrangian Self-Distillation (`lsd_term`)
- **PSD** — Progressive Self-Distillation (`psd_term`), with **uniform** or **midpoint** teacher mixing

## Task

You may **only** change the implementations of:

- `diagonal_term`, `psd_term`, `lsd_term`, `esd_term`

in `flow-maps/py/common/losses.py` (editable lines **36–178** after the scaffold `mid_edit`; same span as `losses_template.py`). Do **not** alter `setup_loss`, imports, or ``scripts/bench_env.sh`` unless the harness changes.

The goal is a **lower FID** under **batch_size=128**, **max_training_step=50000**, **num_eval_step=8**, and **num_fid_samples=50000**. Do not change these hyperparameters.

## Evaluation

``TEST_METRICS: fid=…, best_fid=…`` — the parser records **per-benchmark** keys so one ``test()`` fills all size columns:

- ``fid_train_small``, ``best_fid_train_small``
- ``fid_train_medium``, ``best_fid_train_medium``
- ``fid_train_large``, ``best_fid_train_large``

**Data**: CIFAR-10 via TensorFlow Datasets; scripts ensure ``cifar_stats.npz`` exists.

## Baselines

```bash
mlsbench baseline cv-flowmaps-training --name lsd
mlsbench baseline cv-flowmaps-training --name psd_uniform
mlsbench baseline cv-flowmaps-training --name psd_midpoint
mlsbench baseline cv-flowmaps-training   # all three baselines in parallel
```

By default each baseline run executes **all three** size scripts (`train_small`, `train_medium`, `train_large`). To run **only the small UNet** for every baseline, pass **`--label train_small`** (CLI filters `test_cmds`):

```bash
# All three baselines, small model only (three parallel baseline workspaces)
mlsbench baseline cv-flowmaps-training --label train_small

# One baseline, small model only
mlsbench baseline cv-flowmaps-training --name lsd --label train_small
```

Similarly: `--label train_medium` or `--label train_large` for a single size. To combine sizes, pass `--label` multiple times, e.g. `--label train_small --label train_medium`.

## Setup

1. ``python -m mlsbench.cli fetch --name flow-maps``
2. Build the ``flow-maps`` container.
3. Writable ``DATASET_LOCATION`` for TFDS and FID stats.
