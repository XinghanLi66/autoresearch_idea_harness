# Optimization Bilevel

## Research Question
Can you improve a fixed bilevel-optimization benchmark based on Shen and Chen's penalty-based bilevel gradient descent experiments by selecting a better supported method and tuning only paper-style strategy hyperparameters?

## What You Can Modify
Edit only `penalized-bilevel-gradient-descent/mlsbench/custom_strategy.py` inside the editable block containing:

1. `get_toy_strategy()`
2. `get_hyperclean_strategy(net)`

These functions may only choose among the supported methods already implemented in the fixed driver:
- Toy mode: `v_pbgd`, `g_pbgd`
- Data hyper-cleaning mode: `v_pbgd`, `g_pbgd`, `rhg`, `t_rhg`

You should only change strategy-level choices already present in the paper/codebase, such as:
- method selection
- learning rates
- penalty schedule (`gamma_init`, `gamma_max`, `gamma_argmax_step`)
- inner / outer iteration counts
- RHG truncation depth (`K`) and inner-loop length (`T`)

Do not rewrite the driver, dataset split, pollution protocol, metrics, or model architectures.

## Fixed Setup
### Toy / Numerical Verification
- Problem definition follows Section 5.1 / 6.1 of the paper
- `x` is projected to `[0, 3]`
- 1000 random initial points are sampled as in the official toy script
- Primary metric: `convergence_steps`
- Secondary metrics: `success_rate`, `final_residual`, `runtime_sec`

### Data Hyper-Cleaning
- MNIST split: 5000 train / 5000 validation / 10000 test
- Pollution rate: 50%
- Pollution logic follows the released official code
- Models: linear classifier and 2-layer MLP (`784 -> 300 -> 10`, sigmoid hidden layer)
- Primary metric: `test_accuracy`
- Secondary metrics: `f1_score`, cleaner precision / recall, runtime to best accuracy

## Reference Files
The following official source files are provided read-only for fidelity:
- `penalized-bilevel-gradient-descent/V-PBGD/toy/toy.py`
- `penalized-bilevel-gradient-descent/V-PBGD/data-hyper-cleaning/data_hyper_clean.py`
- `penalized-bilevel-gradient-descent/G-PBGD/data_hyper_clean_gpbgd.py`
- `penalized-bilevel-gradient-descent/RHG/data_hyper_clean_rhg.py`
- `penalized-bilevel-gradient-descent/RHG/hypergrad/hypergradients.py`

## Evaluation
The task runs three benchmark commands:
1. `toy-convergence`
2. `hyperclean-linear`
3. `hyperclean-mlp`

Each command prints structured `TRAIN_METRICS` and `FINAL_METRICS` lines. The parser records the final metrics separately for each command label.

