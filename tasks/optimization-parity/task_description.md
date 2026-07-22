# Optimization Parity

## Research Question
Can you improve a fixed two-layer MLP's ability to learn sparse parity by designing only its initialization, training dataset, and AdamW hyperparameters?

## What You Can Modify
Edit the scaffold file `pytorch-examples/optimization_parity/custom_strategy.py` only inside the editable block containing:

1. `init_model(model, config)`
2. `make_dataset(secret, config, seed)`
3. `get_optimizer_config(config)`

The benchmark is evaluated on three configurations: `(N=32, K=8)`, `(N=50, K=8)`, and `(N=64, K=8)`, all with `W=512`.

## Fixed Setup
- Task: `y = (sum_{i in S} x_i) mod 2` for a hidden secret subset `S`
- Inputs: binary vectors `x in {0,1}^N`
- Model: `Linear(N, W) -> ReLU -> Linear(W, 1) -> Sigmoid`
- Optimizer type: `AdamW`
- Loss: binary cross-entropy
- Batch size: `128`
- Training budget: up to `100000` steps, reshuffling every epoch
- Evaluation: 10 hidden secrets, 10 random epoch-orderings per secret, mean held-out test accuracy over all 100 runs

## Interface Notes
- `init_model(...)` must not depend on the hidden secret.
- `make_dataset(...)` may use the provided secret and must return either `(x, y)` or `{"x": x, "y": y}`.
- `x` must have shape `[num_examples, N]` with binary values only.
- `y` must have shape `[num_examples]` (or `[num_examples, 1]`) with binary labels.
- Training dataset size must stay `<= 12_800_000` examples.
- `get_optimizer_config(...)` must return `lr`, `wd`, `beta1`, and `beta2`.

## Metric
The leaderboard metric is `test_accuracy` (also emitted as `score`), the mean test accuracy across all 100 training runs.

