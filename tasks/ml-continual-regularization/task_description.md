# Continual Learning: Regularization Strategy Optimization

## Objective
Improve continual learning performance by designing better regularization strategies that prevent catastrophic forgetting. You must implement two functions in `custom_regularization.py`:

1. **`estimate_importance(model, dataset, prev_params, device)`** -- Called once after training on each context finishes. Returns a dict mapping parameter names to importance tensors.

2. **`compute_regularization_loss(model, importance_dict, prev_params_dict)`** -- Called at every training step. Returns a scalar loss that penalizes parameter changes based on their importance.

## Background
In continual learning, a model trains on a sequence of tasks ("contexts") and must retain performance on earlier ones. Regularization-based methods add a penalty term `reg_strength * R(theta)` to the training loss, where R penalizes changes to parameters deemed important for previous tasks.

The two dominant approaches are:
- **EWC** (Elastic Weight Consolidation): Uses the diagonal Fisher Information matrix to estimate importance. Loss is `0.5 * sum(F_i * (theta_i - theta_i^*)^2)`.
- **SI** (Synaptic Intelligence): Tracks gradient-weighted parameter changes during training to estimate importance online. Loss is `sum(omega_i * (theta_i - theta_i^*)^2)`.

Both have known weaknesses: EWC's Fisher is a local approximation that may not capture long-range importance; SI's importance can be noisy. Can you design something better?

## Available hooks
- `model.param_list`: List of generators that yield `(name, param)` pairs for all regularized parameters.
- `model._custom_W`: Dict tracking per-step gradient-weighted parameter changes (accumulated by the training loop). Useful for SI-style approaches.
- `model._custom_p_old`: Dict of parameter snapshots from previous training step.
- `model.gamma`: Decay factor for Fisher accumulation (framework default 1.0; Online EWC typically uses <1, e.g. 0.9).
- `model.epsilon`: Damping constant (default 0.1, used by SI).

## Evaluation
Tested on **3 benchmarks** of increasing difficulty:

| Benchmark | Scenario | Contexts | Description |
|-----------|----------|----------|-------------|
| **Split-MNIST** | Task-incremental | 5 (2 classes each) | Digits split into 5 tasks |
| **Permuted-MNIST** | Domain-incremental | 10 | Same classes, different pixel permutations |
| **Split-CIFAR100** | Task-incremental | 10 (10 classes each) | 100 classes split into 10 tasks |

The primary metric is **average accuracy** across all contexts after all training completes. Higher is better.

## Constraints
- Only modify the editable region of `custom_regularization.py` (the two function bodies).
- Do not create new files.
- The `estimate_importance` function receives the training dataset and can do a forward/backward pass over it.
- The `compute_regularization_loss` function is called at every training step and must be efficient.
