# RAIN Convex-Concave

## Research Question
Can you improve gradient-norm convergence on the exact convex-concave benchmark instances used by the official RAIN repository for `src/bilinear_func/exp_gnorm.m` and `src/delta_func/exp_gnorm.m`?

## What You Can Modify
Edit only the scaffold file `RAIN/optimization_convex_concave/custom_strategy.py` inside the editable block containing:

1. `init_state(problem, initial_z, seed, hyperparameters)`
2. `step(state, oracle, problem, hyperparameters, max_sfo_calls)`
3. `get_hyperparameters(problem_name, sigma)`

The benchmark harness, problem definitions, update-noise model, official iteration counts, initializations, and metric computation are fixed.

## Fixed Setup
- Problems:
  - `bilinear`: the official scalar bilinear problem `f(x, y) = x y` with `n = 900`, `tau = 0.1`, `z0 = [10, 10]^T`, `sigma = 0.001`
  - `delta_nu`: the official `(delta, nu)` problem with `d = 100`, `delta = 1e-2`, `nu = 5e-5`, `n = 6000`, `tau = 1`, `sigma = 0.02`, and `z0 ~ N(0, I)` under the script's fixed RNG seed
- The harness mirrors the official scripts' additive Gaussian update noise, not the earlier generalized SFO sweep variant
- Evaluation uses the official per-problem iteration counts and the same gradient-norm quantities plotted by the scripts
- Main metric: `final_gradient_norm`, the mean of the two official final gradient norms

## Interface Notes
- `init_state(...)` must preserve the provided starting point in `state["z"]`
- `step(...)` should implement one official-style iteration of the chosen method
- The oracle exposes deterministic gradients and fixed-scale Gaussian update noise so the update equations can match the MATLAB scripts directly
- `get_hyperparameters(...)` should return the per-problem constants used by the method

## Metrics
- Lower is better
- The harness prints:
  - `STEP_METRICS problem=... iteration=... gradient_norm=...`
  - `RUN_METRICS problem=... final_gradient_norm=... auc_log_iteration_log_grad=...`
  - `FINAL_METRICS final_gradient_norm=...`

## Read-Only References
- `RAIN/README.md`
- `RAIN/src/bilinear_func/exp_gnorm.m`
- `RAIN/src/delta_func/exp_gnorm.m`

These are the primary references. The task now follows those scripts directly rather than the earlier MLS-Bench-specific generalized variant.
