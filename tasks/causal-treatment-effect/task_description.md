# Causal Treatment Effect Estimation

## Research Question
Design a novel estimator for Conditional Average Treatment Effects (CATE) from observational data that is accurate, robust to confounding, and generalizes across explicitly synthetic data-generating processes.

## Background
Estimating heterogeneous treatment effects -- how the causal effect of a treatment varies across individuals -- is a core problem in causal inference. Given observational data with covariates X, binary treatment T, and outcome Y, the goal is to estimate tau(x) = E[Y(1) - Y(0) | X=x], the conditional average treatment effect (CATE).

Key challenges include:
- **Confounding**: Treatment assignment depends on covariates, so naive comparisons are biased
- **Heterogeneity**: Treatment effects vary across the covariate space in complex, nonlinear ways
- **Model misspecification**: The true response surfaces may not match parametric assumptions
- **Double robustness**: Ideally, the estimator is consistent if either the outcome model or propensity model is correct

Classical approaches include S-Learner (single model), T-Learner (separate models), and IPW (propensity reweighting). Modern SOTA methods include Causal Forests (Athey & Wager, 2018), DR-Learner (Kennedy, 2023), and R-Learner (Nie & Wager, 2021), which use orthogonalization/debiasing to achieve better convergence rates.

## Task
Modify the `CATEEstimator` class in `custom_cate.py`. Your estimator must implement:
- `fit(X, T, Y) -> self`: Learn from observational data
- `predict(X) -> tau_hat`: Predict individual treatment effects

You have access to scikit-learn and numpy/scipy.

## Evaluation
Evaluated on three task-local synthetic benchmarks with known ground-truth treatment effects. These are inspired by common causal-inference benchmark families, but they are not the official IHDP, Jobs/LaLonde, or ACIC datasets/settings:
- **ihdp_synth**: IHDP-inspired synthetic observational DGP (n=747, p=25, nonlinear effects)
- **jobs_synth**: Jobs/LaLonde-inspired synthetic observational DGP (n=2000, p=10, economic outcomes)
- **acic_synth**: ACIC-inspired high-dimensional synthetic DGP (n=4000, p=50, complex confounding)

Metrics (lower is better for both):
- **PEHE**: Precision in Estimation of Heterogeneous Effects = sqrt(mean((tau_hat - tau_true)^2))
- **ATE error**: |mean(tau_hat) - ATE_true|

Each dataset is evaluated with 5-fold cross-fitting over 10 repetitions with different random seeds.
