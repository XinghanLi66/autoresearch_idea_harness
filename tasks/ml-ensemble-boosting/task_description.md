# Ensemble Boosting Strategy Design

## Research Question
Design a novel sample weighting and update strategy for gradient boosting that improves over standard methods (AdaBoost, gradient boosting, XGBoost-style Newton step) across classification and regression tasks.

## Background
Gradient boosting builds ensembles of weak learners (decision trees) sequentially, where each new learner corrects errors made by the ensemble so far. The key design choices that differentiate boosting algorithms are:

- **Pseudo-target computation**: What does each new weak learner try to predict? Options include the original labels (AdaBoost), negative gradients of the loss (gradient boosting), or Newton-step targets using second-order information (XGBoost).
- **Learner weighting**: How much influence does each weak learner get? Computed from weighted error (AdaBoost), fixed at 1.0 with learning rate shrinkage (gradient boosting), or via line search / Newton optimization (XGBoost).
- **Sample reweighting**: How does the distribution over training samples shift between rounds? Exponential reweighting of misclassified samples (AdaBoost) vs. uniform weights with pseudo-residual fitting (gradient methods).

These design choices interact with each other and with the loss landscape. There is room for novel strategies that combine ideas from different approaches, use adaptive schedules, or exploit problem structure.

## What You Can Modify
The `BoostingStrategy` class (lines 147-256) in `custom_boosting.py`. This class has four methods:

- `init_weights(n_samples)` -- Initialize sample weights (should sum to 1)
- `compute_targets(y, current_predictions, sample_weights, round_idx)` -- Compute pseudo-targets for the next weak learner
- `compute_learner_weight(learner, X, y, pseudo_targets, sample_weights, round_idx)` -- Compute alpha for the just-fitted learner
- `update_weights(sample_weights, learner, X, y, pseudo_targets, alpha, round_idx)` -- Update sample weights for the next round

You have access to: true labels, current ensemble predictions, sample weights, the fitted learner (can call `learner.predict(X)`), round index, and the config dict with dataset metadata.

Available imports in the FIXED section: `numpy`, `sklearn.tree`, `sklearn.metrics`, `sklearn.datasets`, `sklearn.model_selection`.

## Evaluation
- **Metrics**:
  - Classification (breast_cancer): `test_accuracy` (higher is better)
  - Regression (diabetes, california_housing): `test_rmse` (lower is better)
- **Benchmarks** (3 datasets):
  - Breast Cancer Wisconsin (classification, 569 samples, 30 features)
  - Diabetes (regression, 442 samples, 10 features)
  - California Housing (regression, 20640 samples, 8 features)
- **Training**: 200 boosting rounds, max_depth=3 trees, learning_rate=0.1, 80/20 train-test split
- **Base learner**: Decision tree (fixed, not editable)

