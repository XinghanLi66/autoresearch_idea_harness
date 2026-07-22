# Probability Calibration Method Design

## Research Question
Design a novel post-hoc probability calibration method that maps uncalibrated classifier outputs to well-calibrated probabilities, minimizing the gap between predicted confidence and actual accuracy.

## Background
Modern classifiers (neural networks, gradient boosting, random forests) often produce overconfident or poorly calibrated probability estimates. A well-calibrated model should satisfy: among all predictions where the model outputs probability p for a class, the fraction that are actually correct should be approximately p.

Classic calibration methods include Platt scaling (logistic regression on logits), isotonic regression (non-parametric monotonic mapping), and histogram binning (piecewise constant). Recent advances include temperature scaling, beta calibration, and spline-based methods. Each has trade-offs in flexibility, data efficiency, and monotonicity preservation.

## Task
Implement the `CalibrationMethod` class in `custom_calibration.py`. The class has two methods:
- `fit(probs, labels)`: Learn a calibration mapping from uncalibrated probabilities and ground-truth labels on a held-out calibration set.
- `predict_proba(probs)`: Apply the learned mapping to produce calibrated probabilities.

For binary classification, probabilities are 1-D (positive class only). For multiclass, they are 2-D arrays where rows sum to 1.

## Interface
```python
class CalibrationMethod(BaseEstimator):
    def fit(self, probs, labels):
        # probs: (n,) for binary, (n, C) for multiclass
        # labels: (n,) integer class labels
        return self

    def predict_proba(self, probs):
        # Returns calibrated probabilities, same shape as input
        return calibrated_probs
```

Available imports in the template: numpy, scipy (optimize, interpolate, special), sklearn (various).

## Evaluation
The method is evaluated on 4 classifier-dataset combinations spanning binary and multiclass settings:
- Random Forest on MNIST (10-class)
- MLP on Fashion-MNIST (10-class)
- GBM on Madelon (binary)
- SVM on Breast Cancer (binary)

Metrics (all lower is better):
- **ECE** (Expected Calibration Error): Weighted average of |accuracy - confidence| across probability bins
- **Brier Score**: Mean squared error between predicted probabilities and one-hot labels
- **NLL** (Negative Log-Likelihood): Cross-entropy between predicted probabilities and true labels

