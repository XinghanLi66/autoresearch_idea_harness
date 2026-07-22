# Subgroup Calibration Under Distribution Shift

## Research Question
Design a post-hoc calibration method that remains reliable when subgroup composition shifts between calibration and test time.

## Background
Many calibration methods look good on average but fail on protected or operational subgroups once the test distribution shifts. This task isolates that failure mode. The fixed pipeline trains a tabular classifier, then applies a user-defined calibration mapping on held-out calibration data before evaluation on shifted test data.

Classical baselines include:
- **Temperature scaling**: one global temperature for all samples
- **Isotonic regression**: non-parametric monotone calibration
- **Beta calibration**: a richer parametric mapping on probabilities
- **Group-wise temperature scaling**: separate temperatures per subgroup

## Task
Modify the `CalibrationMethod` class in `custom_subgroup_calibration.py`. The fixed code loads data, creates a shifted split, trains the base classifier, and computes metrics. Your method only controls the post-hoc calibration mapping.

```python
class CalibrationMethod:
    def fit(self, probs, labels, groups=None):
        ...

    def predict_proba(self, probs, groups=None):
        ...
```

Inputs are positive-class probabilities from the base classifier. `groups` contains subgroup IDs when available and may be ignored by group-agnostic methods.

## Evaluation
This benchmark now uses cached high-stakes tabular datasets from AIF360 rather than lightweight sklearn proxies:

- **Adult**: Census income prediction, with sex/race subgroup attributes.
- **COMPAS**: ProPublica recidivism risk data, with race/sex subgroup attributes.
- **Law School GPA**: law-school admissions/outcome data, binarized around median first-year GPA, with race/gender subgroup attributes.

For each dataset, the split is intentionally shifted:
- a domain score determines the held-out test tail
- subgroup labels come from protected attributes exposed by the dataset loaders
- calibration is fit on the source region and evaluated on the shifted region

## Metrics
Lower is better for:
- **worst_group_ece**
- **brier**
- **max_subgroup_gap**

Higher is better for:
- **subgroup_auroc**

## Notes
- The task is deliberately low compute and should run with a small tabular classifier.
- Package-level data preparation for `scikit-learn` downloads the AIF360 datasets during setup so compute-node runs remain offline.
