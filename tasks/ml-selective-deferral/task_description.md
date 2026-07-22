# Selective Deferral Under Subgroup Shift

## Research Question
Design a practical selective prediction and deferral policy for high-stakes tabular decisions.

The task isolates one modular question: given a fixed base classifier, what acceptance / deferral rule best trades off selective risk, subgroup fairness, and overall discrimination?

## Background
Selective prediction systems should be able to say "I do not know" when the classifier is uncertain. In high-stakes settings, that deferral can be handed to a human reviewer or a slower backup process. The benchmark studies whether a policy can:

- keep selective risk low at a fixed target coverage,
- avoid concentrating deferrals on one subgroup,
- preserve AUROC as a confidence ranking signal, and
- remain simple enough to run offline on modest compute.

## Task
Modify the `SelectivePolicy` class in `custom_selective.py`. The rest of the pipeline is fixed: dataset loading, train / calibration / test splitting, base model training, and metric computation.

The policy receives calibration-time base-model probabilities and subgroup labels, then decides whether each test example should be accepted or deferred. You may implement a single global threshold, a learned deferral score, subgroup-specific thresholds, or any other compact policy that fits the interface.

## Evaluation
The benchmark runs on three cached high-stakes tabular datasets from AIF360:

- **Adult**: Census income prediction, with sex/race subgroup attributes.
- **COMPAS**: ProPublica recidivism risk data, with race/sex subgroup attributes.
- **Law School GPA**: law-school admissions/outcome data, binarized around the training-set median, with race/gender subgroup attributes.

Each dataset is split into train / calibration / test partitions. Subgroups come from protected attributes exposed by the dataset loaders so worst-group behavior is measured on semantically meaningful groups.

Metrics:

- `selective_risk_at80`: classification error on accepted examples at 80% target coverage
- `worst_group_selective_risk`: worst subgroup error on accepted examples
- `deferral_rate_gap`: max subgroup deferral rate minus min subgroup deferral rate
- `auroc`: AUROC of the acceptance score for predicting correctness

## Baselines
- `confidence_thresholding`: tune one confidence threshold to hit the target coverage
- `conformal_abstention`: split-conformal abstention with a coverage target
- `learned_deferral`: train a compact meta-model that predicts whether the base model will be correct
- `groupwise_thresholding`: subgroup-specific thresholds as a stronger reference baseline

## Practical Notes
Package-level data preparation for `scikit-learn` downloads these datasets during setup so compute-node runs remain offline.
