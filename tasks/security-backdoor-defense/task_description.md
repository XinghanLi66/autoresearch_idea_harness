# Backdoor Defense via Poisoned-Sample Scoring

## Research Question
How can we design a better poisoned-sample scoring rule that identifies backdoored training examples while preserving clean utility after filtering and retraining?

## Background
Backdoor attacks implant a trigger pattern into a subset of training examples and relabel them to an attacker-chosen target class. Standard models trained on this data retain high clean accuracy while also predicting the target label whenever the trigger appears. Many defenses try to identify suspicious points using feature statistics, confidence patterns, or clustering structure before retraining on the filtered set.

## Task
Implement a stronger backdoor defense in `bench/backdoor/custom_backdoor_defense.py`. The fixed harness will:

1. Construct a poisoned training set for a fixed trigger pattern (full dataset, no subsampling).
2. Train a victim model on the poisoned data for 100 epochs (SGD + CosineAnnealingLR).
3. Extract features from the penultimate layer and logits for the entire training set.
4. Call your defense to assign suspicion scores to training examples.
5. Remove the top `1.5 * epsilon` fraction of highest-scoring samples (over-estimate of the poison count, as recommended by Tran et al., 2018 Sec. 4.1) and retrain on the filtered set for 100 epochs.
6. Evaluate clean accuracy and attack success rate on triggered test inputs on the *retrained* model.

Your goal is to reduce backdoor ASR on the retrained model without sacrificing too much clean accuracy. The primary objective `defense_score = 0.5 * clean_acc + 0.5 * (1 - asr)` follows BackdoorBench's convention (see `defense/spectral.py` in https://github.com/SCLBD/BackdoorBench): filter-stage `poison_recall` is reported as a diagnostic but is **not** directly part of `defense_score`. A defense that under-ranks poison but still triggers the model to forget the backdoor during retrain can still succeed.

### Task-Design Notes

- **CIFAR-100 Blend uses 1% poison fraction**, not 5%, because at 5% the target class becomes 83% poisoned (2500 poisoned + 500 clean), which is a degenerate regime for per-class SVD/clustering defenses where the class mean is dominated by poison and centering makes the *clean* examples the outliers. Tran et al. (2018) assume poison rate within a class is well below 50%; with poison_fraction=0.01 CIFAR-100 target class is ~33% poisoned, matching the paper's setting.
- BadNets on CIFAR-10 (5%) and FashionMNIST (8%) keep the target class at ≤45% poisoned, which is within the operating regime of SVD-style defenses.

## Editable Interface
You must implement:

```python
class BackdoorDefense:
    def fit(self, features, labels, poison_fraction, **kwargs):
        ...

    def score_samples(self, features, logits):
        ...
```

- `features`: feature matrix of shape `(N, D)` from a fixed penultimate layer.
- `labels`: training labels after poisoning.
- `poison_fraction`: approximate fraction of poisoned points in the training data.
- `logits`: model logits of shape `(N, C)`.
- Return value from `score_samples`: 1D suspicion scores, higher means more suspicious.

The model architecture, poison injection process, filtering budget, and retraining schedule are fixed.

## Evaluation
Three benchmark settings are evaluated with research-scale training:

- `resnet20-cifar10-badnets` — ResNet-20 on full CIFAR-10, BadNets trigger, 5% poison fraction
- `vgg16bn-cifar100-blend` — VGG-16-BN on full CIFAR-100, Blend trigger, 1% poison fraction
- `mobilenetv2-fmnist-badnets` — MobileNetV2 on full FashionMNIST, BadNets trigger, 8% poison fraction

All models train for 100 epochs with SGD (lr=0.1, momentum=0.9, weight_decay=5e-4) and cosine annealing schedule.

Reported metrics:

- `clean_acc`: clean test accuracy after defense
- `asr`: attack success rate on trigger-patched test data
- `poison_recall`: fraction of true poisoned points removed by the defense
- `defense_score`: aggregate score used for ranking, higher is better

Primary objective: maximize `defense_score`.

## Baselines
- `confidence_filter`: ranks samples by target-label confidence
- `spectral_signature`: scores by leading singular-vector outlier magnitude
- `activation_clustering`: class-conditional cluster-distance heuristic
- `fine_pruning`: feature-magnitude pruning style heuristic used as a stronger reference
