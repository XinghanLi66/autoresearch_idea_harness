# Poison-Robust Learning under Label-Flip Poisoning

## Research Question
How can we design a stronger loss function or sample-weighting rule that improves robustness to poisoned training labels without changing the model, optimizer, or data pipeline?

## Background
A fraction of poisoned (label-flipped) training labels can disproportionately distort model decision boundaries. Robust learning methods typically modify the objective to downweight suspicious samples or reduce memorization of corrupted targets. This task uses research-scale models (ResNet-20, VGG-16-BN, MobileNetV2) trained on full datasets with standard SGD + CosineAnnealing for 100 epochs.

## Task
Implement a better poison-robust objective in `bench/poison/custom_robust_loss.py`. The fixed harness injects random label-flip corruption into the training set, trains with your loss, and evaluates on a clean test set.

Your method should improve clean test accuracy under poisoning while reducing how much the model memorizes poisoned labels. The approach must be modular and transferable across architectures and datasets.

## Editable Interface
You must implement:

```python
class RobustLoss:
    def compute_loss(self, logits, labels, epoch):
        ...
```

- `logits`: current minibatch model outputs
- `labels`: possibly poisoned labels (label-flip: `(original + 1) % num_classes`)
- `epoch`: current training epoch (0-indexed)
- Return value: scalar loss tensor

The corruption process, model architectures, optimizer, and training schedule are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10-labelflip`: ResNet-20 on CIFAR-10, 10% label-flip poison
- `vgg16bn-cifar100-labelflip`: VGG-16-BN on CIFAR-100, 10% label-flip poison
- `mobilenetv2-fmnist-labelflip`: MobileNetV2 on FashionMNIST, 15% label-flip poison

Reported metrics:

- `test_acc`: accuracy on clean test set
- `poison_fit`: fraction of poisoned samples where model predicts the poisoned (wrong) label
- `robust_score = (test_acc + (1 - poison_fit)) / 2`

Primary metric: `robust_score` (higher is better).

## Baselines
- `cross_entropy`: standard ERM on poisoned labels
- `generalized_ce`: generalized cross-entropy for noisy labels
- `symmetric_ce`: CE plus reverse-CE penalty
- `bootstrap`: target interpolation with model predictions
