# Machine Unlearning via Targeted Update Rules

## Research Question
How can we design a stronger unlearning update rule that removes information about a forget set while retaining as much utility as possible on the retained data?

## Background
Machine unlearning methods approximate the effect of retraining without the deleted data. The central tradeoff is clear: aggressive forgetting reduces utility, while conservative updates leave measurable traces of the forgotten examples.

The harness pretrains a standard vision model (ResNet-20, VGG-16-BN, or MobileNetV2) on the full training set for 80 epochs using SGD with cosine annealing. After pretraining, a single class is designated as the forget set. Your unlearning method then runs for 20 epochs, receiving both retain-set and forget-set minibatches each step, with an Adam optimizer (lr=0.001).

## Task
Implement a better unlearning rule in `bench/unlearning/custom_unlearning.py`. The fixed harness trains an initial model, defines a forget split, and then applies your update rule for a fixed number of unlearning steps using retain and forget minibatches.

Your method should lower forget-set memorization while preserving retained-task accuracy.

## Editable Interface
You must implement:

```python
class UnlearningMethod:
    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        ...
```

- `retain_batch`: `(images, labels)` tuple from retained data (already on device)
- `forget_batch`: `(images, labels)` tuple from the forget set (already on device)
- `optimizer`: fixed Adam optimizer instance (lr=0.001)
- Return value: dict with at least `loss`

The architecture, initial training, forget split, and evaluation probes are fixed.

## Evaluation
Benchmarks:

- `resnet20-cifar10-class0`: ResNet-20 on CIFAR-10, forgetting class 0
- `vgg16bn-cifar100-class0`: VGG-16-BN on CIFAR-100, forgetting class 0
- `mobilenetv2-fmnist-class0`: MobileNetV2 on FashionMNIST, forgetting class 0

Reported metrics:

- `retain_acc`: accuracy on non-forget test data
- `forget_acc`: accuracy on forget-class test data (lower is better)
- `forget_mia_auc`: membership inference attack AUC on forget set (lower is better)
- `unlearn_score`: (retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3

Primary metric: `unlearn_score` (higher is better).

## Baselines
- `retain_finetune`: continue training only on retained data
- `negative_gradient`: ascend forget loss and descend retain loss
- `bad_teacher`: distillation-style forgetting baseline
- `scrub`: stronger representation-scrubbing baseline
