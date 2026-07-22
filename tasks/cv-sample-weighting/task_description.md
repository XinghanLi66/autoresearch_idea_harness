# CV Sample Reweighting Strategy Design

## Research Question
Design a novel sample reweighting strategy for class-imbalanced image classification that improves test accuracy on long-tail distributed datasets across different architectures and imbalance ratios.

## Background
Real-world datasets often exhibit long-tail class distributions where a few head classes dominate while many tail classes have very few samples. Standard training with uniform loss weighting biases the model toward frequent classes, degrading performance on rare ones. Sample reweighting assigns per-class weights to the cross-entropy loss to counteract this imbalance. Classic approaches include:

- **Inverse frequency**: weight[c] = total / (C * count[c]) — directly compensates for imbalance
- **Effective number** (Cui et al., CVPR 2019): models data overlap using E_n = (1 - beta^n) / (1 - beta)
- **Square-root inverse**: weight[c] = 1/sqrt(count[c]) — a gentler smoothed variant

These methods define different mappings from class frequency to loss weight and may behave differently across datasets and imbalance regimes.

## What You Can Modify
The `compute_class_weights(class_counts, num_classes, config)` function (lines 164-195) in `custom_weighting.py`. This function receives per-class sample counts and must return a weight tensor for `nn.CrossEntropyLoss(weight=...)`.

You can modify:
- The functional form mapping class counts to weights (inverse, power-law, logarithmic, piecewise, etc.)
- Use of the `config` dict: `imbalance_ratio`, `dataset`, `arch`, `total_samples`
- Normalization strategy (sum to C, sum to 1, unnormalized, etc.)
- Any pure-computation logic (no access to training data or model parameters)

## Evaluation
- **Metric**: Best test accuracy (%, higher is better) on the balanced test set
- **Benchmarks** (all long-tail imbalanced):
  - ResNet-32 on CIFAR-10-LT (imbalance ratio = 100, 10 classes)
  - ResNet-32 on CIFAR-100-LT (imbalance ratio = 100, 100 classes)
  - VGG-16-BN on CIFAR-100-LT (imbalance ratio = 50, 100 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip
