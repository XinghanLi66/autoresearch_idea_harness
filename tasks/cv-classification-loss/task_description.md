# CV Classification Loss Function Design

## Research Question
Design a novel classification loss function for deep convolutional neural networks that improves test accuracy across different architectures and datasets.

## Background
The cross-entropy loss is the standard training objective for classification networks, but it has known limitations: it treats all misclassifications equally, assigns high confidence to correct predictions without margin, and does not adapt to training dynamics. Researchers have proposed alternatives:

- **Label Smoothing** (Szegedy et al., 2016): Softens hard targets to prevent overconfidence, CE with targets = (1-eps)*one_hot + eps/C
- **Focal Loss** (Lin et al., ICCV 2017): Down-weights easy examples via (1-pt)^gamma modulation
- **PolyLoss** (Leng et al., ICLR 2022): Extends CE with polynomial correction terms, CE + eps*(1-pt)

However, these methods are either static or address only specific failure modes. There is room to design loss functions that combine multiple insights: confidence calibration, curriculum-style epoch adaptation, class-count awareness, or learned temperature scaling.

## What You Can Modify
The `compute_loss(logits, targets, config)` function (lines 165-185) in `custom_loss.py`. This function receives raw logits, integer targets, and a config dict, and must return a differentiable scalar loss.

You can modify:
- The loss formulation (cross-entropy variants, margin losses, etc.)
- Confidence-based reweighting schemes
- Epoch-dependent curriculum strategies using `config['epoch']` and `config['total_epochs']`
- Class-count-dependent behavior using `config['num_classes']`
- Temperature or logit scaling
- Auxiliary regularization terms (entropy, logit penalties, etc.)

The `config` dict provides: `num_classes` (int), `epoch` (int, 0-indexed), `total_epochs` (int).

**Important**: The evaluation loss (for test_loss reporting) always uses standard cross-entropy. Your loss function only affects training.

## Evaluation
- **Metric**: Best test accuracy (%, higher is better)
- **Architectures & datasets**:
  - ResNet-56 on CIFAR-100 (deep residual, 100 classes)
  - VGG-16-BN on CIFAR-100 (deep non-residual with BatchNorm, 100 classes)
  - MobileNetV2 on FashionMNIST (lightweight inverted-residual, 10 classes) — *hidden, evaluated on final submission only*
- **Training**: SGD (lr=0.1, momentum=0.9, wd=5e-4), cosine annealing, 200 epochs
- **Data augmentation**: RandomCrop(32, pad=4) + RandomHorizontalFlip

