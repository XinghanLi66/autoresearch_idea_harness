# Adversarial Training for Model Robustness

## Research Question
How to design better adversarial training methods to enhance model robustness against L_inf adversarial attacks?

## Background
Adversarial training is the most effective approach for improving neural network robustness against adversarial examples. The standard method (Madry et al., 2018) trains on PGD-generated adversarial examples using cross-entropy loss, but suffers from a trade-off between clean accuracy and robust accuracy. Advanced methods like TRADES and MART address this through different loss formulations that decouple the robustness objective from clean classification.

## Task
Implement a novel adversarial training method in `bench/custom_adv_train.py` by modifying the `AdversarialTrainer` class. Your method should improve robust accuracy against white-box L_inf attacks while maintaining reasonable clean accuracy.

## Interface
You must implement the `AdversarialTrainer` class with two methods:

- `__init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs)`: Initialize your trainer.
  - `model`: The neural network to train (nn.Module).
  - `eps`: L_inf perturbation budget (0.3 for MNIST, 8/255 for CIFAR).
  - `alpha`: Step size for inner PGD attack.
  - `attack_steps`: Number of PGD steps for adversarial example generation.
  - `num_classes`: Number of output classes (10 or 100).

- `train_step(self, images, labels, optimizer) -> dict`: Perform one training step.
  - `images`: Clean images, shape `(N, C, H, W)`, values in `[0, 1]`.
  - `labels`: Ground truth labels, shape `(N,)`.
  - `optimizer`: SGD optimizer (lr, momentum, weight_decay already configured).
  - Returns: dict with at least `'loss'` key (float).

The training loop, learning rate schedule (cosine annealing), model architecture, and data loading are handled externally. You only control the adversarial training procedure within each step.

## Evaluation
After training, models are evaluated on:
- **Clean accuracy**: Accuracy on unperturbed test images.
- **Robust accuracy (FGSM)**: Accuracy under 1-step FGSM attack.
- **Robust accuracy (PGD-50)**: Accuracy under 50-step PGD attack (**primary metric**).

Four scenarios (model + dataset):
- SmallCNN on MNIST (eps = 0.3)
- PreActResNet-18 on CIFAR-10 (eps = 8/255)
- VGG-11-BN on CIFAR-10 (eps = 8/255)
- PreActResNet-18 on CIFAR-100 (eps = 8/255)

Higher robust accuracy (PGD-50) across all scenarios is better.

## Baselines
- `standard`: Vanilla training (no adversarial examples). High clean accuracy, ~0% robust accuracy.
- `pgdat`: PGD Adversarial Training (Madry et al., 2018). Trains on PGD adversarial examples with CE loss.
- `trades`: TRADES (Zhang et al., 2019). Balances clean and robust accuracy via KL divergence regularization.
- `mart`: MART (Wang et al., 2020). Misclassification-aware regularization that focuses on hard examples.
- `awp`: AWP + TRADES (Wu et al., 2020). Adversarial weight perturbation on top of TRADES — current SOTA.

