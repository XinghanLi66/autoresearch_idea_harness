# Differentially Private SGD: Privacy-Utility Optimization

## Research Question
Design an improved DP-SGD variant that achieves better privacy-utility tradeoff — higher test accuracy under the same (epsilon, delta)-differential privacy budget.

## Background
Differentially Private Stochastic Gradient Descent (DP-SGD) [Abadi et al., 2016] enables training deep learning models with formal privacy guarantees. The core mechanism has two steps: (1) clip each per-sample gradient to a fixed norm C, and (2) add calibrated Gaussian noise proportional to C. The noise level is determined by the desired privacy budget (epsilon, delta).

The standard approach uses a fixed clipping threshold and constant noise throughout training, which is suboptimal: gradient magnitudes change during training, the fixed threshold either over-clips (losing signal) or under-clips (adding excess noise), and the uniform noise allocation ignores the varying informativeness of gradients across training stages.

## Task
Modify the `DPMechanism` class in `custom_dpsgd.py`. Your mechanism receives per-sample gradients and must return aggregated noised gradients. You control the gradient clipping strategy, noise calibration, and any per-step adaptations.

## Interface
```python
class DPMechanism:
    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        ...

    def clip_and_noise(self, per_sample_grads, step, epoch) -> list[Tensor]:
        # per_sample_grads: list of tensors [B, *param_shape]
        # Returns: list of noised gradients [*param_shape]
        ...

    def get_effective_sigma(self, step, epoch) -> float:
        # Returns current noise multiplier for privacy accounting
        ...
```

## Constraints
- The total privacy budget (target_epsilon, target_delta) is FIXED and checked externally.
- The model architecture, data pipeline, optimizer, and training loop are FIXED.
- Focus on algorithmic innovation in the DP mechanism: clipping strategies, noise schedules, gradient processing.
- Available imports: `torch`, `math`, `numpy` (via the FIXED section), `scipy.optimize`.

## Evaluation
Trained and evaluated on three datasets at epsilon=3.0, delta=1e-5:
- **MNIST** (28x28 grayscale digits, 10 classes)
- **Fashion-MNIST** (28x28 grayscale clothing, 10 classes)
- **CIFAR-10** (32x32 color images, 10 classes)

Metric: **test accuracy** (higher is better) under the same privacy budget.

