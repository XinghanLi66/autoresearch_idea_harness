# Meta-Learning: Inner-Loop Optimization Algorithm Design

## Research Question
Design and implement a novel inner-loop adaptation algorithm for gradient-based meta-learning. Your code goes in the `InnerLoopOptimizer` class in `custom_maml.py`. Reference implementations of MAML, Meta-SGD, and GBML from the learn2learn library are provided as read-only context.

## Background
Gradient-based meta-learning (MAML-style) learns a model initialization that can be quickly adapted to new tasks via a few gradient steps. The **inner loop** determines *how* parameters are updated during task-specific adaptation, while the **outer loop** optimizes the initialization (and any optimizer state) across tasks.

The simplest inner loop is vanilla SGD (MAML), but many improvements exist:
- **Per-parameter learning rates**: Meta-SGD learns a separate learning rate for each parameter
- **Selective adaptation**: ANIL adapts only the classification head, freezing the backbone
- **Preconditioning**: Meta-Curvature applies learned curvature matrices to gradients
- **Learned update rules**: Using neural networks or structured transforms to generate updates

Key design choices include:
- **What to adapt**: All parameters, head only, or a learned subset
- **How to scale gradients**: Fixed LR, per-parameter LR, preconditioned, or transformed
- **Momentum and memory**: Whether to incorporate information across inner-loop steps
- **Regularization**: Constraining adaptation to prevent overfitting on few-shot support sets

## Task
Modify the `InnerLoopOptimizer` class in `custom_maml.py` to implement your inner-loop adaptation algorithm. The class has three methods:
- `__init__(model, inner_lr)`: Initialize the optimizer and any learnable state
- `adapt(model, support_x, support_y, n_steps)`: Perform n_steps of adaptation on the support set
- `meta_parameters()`: Return any learnable parameters of the optimizer itself

## Interface
```python
class InnerLoopOptimizer:
    def __init__(self, model: nn.Module, inner_lr: float):
        # model is the base model (for inspecting parameter shapes)
        # inner_lr is the default learning rate
        # Create any learnable parameters here
        ...

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        # model is a CLONE — safe to modify in-place
        # Must use differentiable operations (torch.autograd.grad, NOT torch.optim)
        # Return the adapted model
        ...

    def meta_parameters(self) -> List[Tensor]:
        # Return learnable optimizer parameters for outer-loop optimization
        # Empty list if optimizer has no learnable state (e.g., vanilla MAML)
        ...
```

## Available Context
- `learn2learn/algorithms/maml.py`: MAML implementation (clone + adapt via differentiable SGD)
- `learn2learn/algorithms/meta_sgd.py`: Meta-SGD (per-parameter learned learning rates)
- `learn2learn/algorithms/gbml.py`: GBML base class (general gradient-based meta-learning with transforms)

## Evaluation
Trained and evaluated on three few-shot image classification benchmarks using CNN4 backbone:
- **miniImageNet 5-way 1-shot** (64 training / 16 val / 20 test classes, 600 examples each)
- **miniImageNet 5-way 5-shot** (same dataset, 5 support examples per class)
- **CIFAR-FS 5-way 5-shot** (100 classes from CIFAR-100, 5 support examples per class)

Metric: mean classification accuracy over 600 test episodes (higher is better). Meta-training runs for 60,000 iterations with 4 tasks per meta-batch. Inner loop uses 5 steps during training, 10 steps during evaluation.

