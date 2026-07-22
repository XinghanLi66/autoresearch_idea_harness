# Variance Reduction for Stochastic Optimization

## Research Question
Design an improved variance reduction strategy for stochastic gradient descent on finite-sum optimization problems. Your method should accelerate convergence compared to vanilla mini-batch SGD by reducing the variance of gradient estimates.

## Background
Many machine learning problems take the form of finite-sum optimization:

    min_x  F(x) = (1/n) * sum_{i=1}^{n} f_i(x)

Standard SGD uses a stochastic gradient from a random mini-batch, which has variance proportional to 1/b (where b is the batch size). Variance reduction methods use auxiliary information (snapshots, recursive corrections, momentum) to reduce this variance, enabling faster convergence -- often achieving linear convergence rates for strongly convex problems where SGD only achieves sublinear rates.

Key methods in this area include SVRG (periodic full gradient + control variate), SARAH (recursive gradient correction), STORM (momentum-based online variance reduction), SPIDER, and PAGE.

## Task
Modify the `VarianceReductionOptimizer` class in `custom_vr.py` (lines 286-370). You must implement:

1. **`__init__(self, model, lr, l2_reg, loss_type, n_train, batch_size, device)`**: Initialize any state needed for variance reduction (snapshot parameters, running gradient estimates, buffers, etc.)
2. **`train_one_epoch(self, X_train, y_train)`**: Train for one epoch over the data, returning a dict with at least `'avg_loss'` (and optionally `'full_grad_count'` if you use full gradient computations)

The default implementation is vanilla mini-batch SGD. Your goal is to design a variance reduction mechanism that improves convergence.

## Interface

### Available helper functions (FIXED, use these for gradient computation):
```python
compute_full_gradient(model, X_train, y_train, loss_type, l2_reg, device)
# -> returns list of gradient tensors (one per parameter)

compute_stochastic_gradient(model, X_batch, y_batch, loss_type, l2_reg)
# -> returns list of gradient tensors for a mini-batch

compute_loss_on_batch(model, X_batch, y_batch, loss_type, l2_reg)
# -> returns scalar loss tensor
```

### Constraints
- You may call `compute_full_gradient` at most once per epoch
- Parameter updates must use `p.data.add_(...)` or similar in-place operations
- Must work across all three problems with the same code
- The learning rate (`self.lr`) and L2 regularization (`self.l2_reg`) are fixed
- Do not modify the model architecture, loss function, or evaluation code

## Evaluation
- **Problems**:
  - `logistic`: L2-regularized multinomial logistic regression on MNIST (convex, n=60K, 20 epochs)
  - `mlp`: 2-layer MLP on CIFAR-10 (non-convex, n=50K, 40 epochs)
  - `conditioned`: L2-regularized linear regression on synthetic ill-conditioned data (strongly convex, kappa=100, n=10K, 30 epochs)
- **Metrics**: `best_test_accuracy` (logistic, mlp; higher is better) and `best_test_mse` (conditioned; lower is better)
- All three problems run in parallel with shared compute

