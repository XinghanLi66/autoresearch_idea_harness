Core idea: Introduce a momentum-based online variance reduction mechanism (à la STORM) that tracks a running average of the stochastic gradients and updates a bias-corrected momentum term on-the-fly, avoiding periodic full gradients while exploiting gradient smoothness to control variance.

Non-trivial crux: The momentum buffer must be warm-started and reset every epoch to prevent stale gradients from corrupting the variance-reduction estimate, and the bias correction term must be carefully scaled to the effective number of gradient steps since the last reset to maintain a bounded variance bound.

---

**Reasoning process:**

I'm going to start from the variance bound for a single stochastic step. Let g_t be the true gradient and \hat{g}_t be the unbiased stochastic estimator (mini-batch gradient). The expected variance of the update direction is:

E[||\hat{g}_t - g_t||^2] ≈ σ^2 / b

This is the 1/b variance that SGD suffers from. Variance reduction aims to make the effective variance decay with iteration rather than be fixed by batch size.

SVRG does this by a control variate: it stores a snapshot full gradient, then constructs a variance-reduced estimator

\hat{g}_t^VR = \hat{g}_t - (\hat{g}_t - \hat{g}_t^{snap})

This removes the part of the noise that is independent of the snapshot point. But it requires full gradients at fixed intervals, which is expensive and breaks the nice O(n) per-epoch complexity of online methods.

The STORM/SARAH-style approach is more elegant. Define a running momentum buffer v_t = β v_{t-1} + (1 - β) \hat{g}_t. Then define a "corrected" estimator

\hat{g}_t^{corrected} = (1 - β)^{-1} v_t

Because of the exponential decay, v_t is a weighted average over recent gradients, and the corrected estimator has variance that decays as 1 / (1 - β^t) rather than 1/b.

This is exactly the mechanism I want to borrow. I'll implement it by:
1. Adding a momentum buffer (one per parameter) that accumulates gradients with exponential weighting
2. Using the corrected estimator for the update
3. Resetting the buffer at the start of each epoch to ensure the weights are fresh

Why is this non-trivial? The momentum buffer is a running average. If I never reset it, the weighting gets skewed toward early noisy gradients. The variance bound only holds if β^t is small relative to the number of steps since the last reset. Resetting every epoch ensures that β^t ≈ β^batches_per_epoch, which is small for β < 1.

The bias correction factor (1 - β)^{-1} is essential to make the estimator unbiased. Without it, the corrected gradient has a multiplicative bias that could push the iterates away from the optimum.

Implementation notes:
- Initialize the momentum buffer to zero
- At the start of each epoch, reset all buffers to zero (not to a snapshot)
- Compute the corrected gradient as buffer / (1 - β) before applying the update
- Use a β close to 1 (say, 0.9 or 0.95) to get long averaging windows without resetting too often
- I'll need to store the per-parameter buffers as tensors on the same device as the parameters

This is a clean, lightweight replacement for periodic full gradients. It should give faster convergence than mini-batch SGD while maintaining the O(n) per-epoch budget.

---

**Code changes** (not full code, just the essential additions to the class):

1. In `__init__`:
```python
self.momentum_buffers = None  # To be initialized on first call
self.beta = 0.95  # Momentum coefficient
```

2. In `train_one_epoch`:
```python
# At start of epoch: reset momentum buffers
if self.momentum_buffers is None or reset_buffers:
    self.momentum_buffers = [torch.zeros_like(p) for p in self.model.parameters()]

# Inside the batch loop:
# Compute stochastic gradient
stochastic_grads = compute_stochastic_gradient(...)
# Update momentum buffer
for b, s in zip(self.momentum_buffers, stochastic_grads):
    b.data.mul_(self.beta).add_(s, alpha=1 - self.beta)
# Bias-corrected estimator
corrected_grads = [b / (1 - self.beta) for b in self.momentum_buffers]
# Update parameters
for p, g in zip(self.model.parameters(), corrected_grads):
    p.data.add_(g, alpha=-self.lr)
```

(If I'm being careful, I'd add a small epsilon to the denominator for numerical stability, e.g., `1 - self.beta + 1e-8`.)

This replaces the vanilla mini-batch update with a momentum-corrected variance-reduced update, and the momentum buffers are automatically reset every epoch.