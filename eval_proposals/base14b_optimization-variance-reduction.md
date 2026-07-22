Core idea: Combine a single epoch-wide full gradient snapshot with a momentum-based correction that dynamically adjusts stochastic gradients toward the full gradient, leveraging the snapshot as a stable reference point to reduce variance.

Non-trivial crux: The momentum correction must be scaled such that it maintains unbiasedness while exploiting the lower variance of the full gradient, requiring careful balancing of the contribution between the stochastic gradient and the full gradient snapshot.

**Explanation**:
1. **Initialization**: At the start of each epoch, compute the full gradient `g_0` using `compute_full_gradient` and store it as a reference.
2. **Momentum Correction**: For each mini-batch, compute the stochastic gradient `g_t`, then apply a correction:  
   `corrected_g = g_t - β * (g_t - g_0)`, where β is a hyperparameter. This pulls `g_t` toward `g_0`, reducing variance by aligning with the more stable full gradient.
3. **Update Rule**: Use `corrected_g` for parameter updates. The correction ensures unbiasedness since `E[corrected_g] = E[g_t] - β(E[g_t] - g_0) = (1 - β)E[g_t] + βg_0`, which remains an unbiased estimator of the true gradient if `g_0` is unbiased (it is).

This method leverages the full gradient once per epoch while maintaining computational efficiency, and the β parameter controls the trade-off between variance reduction and adherence to the stochastic gradient direction.