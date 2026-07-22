**Core idea:** Introduce a learnable, variable-specific imputation bias layer after the decoder that adds an offset to the prediction, calibrated per variable to correct systematic reconstruction errors.

**Non-trivial crux:** The bias must be masked during training (via `torch.masked_scatter`) to only activate on missing values, so it doesn't simply memorize the inputs, and its initialization should be small to avoid distorting the decoder's learned patterns.

**Reasoning path:**
1. Observe that standard transformer/linear baselines have a consistent per-variable imputation gap on masked regions (visible in the ETTh1 MAE/MSE table)
2. Identify the root cause: the decoder is forced to simultaneously predict missing values and reconstruct observed values, with different error dynamics
3. Decompose the error: the reconstruction error on observed values is often a consistent, variable-specific offset relative to the true values
4. Apply the bias-as-a-corrective-offset principle from classical machine learning (e.g. bias terms in linear models) to the deep setting
5. Add the bias as a learnable per-variable offset vector that is only applied to missing positions (masked training)
6. Argue for masked initialization: a zero-initialized bias would dominate the prediction, so initialize small and let it grow from the decoder's residual error
7. Test against the reference models: this small targeted addition should improve over DLinear/TimesNet/patchTST without increasing the parameter count much

**Non-trivial crux elaboration:** The mask is essential - without it, the bias term would overfit the observed values rather than correct the missing ones. And initialization matters: a large bias would make the network ignore the decoder's output entirely.