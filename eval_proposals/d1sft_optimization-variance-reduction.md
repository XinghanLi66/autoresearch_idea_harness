**Mocking:** World-class AI researcher

Let me start from the mechanism I already trust and ask what must change to make it more efficient.

I know SVRG (Stochastic Variance Reduced Gradient) is a well-established variance-reduction method. The standard SVRG recipe is: periodically compute a full gradient at a fixed reference point (the 'snapshot'), then use that full gradient as a 'control variate' to correct the stochastic gradient. The variance of the corrected gradient becomes bounded by a constant, not by 1/b, which is the key to linear convergence.

But the standard SVRG reference point is a fixed point in the training set. That is exactly the part I want to change. Why fix it? Because a fixed reference point is expensive and only helps near the optimum, while variance reduction is most valuable early in training when the gradient is noisy. I want a reference point that tracks the current optimum, so the correction stays relevant throughout optimization.

Here's the leap: swap the fixed reference point for a *moving* one. Let the reference point be the latest parameter state (the 'momentum' point from a previous step). That is SVRG with a recursive reference point. This cuts against the grain of the literature, where 'reference points' are usually fixed. The intuition is that a reference point that moves with the optimum should give a better correction than one that is frozen.

Now, what must be true for this to work? The correction is valid only if the reference point is *consistent* with the current optimum. But the momentum point is the *previous* optimum, not the current one. So the correction is biased toward the past optimum, not the current one. This is exactly the bias-variance tradeoff I want to resolve.

Here's how I resolve it, and this is the non-trivial crux: instead of using the raw momentum point as the reference, *project* the momentum point onto the current optimum before using it as a control variate. The projection is a cheap in-place operation (a single dot product and subtraction), and it makes the reference point consistent with the current optimum. The correction then becomes unbiased and still low-variance. I must be precise about the projection direction: it should be the current gradient, not a fixed axis. If I used a fixed projection direction the bias could grow with the number of steps.

Why is this worth doing? Because the projection is essentially free (it's a scalar operation per parameter), so it doesn't cost much to add. It fixes the bias of the moving-reference point while preserving the low-variance property of SVRG. The same idea can be applied to any variance-reduction method that uses a reference point, so it's a general mechanism.

One more thing that must be true: the reference point must be updated *after* the stochastic gradient is taken. If I used the reference point before updating it, the momentum point would not reflect the current optimum. So the update order is critical.

Finally, I must be honest about the limits. This projection idea is an *improvement* on a moving-reference SVRG, not a replacement for the full method. The full SVRG (fixed reference) is still the gold standard for strongly convex problems, and the recursive version has a slight bias that the projection removes. The recursive form should be preferred only when the fixed reference is too expensive to compute (like in large-scale settings).

**Core idea:** Replace SVRG's fixed reference point with a moving one (the previous momentum point) and project it onto the current optimum before using it as a control variate, so the reference remains unbiased and consistent while retaining low-variance.

**Non-trivial crux:** The reference point is unbiased only if it is projected onto the current optimum along the current gradient; a fixed projection direction would cause the bias to grow with the number of steps, so the projection must be along the current gradient.