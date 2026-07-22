**Mocking:** World-class AI researcher

Let me start from a mechanism I actually trust and see what breaks when I try to apply it to stochastic gradient descent. SVRG and SARAH both use a *snapshot* of the model at a previous iteration as a control variate. The intuition is that if I keep a fixed point, the variance of the SGD step is reduced because the stochastic gradient is correlated with the full gradient at that fixed point. But the naive implementation of SVRG/SARAH is *not* a variance reduction algorithm in the sense of the research paper I want to beat -- it's a *periodic* full gradient plus a control variate. The periodicity is the weak link. When I'm stuck in a plateau, I keep waiting for the next full gradient computation and the control variate doesn't keep pace.

So I want to *recycle* the full gradient instead of recomputing it from scratch every time. The mechanism is: keep a *running* estimate of the full gradient, and update it *incrementally* with each stochastic step. This is exactly the idea behind SPIDER/SARAH-style variance reduction, but I'm going to implement it as a *recursive* update rather than a periodic one. The crux is that the recursive update must be *exact*, not an approximation.

Why does this cut against the consensus? Most variance reduction methods are presented as periodic full-gradient schemes. The claim I'm testing is that you can make variance reduction *non-periodic* by holding the full gradient in a buffer and updating it step-by-step. This is the same mechanism as the "stochastic recursive gradient estimation" in SPIDER and PAGE, but I'm going to name it SPIDER-style recursion and explain it as a single running buffer.

Now the honest part. The recursion has two places where it can break, and I have to be precise about what *must* be exact and what can be approximate.

First breaking point: the recursion formula. The correct update is:

    g_running = (b-1)/b * g_running + (1/b) * g_batch

But the naive implementation would write:

    g_running = g_running + g_batch - g_running_prev

This is equivalent algebraically, but it's *not* the same numerically because of floating-point precision. The exact form must be used because any in-place update that subtracts the previous state would accumulate rounding error over many steps. I'll keep the exact form and update it in-place only once per step, to minimize the number of operations.

Second breaking point: the variance argument. The standard SVRG/SARAH analysis assumes a *fixed* control variate (the snapshot). If I use the *running* gradient as the control variate, the analysis must be re-derived. This is the crux -- the running buffer is not a control variate, it's the *true* gradient. So the variance should be zero *before* I add the learning rate term, because the running gradient is an unbiased estimator of the full gradient. I need to verify this in the derivation, not assume it. If the derivation shows it, then the variance argument transfers; if not, the method is not a variance reduction method. My decision rule: keep the exact recursion and let the variance be a consequence of the derivation, not a premise.

Why does this cut against the consensus? Because the field has settled on periodic full-gradient schemes, and recursion is often seen as a second-order (momentum) method. I'm claiming it's a first-order variance reduction method, and I want to keep it simple and non-periodic.

How do I keep myself honest? I won't assert a rate; I'll run the three problems and look at the epoch curve. If the running-buffer method doesn't clearly beat periodic full gradient in convergence speed and doesn't reduce wall-clock time, I fall back to the simpler periodic scheme. And I should run a sanity check that the running buffer is an unbiased estimator of the full gradient -- if it drifts, the variance argument is wrong and I should revert.

**Core idea:** Recycle the full gradient with a SPIDER-style *exact* incremental update (g_running = (b-1)/b * g_running + (1/b) * g_batch) into a single running buffer, so variance reduction is non-periodic and the buffer is the true gradient (hence zero variance before the learning rate term).

**Non-trivial crux:** The exact form of the recursive update (not the in-place naive variant) is required to keep the running buffer unbiased and to transfer the SVRG/SARAH variance argument; the variance is a consequence of the derivation, not a premise.