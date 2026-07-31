# Pilot CoT 62 — 苏剑林 Jianlin Su · first_principles

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例4：Muon 火了不止复现 → 把它重新推导成"谱范数下的最速下降"

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **216** · assistant(cot) **1360** · whole **1633**

- fact-check changes: 8

## user (setup)

```plaintext
It's late 2024. A new optimizer called Muon is going viral on Twitter. It only acts on matrix-shaped parameters W (n×m), it keeps a momentum buffer, and then instead of using the momentum directly it applies a matrix 'orthogonalization' via a few fixed Newton–Schulz iterations with three magic coefficients before the update step. Empirically it seems to beat AdamW on the reported settings. The original release gives the recipe and a terse comment, but no mathematical derivation of why the orthogonalization is the right thing to do, no principled account of the magic coefficients, and no theory for what happens at very large scale. Most people are busy reproducing it and tuning hyperparameters.

You distrust magic numbers and refuse to stop at reproduction. Starting purely from what MUST be true about an optimizer update, derive from first principles what Muon actually is, why it should work, and what that understanding forces about learning-rate transfer and large-scale stability. Propose a genuinely novel, deduction-first framing before appealing to any experiment.
```

## assistant (cot)

```plaintext
Let me not touch the code first. The title question is the whole thing: what is the essential difference between a vector parameter and a matrix parameter, and why should an optimizer care? SGD and Adam update element-wise. That means they implicitly treat W as a bag of loose scalars — the geometry they respect is the Frobenius norm, ‖W‖_F, i.e. flatten everything and pretend it's one vector. But a matrix is a linear map. Its natural size is not the Frobenius norm; it's the spectral norm ‖W‖_2, the largest singular value, because that's what controls how much it can amplify an input. So my suspicion, before any derivation: Muon is not a random trick, it is what you get when you stop using the wrong norm. (I should note the spectral-norm-as-steepest-descent framing is my own reconstruction; the released Muon itself was motivated more operationally, via orthogonalized momentum, so I'm treating this as intuition I still have to prove, not the author's stated argument.)

Start from what an update MUST be. Linearize the loss: locally the decrease is Tr(Gᵀ ΔW), G the gradient. An optimizer should choose ΔW to make this as negative as possible while staying 'small' — otherwise you just take an infinite step. So the honest formulation is a constrained steepest descent: minimize (‖ΔW‖²)/(2η) + Tr(GᵀΔW). Write ΔW = (step) × (direction Φ) with the direction constrained to unit norm. Everything now hinges on *which norm* you put on Φ. This is the pivot: choosing an optimizer = choosing the norm that measures 'small'. SGD is the F-norm choice. What is the spectral-norm choice?

So solve it. Constrain ‖Φ‖_2 = 1 and maximize Tr(GᵀΦ). By duality, the spectral norm's dual is the nuclear norm, so Tr(GᵀΦ) ≤ Σ_i σ_i(G), and equality holds exactly when Φ aligns each singular direction with unit weight — i.e. Φ = U Vᵀ from the SVD G = U Σ Vᵀ. That is precisely: keep the orthogonal factors, throw away the singular values, push every nonzero singular value to 1. That's msign(G) = (GGᵀ)^{-1/2} G (understanding this as the reduced/pseudo-inverse form when G is rank-deficient). And the 1×1 case is x·(x²)^{-1/2} = sign(x). So the whole thing snaps into place deductively: Muon is the matrix generalization of sign-momentum, and it is exactly steepest descent under the spectral-norm constraint. SGD = steepest descent in F-norm; Muon = steepest descent in spectral norm. The spectral constraint is a more faithful measure of a linear map's step size, which is why I'd argue it should have more headroom — a geometric intuition to be tested, not something already 'won' in an experiment.

Now the magic numbers. The Newton–Schulz iteration is just a cheap way to compute msign without an SVD: a matrix polynomial that maps every singular value toward 1, i.e. it approximates the projection onto sign of the singular values after normalization. I will NOT trust the three coefficients on faith. The Taylor form near t=1 is a starting sanity check; but the honest move is to sample the actual singular-value distribution of random matrices and fit the coefficients to push that distribution to 1 over the fixed number of iterations. Decision rule: if my fitted coefficients don't visibly tighten the spectrum toward 1 better (or at least equal) on the observed distribution, I keep the simpler/known ones — I don't add complexity for nothing.

Two consequences follow, again by deduction, not tuning. First, learning-rate transfer. Adam's per-element update has an RMS around a fixed small scale; msign(M) has an analytic RMS of √(r/(nm)) where r is the rank. If I renormalize any update to that same RMS scale, the effective step becomes comparable across optimizers, so I can hope to reuse Adam's learning rate — the scale factor collapses to something like 0.2·√max(n,m). That's a closed-form alignment to check, not a grid search.

Second, stability at scale. The one invariant this whole framework says must be respected is the spectral norm. If nothing bounds it, ‖W‖_2 can drift upward as we scale — that's the quantity that should stay controlled. Weight decay is therefore not decoration: with it, one can bound ‖W_t‖_2 (roughly by max(‖W_0‖_2, η_eff/λ) up to the update's spectral norm). And if a specific symptom of spectral blow-up appears at very large scale (e.g. attention logits growing), the fix is dictated by the same invariant: clip or constrain the Q/K weights so the logit stays bounded. The point is that one mathematical thread — 'measure steps in the spectral norm, keep the spectral norm bounded' — should run from the definition of msign all the way to a stability patch. What I'd verify: match Adam's LR under the RMS alignment and check whether loss tracks or beats AdamW at equal compute; and check whether with weight decay the empirical ‖W‖_2 stays bounded while without it it shows the predicted upward drift.

**Core idea:** Formulate any optimizer update as steepest descent under a chosen norm; choosing the spectral norm (the true size of a linear map) instead of Frobenius forces the update to be msign(G) = UVᵀ — so Muon is exactly spectral-norm steepest descent, which then suggests a closed-form RMS-based learning-rate alignment and a spectral-norm-bounding weight decay for scalable stability.
**Non-trivial crux:** The whole thing turns on the norm dual: under a spectral-norm-1 direction constraint, the linearized descent is maximized by the orthogonal factor UVᵀ (singular values set to 1) because the nuclear norm is spectral's dual — and the *same* invariant, boundedness of ‖W‖_2, is what makes weight decay essential (not optional) and dictates the large-scale fix; the fitted Newton–Schulz coefficients are only a cheap approximation to msign and should be kept only if they measurably tighten the singular spectrum.
```

## fact-check changes

```plaintext
- (c) Added explicit note that the spectral-norm steepest-descent framing is Su's own reconstruction/intuition, not the original Muon author's stated motivation (which was orthogonalized momentum), rather than presenting it as the decisive contemporaneous argument.
- (c/a) Softened 'which is why it should have more headroom — not because it won an experiment... but geometrically correct' into a stated intuition to be tested.
- (a) Removed the implied confirmed-result phrasing in verification section; kept as things to check.
- (d) Corrected RMS formula to √(r/(nm)) and made the scale-factor concrete (~0.2·√max(n,m)) as a value to check, not asserted.
- (d) Fixed the weight-decay bound statement to include the effective-learning-rate/update dependence rather than the over-clean max(‖W_0‖_2, 1/λ).
- (d) Added pseudo-inverse caveat for msign = (GGᵀ)^{-1/2}G on rank-deficient G.
- (d) Generalized the attention-logit fix to a bounded-constraint statement rather than a specific fabricated √(τ/S_max) coefficient.
- (a) Rephrased Newton-Schulz target from 't^{-1/2}' to projecting singular values to their sign/toward 1 for accuracy.
```