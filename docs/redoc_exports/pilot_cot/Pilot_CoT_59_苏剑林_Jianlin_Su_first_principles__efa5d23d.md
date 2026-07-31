# Pilot CoT 59 — 苏剑林 Jianlin Su · first_principles

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例3：想让注意力"训短测长" → 从"熵不变性"推出 log n 缩放因子

- route: first_principles  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **204** · assistant(cot) **1178** · whole **1439**

- fact-check changes: 4

## user (setup)

```plaintext
It's late 2021. Transformers scale logits by 1/√d before softmax; the usual justification is variance control at initialization — keep the dot products from saturating softmax early in training. That story explains training stability. It says nothing about a different, increasingly painful problem: a model trained on sequences of length ~512 degrades badly when you ask it to attend over 1024, 2048, or more tokens at inference. Length extrapolation has no clean standard solution yet. People patch it by grid-searching a temperature or scale, or by fiddling with the position encoding, but these fixes don't transfer across lengths or models. Suppose you refuse to treat the scale factor as a knob to tune. Can you derive, from a first principle about what SHOULD stay invariant as the sequence grows, what the correct length-dependent scale factor must be — as a clean closed form, ideally with no free hyperparameter? Propose the idea and the derivation.
```

## assistant (cot)

```plaintext
Let me refuse the variance story for a moment. The √d that everyone quotes is derived from a statement about training-time stability. But my question is a generalization question — train short, test long — and there is no reason a variance argument at initialization should answer a generalization question at inference. So I should ask: what is the object that actually changes when the sequence grows, and what quantity about it OUGHT to stay fixed?

The attention weights a_ij for a fixed query i form a probability distribution over the j tokens. That's the honest way to see softmax — it's a distribution, not just a normalization. Now the natural invariant on a distribution is its entropy. And entropy has exactly the meaning I care about: entropy 0 means the query has locked onto a single token (perfect focus); entropy log n means uniform smearing over everything. Here is the mechanism of the failure, stated as a sentence: when I append new tokens, plain softmax has to spread some probability mass onto them, the distribution flattens, entropy climbs, focus is lost. The already-present tokens no longer attend to the old tokens the way they did during training. So the principle almost writes itself: **the attention distribution's entropy should be invariant to sequence length n.** If I can hold entropy fixed, an added token doesn't get to dilute the existing focus. This is the target — not a trick, a constraint.

Now I make the scale a general factor λ in front of q·k and I compute the entropy. The entropy of a softmax is a standard logsumexp identity: H = log Σ_j e^{λ q·k} − (weighted mean of λ q·k). I want to see where n enters. Each sum over j is over n terms; approximate Σ_j e^{λ q·k} ≈ n · E_j[e^{λ q·k}]. The moment I do that, a clean additive log n falls out of the log-sum term. That log n IS the entropy inflation caused by longer sequences. Everything else is O(1) in n. So the whole problem has collapsed to: choose λ(n) so that λ's contribution cancels that log n.

I have to be honest that this is a semi-quantitative estimate — exact entropy invariance is not achievable, so I'll state my assumptions plainly. The residual O(1) term is an expectation over the geometry of q and k. With LayerNorm plus near-orthogonal projections, I can take ‖q‖=‖k‖=√d and k roughly uniform on the sphere of radius √d, so q·k = d cosθ and the expectation reduces to an integral over the angle θ. A Laplace-type approximation — the same collapse-to-the-dominant-point trick I've leaned on before for entropy-style estimates — turns that θ-expectation into a term roughly linear in λd. So H ≈ log n − c·λd + O(1) for some constant c. Setting log n − c λd = 0 gives λ ∝ log n / d. I don't trust the exact constant c coming out of a rough geometric assumption, so I fold it into a hyperparameter κ: λ = κ log n / d.

A free κ offends me. So calibrate it away by demanding that at the mainstream training length n = 512 the whole thing must reduce to ordinary √d attention: (κ log 512)/d = 1/√d fixes κ = √d/log 512. Substituting back, the scale is simply (1/√d)·log_512(n). No free hyperparameter: it equals 1/√d exactly at 512, grows for longer sequences, shrinks for shorter. That's the elegant closed form I was after — a log n correction, not a tuned scale.

How do I keep myself honest given the approximations? A controlled MLM setup: train at one short length, then evaluate accuracy across increasing test lengths against the plain-√d baseline. The decision rule I set in advance: at the training length the two should be essentially tied (otherwise I've merely retuned, not fixed extrapolation); the entropy-invariant scale is only worth keeping if it stays tied at training length AND holds up better as length grows. If the log n machinery doesn't buy that separation over plain scaling, I drop it on parsimony grounds. And I note a real caveat that follows straight from the form: if someone used a natural log with no length calibration, log n / √d grows without bound at large n and shrinks the logits' spread — such a variant would only look fine on very short sequences that never expose the problem. Calibration to 512 is not cosmetic; it's what keeps the invariant usable at scale.

**Core idea:** Treat attention weights as a probability distribution and demand its entropy stay invariant to sequence length; a logsumexp+Laplace derivation isolates the length-induced log n entropy inflation, and cancelling it forces the scale factor to be λ ∝ log n / d, calibrated to (1/√d)·log_512(n) so it reduces to standard attention at 512 with no free hyperparameter.

**Non-trivial crux:** The derivation is only semi-quantitative (it relies on the n·E approximation and geometric assumptions, so exact entropy invariance is unattainable) — and the log n scale MUST be length-calibrated, otherwise the factor grows unboundedly at large n and over-shrinks the logits, a failure hidden if one only tests very short sequences.
```

## fact-check changes

```plaintext
- (a) Rewrote the narrated evaluation so the winning outcome ('clearly wins as length grows') is framed as the pre-set decision rule/hypothesis to be tested, not an asserted confirmed result.
- (d) Softened the vanishing-gradient claim: an uncalibrated log n/√d scale over-shrinks logits (flattens the distribution) rather than literally 'killing gradients'; corrected in both body and crux for accuracy.
- (c) Toned down the Laplace step ('term linear in λd' -> 'roughly linear') to mark it as an approximation rather than an exact result, consistent with the stated semi-quantitative nature.
- Minor: reworded 'this collapse-to-the-dominant-point trick I've used before for estimating dimensions from an entropy principle' to a less specific, unembellished reference to keep the claim within verifiable ground.
```