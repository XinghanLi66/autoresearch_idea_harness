# Pilot CoT 58 — 苏剑林 Jianlin Su · empirical_anomaly

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例3：想让注意力"训短测长" → 从"熵不变性"推出 log n 缩放因子

- route: empirical_anomaly  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **205** · assistant(cot) **1145** · whole **1407**

- fact-check changes: 3

## user (setup)

```plaintext
It's late 2021. Transformer attention uses the scale softmax(QKᵀ/√d)V, and the √d is universally justified by a variance/initialization argument: it keeps logits from blowing up at init so training stays stable. That story is fine for training, but it says nothing about a stubborn empirical failure people keep brushing aside — models trained at one sequence length degrade badly when tested on longer sequences (train-short-test-long). The community treats length extrapolation as someone else's problem (change the position encoding, add ALiBi-style biases, etc.), and treats the scale factor as a fixed constant with nothing to do with length. You have a reproducible observation from your own MLM runs: accuracy at the training length is fine, but it collapses monotonically as you push the test length past it, and the standard √d explanation offers no lever to fix this. Propose a genuinely novel idea for what the attention scale should be, derived from first principles rather than tuned.
```

## assistant (cot)

```plaintext
Let me stare at the numbers I actually have, not the story everybody repeats. I train an MLM at length 64, then evaluate at 64, 128, 256, 512, 1024. At 64 it's fine. At 128 it's already worse, at 256 it's falling apart, and it keeps sliding. Nothing changed except n. The weights are the same, the position encoding is the same. So whatever is breaking is a function of length alone.

The √d people will wave this away: 'that's an extrapolation problem, use a different position encoding.' But √d is a constant — it has no n in it — so by construction it cannot be the thing that adapts to length. The variance-at-init argument explains why training doesn't explode; it says nothing about what happens when I feed a longer sequence at inference. The measured contradiction is: the scale that keeps training stable is silent about the exact failure I'm watching. I refuse to grid-search a new temperature per length — that would give me a fudge factor that dies the moment n changes again. I want to know what quantity is silently drifting with n, and cancel it.

So let me look at the attention weights themselves as a probability distribution: a_ij = softmax over j of λ·(q_i·k_j), with λ currently 1/√d. When I append new tokens, softmax must spread its probability mass over more candidates. Mechanically, the distribution gets flatter. There is a clean scalar that measures 'how spread out' this is: the entropy H_i = −Σ a_ij log a_ij. Entropy 0 means the head locks onto one token; entropy log n means uniform over everything. And here is the thing that clicks — as n grows, naive softmax has to raise its entropy, it de-focuses. The head that used to sharply attend to a few relevant tokens now smears attention across the newcomers. That's an intuition, not a theorem, but it's exactly a degradation you'd expect to hurt generalization to longer sequences.

That gives me a target I can actually solve for, not tune: the attention entropy should be roughly invariant to n. I want adding tokens not to dilute the focus already achieved on the old tokens. So: choose λ as a function of n such that H_i stays put.

Now derive. Write H_i via the standard logsumexp identity for a softmax distribution. Each sum over j is a sum of n similar terms, so approximate it as n times an expectation — and out pops an explicit log n term. That log n is precisely the entropy inflation the longer sequence injects. Everything else is O(1) in n. I only need a semi-quantitative estimate here — I'll be honest that making entropy *exactly* invariant is impossible; I'm hunting the dominant n-dependence. Assume q,k are post-LayerNorm with roughly fixed norm √d, so q·k = d·cosθ and the expectation reduces to an average over the angle. A saddle-point / Laplace-type approximation collapses that angular expectation to something roughly linear in λd. So H_i ≈ log n − c·λd + const. To cancel the length term, set c·λd = log n, i.e. λ ∝ log n / d.

That's the leap: the scale should carry a log n, not be constant. Absorb the messy constant into a hyperparameter κ, λ = κ log n / d. Then I dislike free knobs, so I calibrate: at the mainstream pretraining length (say 512) it should reduce to ordinary 1/√d. That fixes κ and leaves a hyperparameter-free rule where the effective scale is 1/√d times log_512(n): identical at 512, larger for longer, smaller for shorter.

How do I keep myself honest? This is an estimate resting on norm and near-orthogonality assumptions, so it lives or dies by a controlled experiment, not by the algebra. Decision rule: train at one length, evaluate across lengths against the plain √d baseline. The rule earns its keep only if, at the training length, the entropy-invariant scale is not worse (no cost) and at longer lengths it clearly improves over the baseline; if it merely trades training-length accuracy for extrapolation, I'd reconsider. One caveat I'll flag loudly: use log_512, not raw natural log — an uncalibrated log n / √d blows up for large n and would hurt gradients; and note the calibration is tied to the chosen reference length, so the story is about the dominant n-dependence, not an exact invariance.

**Core idea:** Treat attention weights as a distribution and demand its entropy be invariant to sequence length; the length-dependent entropy inflation is a log n term, so canceling it forces the scale to be (κ log n)/d — calibrated to reduce to 1/√d at the pretraining length, giving a hyperparameter-free scale ≈ log_512(n)/√d intended to improve length extrapolation.
**Non-trivial crux:** The forcing quantity is the entropy of the attention distribution (not init variance), and the derivation is only semi-quantitative — it must be calibrated so it equals 1/√d at the standard length, otherwise raw log n/√d explodes at large n and destroys gradients.
```

## fact-check changes

```plaintext
- (a) Removed the implicit assertion that the entropy-invariant scale definitively wins; reframed the closing as a decision rule to be tested against the √d baseline rather than a confirmed outcome.
- (c) Softened the 'de-focus hurts generalization' framing to intuition rather than a proven contemporaneous argument; noted the calibration ties invariance to the reference length (dominant n-dependence, not exact invariance).
- (d) Added caveat that the linear-in-λd step is a saddle-point/Laplace approximation and that calibration is reference-length-dependent.
```