# Pilot CoT 65 — 苏剑林 Jianlin Su · first_principles

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例5：把"Softmax+交叉熵"重解成"两两比较" → 推广出无类别不均衡的多标签损失（GlobalPointer 的地基）

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **186** · assistant(cot) **1307** · whole **1550**

- fact-check changes: 3

## user (setup)

```plaintext
It's early 2020. For single-label classification, everyone agrees the golden pairing is softmax + cross-entropy: clean, well-behaved, and crucially free of class-imbalance trouble because it normalizes over classes jointly. But multi-label classification (each sample belongs to k of n classes, where k varies per sample) has no such clean default. The standard move is to split it into n independent sigmoid binary classifiers. When n is large and k is small, negatives drown positives, so class imbalance sneaks back in, and you're stuck hand-tuning a decision threshold per task. Question: can you derive a single loss for multi-label classification that keeps the elegance and imbalance-immunity of softmax-CE, and that also tells you at inference time how many classes to output — without per-task threshold tuning? Give a genuinely novel construction, reasoned from first principles rather than assembled from tricks.
```

## assistant (cot)

```plaintext
Let me refuse the usual starting point. The reflex answer — n sigmoids — I distrust, and I distrust it for a *derivable* reason, not an empirical one: sigmoid decomposes the joint problem into n isolated binary decisions, and the moment you do that, each decision sees a hugely skewed positive/negative ratio. Imbalance isn't a bug of the data here; it's a bug you *introduce* by choosing the wrong factorization. So before touching any trick, I want to ask: what does softmax + cross-entropy actually *do*, structurally, that makes it imbalance-free?

People say 'it outputs a probability distribution.' I think that's the implementation form, not the essence. Let me write single-label cross-entropy for logits s_1..s_n with target class t and simplify it as an identity, no approximations: -log( e^{s_t} / sum_i e^{s_i} ) = log( sum_i e^{s_i} ) - s_t = log( 1 + sum_{i≠t} e^{s_i - s_t} ). Stare at that last form. It contains no probabilities at all. It is logsumexp over the quantities (s_i - s_t) for every non-target i, padded with a zero. And logsumexp is just the smooth maximum. So this loss is a soft version of max(0, max_{i≠t}(s_i - s_t)). Its *true* objective is: push every non-target score below the target score. It's a pairwise comparison — target beats each non-target — not a probability statement.

That reframing is the whole leap, and once I have it the generalization is almost forced by symmetry. If the single-label essence is 'the one target beats every non-target,' then the multi-label essence must be 'every target beats every non-target.' So I just extend the sum over all ordered pairs (i in negatives, j in positives): log( 1 + sum_{i∈neg, j∈pos} e^{s_i - s_j} ). This is derived, not fitted. I notice it's exactly the general shape that, with a scale and a margin, connects to the metric-learning loss family — things like triplet loss and margin-softmax variants are special cases — which is a good consistency check: my construction sits where known metric-learning losses already live, so I haven't invented something off in the weeds.

But there's a genuine gap this pairwise loss does NOT close, and I have to be honest that it's a real gap: it only enforces relative ordering. At inference it tells me positives rank above negatives, but not *how many* to output — and k varies per sample, so I can't just take top-k. This is the crux, and it wants a construction, not a heuristic threshold search. So I introduce an invariant reference point: a virtual class with score fixed at s_0 = 0. Now the requirement becomes absolute and self-calibrating — every real target score should be > 0, every non-target score should be < 0. The virtual zero *is* the threshold, baked into the loss.

Substitute s_0 = 0 into the pairwise form and let the pairs split cleanly. Because 0 acts both as a 'negative to beat' for positives and a 'positive to beat' for negatives, the double sum decouples into two independent logsumexp terms: log( 1 + sum_{i∈neg} e^{s_i} ) + log( 1 + sum_{j∈pos} e^{-s_j} ). Beautiful — it separates. The first term drags all negative scores below 0; the second lifts all positive scores above 0. No imbalance weighting anywhere, because logsumexp auto-balances the contribution of each term by construction, and no per-task threshold, because the threshold is the fixed zero.

How do I keep myself honest before believing any of this? First, degeneracy checks I can do on paper: set the positive set to a single class and it must collapse exactly back to single-label softmax-CE — it does. Second, an explicit warning that follows from the derivation: the scores must range over all reals, so no sigmoid or softmax activation before this loss — applying one would break the s vs 0 comparison. Third, the empirical decision rule I'd commit to: since the whole selling point is 'no imbalance, no threshold tuning,' I'd test it against the n-sigmoid baseline specifically in the large-n small-k regime where imbalance bites, and hold myself to this rule — if it needs any per-task threshold retuning to match, my zero-class claim has failed and I should not claim it. I haven't run that comparison yet; the loss stands or falls on that test, not on the elegance of the derivation. I'd write the full motivation→derivation→code→experiment up publicly so every 'I think this is right' is forced into 'I can derive it / I verified it.' And I already suspect this clean form could grow beyond plain classification: score every span (i,j) with a biaffine-style head and treat 'is this span an entity of type c' as one giant multi-label problem with the zero-class separating entity from non-entity spans — that would aim to handle nested and flat NER in one uniform loss. But that generalization is a downstream idea to be tested; the loss stands on its derivation and the imbalance test alone.

**Core idea:** Reinterpret softmax cross-entropy as a smooth 'each target score beats each non-target score' pairwise comparison (not a probability), extend it to all positive-vs-negative pairs for multi-label, and insert a virtual class with score fixed at 0 as a built-in threshold — yielding log(1+Σ_{neg}e^{s_i}) + log(1+Σ_{pos}e^{-s_j}), imbalance-free and threshold-free, outputting exactly the classes with score > 0.

**Non-trivial crux:** The pairwise reformulation only gives relative ordering; the decisive, easily-missed move is the fixed-zero virtual class that converts an ordering loss into a self-calibrating decision rule — and it only works if scores span all reals, so no sigmoid/softmax activation may precede this loss.
```

## fact-check changes

```plaintext
- (b) Replaced the anachronistic/specific 'Circle-loss family — AM-softmax, triplet, all special cases' reference: Circle loss (2020, CVPR) is roughly contemporaneous and its exact 'all special cases' claim is Su's later framing; softened to a generic metric-learning connection (triplet, margin-softmax variants) to avoid asserting an unearned unification.
- (a) Removed the on-paper assertion that the empirical comparison outcome was known; explicitly marked the imbalance/threshold test as not-yet-run with a decision rule.
- (c) Softened the NER/GlobalPointer generalization from a stated capability to a downstream idea to be tested, since that was a later development rather than a decided contemporaneous result.
```