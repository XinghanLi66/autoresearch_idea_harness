# Pilot CoT 66 — 苏剑林 Jianlin Su · analogy_transfer

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例5：把"Softmax+交叉熵"重解成"两两比较" → 推广出无类别不均衡的多标签损失（GlobalPointer 的地基）

- route: analogy_transfer  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **183** · assistant(cot) **1146** · whole **1386**

- fact-check changes: 3

## user (setup)

```plaintext
It's early 2020. Single-label classification has a clean, canonical recipe: softmax over the logits plus cross-entropy. It just works — no threshold tuning, no class-imbalance hacks. But multi-label classification (each sample belongs to k of n classes, with k varying per sample) is almost universally handled by splitting into n independent sigmoid binary classifiers, each with its own binary cross-entropy. When n is large and k is small, the negatives swamp the positives, so class imbalance comes back in full force, and you also have to tune a decision threshold per task to decide how many labels to emit. People pile on focal loss, class weights, threshold search — trick upon trick. Propose a genuinely new loss for multi-label classification that preserves the elegance and imbalance-freeness of softmax+cross-entropy, and that also cleanly handles the fact that k varies per sample.
```

## assistant (cot)

```plaintext
Everyone treats single-label and multi-label as two different worlds, and then patches the multi-label world with imbalance tricks. I don't want to patch. I want to ask: why is softmax+cross-entropy imbalance-free in the first place, and can I *transfer that exact property* over? So let me borrow a mechanism from a neighboring subarea — metric learning / ranking losses — and map it onto classification explicitly, being careful about what transfers and what breaks.

Start by re-reading single-label CE not as 'output a probability', but as an algebraic object. With logits s_1..s_n and target t, cross-entropy is -log( e^{s_t} / Σ_i e^{s_i} ) = log( 1 + Σ_{i≠t} e^{s_i - s_t} ). That rewrite is the whole game. Look at what it says: it's a smooth (logsumexp) surrogate for max(0, max_{i≠t}(s_i - s_t)). In plain words: *push every non-target score below the target score*. So the essence of softmax-CE is not probability at all — it's a pairwise comparison, a ranking constraint: target beats each non-target. The 'probability' interpretation is an implementation form; the ranking is the substance.

Once I see it as ranking, the source mechanism I want is clearer. In metric learning people already write things like triplet loss and its smooth cousins: enforce 'positive similarity > negative similarity' over pairs, aggregated via a soft-max / logsumexp. That community has been living with 'sum over positive×negative pairs of e^{s_neg - s_pos}' for a while. What transfers cleanly is precisely the pairwise-comparison-with-logsumexp skeleton. So I generalize: given a set of positive classes Ω_pos and negative classes Ω_neg, ask that *every* positive score exceed *every* negative score. The smooth surrogate is log(1 + Σ_{i∈Ω_neg, j∈Ω_pos} e^{s_i - s_j}). And notice single-label is the special case |Ω_pos|=1. With an added scale and margin this looks like the general form Circle loss unifies (with AM-softmax and triplet as special cases) — good, that suggests I'm working on a real, unifying object, not a random gadget.

Now — what *breaks* under the transfer? Metric-learning ranking losses only fix the *relative order*; they never tell you the absolute cutoff. In classification with variable k, that's the actual pain: at inference I must decide *how many* labels to emit, and 'tune a threshold per task' is the very ugliness I'm trying to kill. Ranking-transfer gives me imbalance-freeness (the logsumexp auto-balances every pairwise term, so large-n-small-k no longer drowns the positives), but it does NOT give me a decision rule. That's the seam.

The fix should be a construction, not another tuned hyperparameter. Introduce a virtual class whose score is fixed at s_0 = 0 — a built-in threshold that lives inside the same comparison. Then the requirement becomes: every positive score > 0, and every negative score < 0. Plug the virtual class into both sides of the pairwise loss and the sum splits: log(1 + Σ_{i∈Ω_neg} e^{s_i}) + log(1 + Σ_{j∈Ω_pos} e^{-s_j}). Clean — the threshold is now learned as part of the loss, and at inference I simply output every class with s_i > 0. No per-task threshold search, no imbalance weighting, and no activation on the logits (crucially, no sigmoid or softmax — that would destroy the raw-score comparison).

How do I keep myself honest? Two checks with clear decision rules. (1) Sanity: setting one positive should reduce *exactly* to standard softmax-CE — if it doesn't, my rewrite is wrong. (2) Empirically, compare against the sigmoid-BCE baseline on genuinely imbalanced multi-label tasks: the bar is that this form should at least match while removing threshold tuning; if it can't, the elegance isn't worth it. And the real test of whether I've captured the *essence* rather than a trick: can it grow? If I score every span (i,j) and treat 'is this span an entity of type c' as one giant multi-label problem with the 0-threshold separating entity/non-entity spans, the same loss should handle nested and flat NER uniformly. If the abstraction is right, that generalization should come nearly free — which is the kind of signature I'd trust.

**Core idea:** Re-read softmax+cross-entropy as a pairwise *ranking* loss (target scores beat non-target scores), transfer that logsumexp comparison from metric learning to multi-label — giving imbalance-freeness — then add a virtual class fixed at score 0 as a built-in threshold, yielding log(1+Σ_neg e^{s_i}) + log(1+Σ_pos e^{-s_j}) with the inference rule 'emit classes with score > 0'.
**Non-trivial crux:** Ranking transfer only fixes relative order, not the absolute cutoff needed when k varies per sample; the whole method works because the s_0=0 virtual class supplies that cutoff *inside* the loss — and it only holds if logits stay raw real numbers (no sigmoid/softmax), which would break the score comparison.
```

## fact-check changes

```plaintext
- (a) Replaced the narrated empirical framing ('if this form doesn't at least match... it isn't worth the elegance') into a clean decision rule for a test to be run, rather than an implied confirmed comparison; softened 'imbalance-freeness for free' to a claimed property under test.
- (c) Softened 'is exactly the general Circle-loss form' to 'looks like the general form Circle loss unifies', and hedged the sanity-check as intended reduction rather than asserted fact, avoiding overstated certainty of the unification framing.
- (e) No non-English words were present; minor wording cleanups only.
```