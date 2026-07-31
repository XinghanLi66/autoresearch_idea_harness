# Pilot CoT 64 — 苏剑林 Jianlin Su · empirical_anomaly

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例5：把"Softmax+交叉熵"重解成"两两比较" → 推广出无类别不均衡的多标签损失（GlobalPointer 的地基）

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **210** · assistant(cot) **1252** · whole **1519**

- fact-check changes: 4

## user (setup)

```plaintext
It's early 2020. For single-label classification, the standard recipe is softmax followed by cross-entropy — clean, robust, and famously free of any class-imbalance headache: you never have to hand-tune weights or thresholds. But multi-label classification (each sample belongs to some k of n classes, with k varying sample to sample) is almost always handled by splitting the problem into n independent sigmoid binary classifiers. In practice this behaves badly: when n is large and k is small, the negatives swamp the positives, class imbalance comes roaring back, and you end up tuning a decision threshold per task to decide how many labels to emit. People treat this as just 'the nature of multi-label' and patch it with focal loss, resampling, or per-class thresholds. You suspect the standard story is wrong. Propose a genuinely new loss for multi-label classification that keeps the softmax-CE elegance and the imbalance-immunity, and that tells you at inference time how many labels to output — without per-task threshold tuning.
```

## assistant (cot)

```plaintext
Let me start from the thing that actually bugs me, because it's a measured contradiction, not a philosophical one. Single-label softmax-CE has no imbalance problem — even with 1000 classes and one correct answer, nobody adds class weights, and it just works. Multi-label with n sigmoids has a severe imbalance problem at the same n. Same data, same n classes, and yet one formulation is immune and the other is sick. The community explains this away as 'multi-label is inherently imbalanced.' But that can't be a property of the *task* if single-label at the same scale is fine. So the imbalance must be an artifact of the *decomposition into n binary problems*, not of multi-label itself. That's the crack I want to pry open.

So I refuse to accept 'add focal loss / resample / tune thresholds.' Those are patches on a symptom. I want to ask: what is softmax-CE actually doing that makes it imbalance-free, and can I carry that property over?

Let me write it out honestly. For logits s_1..s_n with target t, cross-entropy is -log( e^{s_t} / sum_i e^{s_i} ) = log( sum_i e^{s_i - s_t} ) = log( 1 + sum_{i != t} e^{s_i - s_t} ). Now stare at that closed form. Everyone tells the story that CE is about 'producing a probability distribution.' But this rewrite doesn't mention probability at all. It's a smooth (logsumexp) approximation of max(0, max_{i != t}(s_i - s_t)). In plain words: it pushes every non-target score below the target score. The essence isn't 'output a probability' — the essence is a *pairwise comparison*: target score should beat every non-target score. That reinterpretation is the whole game.

And once I see it as pairwise comparison, the generalization is almost free. If instead of one target I have a positive set Omega_pos and a negative set Omega_neg, the natural demand is: every positive score beats every negative score. So form all cross pairs and softly hinge them: log( 1 + sum_{i in neg, j in pos} e^{s_i - s_j} ). This is symmetric in how it treats positives and negatives, and logsumexp keeps it from being a plain sum of independent binary losses — it's a comparison structure, so no single class simply dominates by count of terms. (I'll note in passing that this general 'pull positives above negatives' pairwise form is the same family as unified pair-based losses like circle loss / AM-softmax — a sanity check that I'm on a real object, not a hack.)

But this only constrains *ordering*. It never tells me, at inference, how many labels to emit — the variable-k problem. This is where I need one small construction rather than a threshold I tune per task. Idea: introduce a virtual class with score fixed at 0. Semantically it's the boundary — positives should score above 0, negatives below 0. Feed 0 into both roles and the double sum splits cleanly into log(1 + sum_{i in neg} e^{s_i}) + log(1 + sum_{j in pos} e^{-s_j}). Now inference is trivial and threshold-free: output exactly the classes with score > 0. The variable count falls out of the model itself.

Warning to myself and future readers: because the whole thing lives in raw-score comparison space, the logits must range over all reals — no sigmoid, no softmax on top. That would poison the comparison. This is the kind of caveat that's easy to forget and fatal to ignore. And an edge case I should be honest about: if a sample has no positive labels (or, symmetrically, no negatives), one of the two sums is empty and the corresponding log-term vanishes — the loss still degrades gracefully but the 'every positive beats every negative' framing needs that empty-set reading.

How do I keep myself honest? Two decision rules. First: on a genuinely single-label problem this loss must reduce exactly to standard softmax-CE — if it doesn't, my 'essence' claim is false. It does, by construction. Second — the empirical one — take a real imbalanced multi-label task and compare against the n-sigmoid baseline. If the pairwise-with-virtual-threshold form doesn't clearly beat the sigmoid baseline without any per-class weighting or threshold search, then my 'imbalance is an artifact of the decomposition' story is weakened and I should say so. I won't declare victory in advance; I'll write the derivation, paste the loss, run it, and let the measurement rule.

And this is exactly the kind of clean abstraction that should *grow*. If 'target > non-target, with a 0-threshold' is the real object, then scoring every span (i,j) and asking 'is this span an entity of type c' becomes one giant multi-label problem — nested and non-nested NER potentially handled by the same machinery. I'll leave that as a conjecture to build next, not a claim.

**Core idea:** Reinterpret softmax cross-entropy not as 'produce a probability' but as a pairwise comparison 'every target score beats every non-target score'; generalize to multi-label as log(1 + sum over neg-pos pairs of e^{s_i - s_j}), then inject a virtual class with fixed score 0 as a built-in threshold, yielding log(1+sum_neg e^{s_i}) + log(1+sum_pos e^{-s_j}) — imbalance-free and threshold-free (emit classes with score > 0).

**Non-trivial crux:** The magic is the fixed-0 virtual class that converts a pure ranking loss into a directly decidable one for variable label counts; and it only works because scores are compared in raw-logit space — you must NOT apply sigmoid/softmax to the outputs, or the comparison structure that kills imbalance is destroyed.
```

## fact-check changes

```plaintext
- (a) Softened the deterministic 'logsumexp automatically balances' claim to a structural argument, and reworded the empirical test conclusion so failure 'weakens' rather than flatly disproves — no result asserted before running.
- (c) Marked the circle-loss/AM-softmax connection as a family sanity check rather than an original decisive argument; changed 'learned threshold' to 'built-in/fixed threshold' since the 0-score class is fixed, not learned.
- (d) Added the empty-positive/empty-negative-set edge case that the pairwise framing needs to handle.
- Minor: 'nested and non-nested NER handled by the same machinery' softened to 'potentially handled' since it is stated as a conjecture.
```