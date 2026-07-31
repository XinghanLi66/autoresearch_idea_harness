# Pilot CoT 29 — 何恺明 Kaiming He · first_principles

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例3：Rethinking ImageNet Pre-training——把控制变量做到极致，并"偏袒对手"

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **186** · assistant(cot) **1086** · whole **1329**

- fact-check changes: 4

## user (setup)

```plaintext
It is 2018. Across object detection and instance segmentation, one workflow is treated as self-evident: pre-train a backbone on ImageNet classification, then fine-tune on the target task (COCO, etc.). Every strong system does it; leaderboards, tutorials, and reviewers all assume it. The intuition is that pre-training injects both a useful initialization and a regularizing prior learned from a million labeled images, and that without it detectors would either fail to converge or overfit the smaller target set. Yet no one has cleanly isolated what pre-training actually buys. Suppose you suspect the belief is at least overstated — that with enough training a model initialized randomly could match the fine-tuned one. Before running a single large experiment, reason on paper toward a genuinely novel claim and, crucially, toward the argument structure that could make such a claim survive attack. What is the idea?
```

## assistant (cot)

```plaintext
Start from what the consensus actually asserts, stated precisely enough that it could be false. "ImageNet pre-training is necessary for good detection" bundles two very different claims: (a) it gives a better *initialization* (faster, easier optimization), and (b) it gives *regularization / a better final solution* that from-scratch training cannot reach given any budget. People treat these as one thing. My first move is to refuse the bundle. If I can separate them, most of the mystique may collapse into claim (a), which is unsurprising, leaving claim (b) — the interesting, load-bearing one — exposed and possibly wrong.

Now the deductive core. Ask what *must* be true. A detector trained from random initialization and one initialized from ImageNet weights have the *same architecture, same capacity, same target data, same loss*. The only difference is the starting point in parameter space. Optimization is a path, not a destination in principle: given a sufficiently expressive network and enough gradient steps, there is no theorem saying the random-init path cannot reach a solution as good as the pre-trained one on the target distribution. The pre-trained weights are just one point that happens to sit closer to a good basin. So on pure representational grounds, from-scratch is *not forbidden* from matching — it can only be forbidden by an *optimization* obstacle or by a genuine *statistical* (regularization) advantage. That splits the question sharply: is the gap I expect to see a path-length problem or a destination problem? This is the same construction discipline I trusted with residual learning — write down the solution that ought to exist, and if experiments can't find it, suspect optimization before capacity.

So the leap is not "prove from-scratch is better." That would be cheap and easy to dismiss. The leap is methodological: **make fairness itself the design object.** If I want to claim the consensus is wrong, the only load-bearing support is a comparison no one can wave away by saying "you didn't tune the baseline." So I invert the usual instinct. Instead of tuning my side, I pour every advantage onto the *pre-training* side and compensate every known disadvantage of from-scratch, then check whether the gap still exists. Reason it through: what makes from-scratch legitimately worse? Two things I can name a priori. First, it converges more slowly — that is claim (a), and it is real, so I must not let it masquerade as claim (b). The fix is deductive, not empirical: hold the *comparison* on a same-budget axis so "slower" and "worse-at-the-end" are visibly different curves. Second, from-scratch on the small batch sizes typical of detection breaks normalization statistics — BatchNorm estimates become unreliable, and that is a confound that has nothing to do with pre-training per se. So I must remove it with a batch-independent normalization (such as Group Normalization or synchronized BN), or the whole attribution is polluted.

Everything else must be frozen word-for-word: one variable only, initialization. Same architecture, same data, same hyper-parameters as the fine-tuning recipe, same evaluation protocol. Minimal changes. If I change five things and win, I've proven nothing.

Why this cuts against the room: the field assumed pre-training gives regularization and a better ceiling. My construction says: strip the confounds, extend training to pay off the convergence debt, and the ceiling should be reachable from noise. If that holds, pre-training reduces to a *speed-up*, not a *quality* prior — but that is the hypothesis to test, not a foregone conclusion.

Keeping myself honest. I write the prediction down before running: from-scratch will trail early and, given enough iterations, close the gap toward matching final accuracy; if instead a *persistent* final-accuracy gap survives after I've compensated normalization and given generous budget, then pre-training really does regularize and I was wrong — I report that. Decision rule: the claim only stands when the pre-trained side has been fed to satiety (best tuning, all its advantages) and the gap *still* vanishes at equal budget. I also demand it replicate across different amounts of target data, so it isn't a single lucky point. And I keep the honest caveat: this is about sufficient-data target tasks with enough training budget; when target data is very small or budget is short, pre-training's early speed-up is a real, practical benefit I am not denying.

**Core idea:** Decompose "pre-training helps" into initialization-speed vs. final-quality, then test the disputed quality claim with a pro-competitor, single-variable comparison — freeze everything but initialization, compensate from-scratch's real handicaps (convergence budget, normalization), and compare on a same-budget axis; the conjecture to test is that pre-training buys faster convergence, not a better or more regularized final solution.

**Non-trivial crux:** The result rests entirely on isolating one variable while *favoring the opponent* and separating "converges faster" from "ends up better" — and any such conclusion holds only under sufficient target data and training budget; short-budget or low-data regimes still genuinely benefit from pre-training's speed-up.
```

## fact-check changes

```plaintext
- (a) Softened the pre-written 'prediction confirmed' framing: the final paragraph and 'why this cuts against the room' now state the speed-up-not-quality result as a hypothesis to be tested with a decision rule, rather than asserting the ceiling 'should be reachable' as settled outcome.
- (a/c) Toned down the residual-learning analogy from 'the fault is optimization, not capacity' to 'suspect optimization before capacity' to avoid stating an untested conclusion as fact.
- (d) Made the normalization confound accurate: specified it is small batch sizes typical in detection that break BatchNorm, and named the actual contemporaneous batch-independent fixes (Group Normalization / synchronized BN) used in the real work.
- (a) Removed the claim of replication 'across different frameworks' as an asserted design demand; kept replication across amounts of target data, which matches the actual study.
```