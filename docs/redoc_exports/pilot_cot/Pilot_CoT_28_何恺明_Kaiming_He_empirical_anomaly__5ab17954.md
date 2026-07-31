# Pilot CoT 28 — 何恺明 Kaiming He · empirical_anomaly

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例3：Rethinking ImageNet Pre-training——把控制变量做到极致，并"偏袒对手"

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **171** · assistant(cot) **1024** · whole **1252**

- fact-check changes: 4

## user (setup)

```plaintext
It is 2018. In object detection and segmentation, 'pre-train the backbone on ImageNet, then fine-tune on the target task' is treated as an unquestioned default. Nearly every state-of-the-art detector cites ImageNet pre-training as essential, and the community's intuition is that classification pre-training provides both a warm start and a form of regularization that boosts final accuracy. But there is a nagging observation that people keep waving away: when a few groups have tried training detectors from random initialization, they got worse numbers and everyone concluded 'see, you need pre-training.' You suspect that conclusion may have been reached under an unfair comparison. Propose a genuinely novel way to settle whether ImageNet pre-training is actually necessary for detection/segmentation accuracy — not just faster — and how you would make the claim survive scrutiny.
```

## assistant (cot)

```plaintext
Let me start from the observation that won't sit still. People say 'we trained from scratch and it was worse, therefore pre-training is necessary.' I keep looking at those runs and something is off. The gap they report always shows up early — the scratch model is way behind at the start — and then the curves seem to be still moving when they stop. That's not obviously the signature of a model that fundamentally can't reach the same accuracy; it could just be a model that needs more iterations. So the standard story 'pre-training gives a better final solution / regularization' may be conflating two different things: converging faster and converging to a better place.

First, kill the cheap explanations before I let myself get excited. Is the scratch gap caused by overfitting? If anything scratch has less to memorize, and the gap I care about is on the target metric, not a train/val divergence — so I should check that directly. Is it a bug in their setup? Possibly, but I can't assume that. Is it optimization? Partly. But there's one more confound I refuse to let hide inside 'is it pre-training': normalization. Detectors run at tiny effective batch sizes because images are huge, and BatchNorm statistics are noisy at small batch. A pre-trained backbone smuggles in stable, frozen BN statistics learned on ImageNet. A scratch model has no such gift — so any 'scratch is worse' result might partly be 'BN is broken at this batch size.' That is not a fact about pre-training. I have to pull that variable out cleanly, using a normalization that doesn't depend on batch statistics (e.g. Group Normalization, or synchronized/frozen BN), or the whole comparison is compromised.

Now the move, and it's a methodological one rather than a new module. The wrong instinct here is to go build some clever scratch-training trick and announce 'scratch is actually better.' I don't want to win that way. The honest and much sharper move is to design the comparison so that I am *helping the opponent* — pre-training — as much as possible, freeze every other variable, leave exactly one thing free (initialization), and compensate the one plausible handicap of scratch (it may need more iterations, so give it the iterations). If, after I have fed pre-training every advantage and removed scratch's spurious disadvantages, scratch still lands at the same final accuracy, then the necessity claim collapses on its own. I don't have to prove scratch is better. I only have to show it is *not worse* under a comparison nobody can attack.

The control has to be almost fanatical: identical architecture, identical data, identical hyper-parameters as the fine-tuning recipe, identical evaluation protocol — the only difference is the initial weights. And I will not report a single accuracy number without saying at what training budget it was measured, because I suspect the entire misconception lives in that omission. So I'll put both on the same x-axis of total iterations/samples seen and read off two separate quantities: does pre-training reach a given accuracy sooner, and does it reach a higher accuracy in the limit? My hypothesis going in is 'faster yes, higher no,' but the decision rule is what matters: I record that hypothesis, then let the same-budget curves decide. If scratch matches at convergence, the necessity claim fails; if it doesn't match, I don't get to invent a comforting story afterward — any post-hoc explanation stays unproven until re-tested on a different framework and data regime.

This cuts hard against consensus because 'you need ImageNet' is treated as obvious. My claim — Kaiming here — is narrower and more dangerous: the hypothesis is that pre-training buys convergence speed, not necessarily a better optimum or free regularization, and that the field may have mistaken the former for the latter because nobody trained scratch long enough with a batch-independent normalization. Keeping myself honest: repeat across multiple detection/segmentation frameworks and multiple training-data sizes; if any equivalence I find holds only at full data, I'd keep the caveat that in a genuinely small-data regime pre-training may still help, and I would say so rather than overclaim.

**Core idea:** Settle 'is ImageNet pre-training necessary' with a maximally fair, single-variable comparison — freeze everything but initialization, remove the batch-dependent normalization confound, and *give pre-training every advantage* while compensating scratch's likely handicap (more iterations) — then, on a same-budget x-axis, separate 'converges faster' from 'converges better' and let the curves decide which one pre-training actually provides.
**Non-trivial crux:** The whole design hinges on disentangling two things the community fused — speed vs. final accuracy — and on eliminating small-batch BatchNorm as a hidden confound; the test is whether scratch is 'not worse for final accuracy given enough training,' with the standing caveat that in a truly data-scarce regime pre-training can still help.
```

## fact-check changes

```plaintext
- (a) Removed narrated confirmation ('I expect yes'/'my prediction: no' treated as settled, 'confirms the pre-hoc hypothesis'); reframed as a recorded hypothesis plus an explicit decision rule that the same-budget curves decide.
- (a) Changed final claims from asserted findings ('the finding is that pre-training buys the former, not the latter') to what would be tested/hypothesized, since the CoT is at the design stage.
- (d) Softened absolutized causal claims (e.g. 'might just be BN broken' -> 'might partly be'; BN stats 'garbage' -> 'noisy'; 'less data to memorize' phrased as something to check directly).
- (b/c) Named the contemporaneous batch-independent normalization option (Group Normalization / sync or frozen BN) rather than leaving it generic and implicitly hindsight.
```