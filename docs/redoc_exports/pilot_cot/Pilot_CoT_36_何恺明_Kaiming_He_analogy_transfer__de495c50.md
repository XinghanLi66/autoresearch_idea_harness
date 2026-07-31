# Pilot CoT 36 — 何恺明 Kaiming He · analogy_transfer

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例5：面对"不可复现"质疑，用协议化的精度数字回应

- route: analogy_transfer  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **206** · assistant(cot) **1124** · whole **1387**

- fact-check changes: 5

## user (setup)

```plaintext
It is early 2015. The trend in image recognition is clear: deeper convolutional networks tend to be more accurate — VGG at ~19 layers, GoogLeNet with careful branching. Batch normalization has just made it possible to train much deeper stacks without exploding or vanishing signals, so the natural expectation is that piling on more layers should keep helping. But when people push plain stacked networks well past 20 layers toward 30, 50, and beyond, accuracy stops improving and then gets *worse* — and, oddly, the degradation shows up even on the training set, not just at test time. The usual bag of remedies (better initialization, normalization between layers, more regularization) does not fix it. You are trying to build networks that are genuinely deeper AND better. Propose a genuinely novel idea for how to make very deep networks trainable — and be explicit about where your idea comes from, what transfers, and what would break if you got the mapping wrong.
```

## assistant (cot)

```plaintext
Let me start from a mechanism I trust from a completely different corner of computer vision, not from the deep-learning literature itself. In classical image processing and in optimization we constantly work with *residuals*: you don't estimate the full quantity, you estimate the correction on top of a known reference. Multigrid solvers, iterative refinement, even something as humble as predicting the difference from a base estimate — the recurring wisdom is that if a good reference is already available, learning the small deviation from it is far easier and better-conditioned than learning the whole thing from scratch. That is the source mechanism I want to transfer. I am Kaiming He, and my instinct is to ask: can a layer be asked to predict a residual rather than a fresh transformation?

But before transferring I force myself to be precise about what the reference *is* here, because a sloppy analogy will fail. In iterative refinement the reference is the previous estimate of the same quantity. In a stack of network layers, what is the natural reference for a block's output? The obvious candidate is the block's own input. So the mapping is: instead of asking a stack of layers to compute a target H(x), ask it to compute F(x) = H(x) - x, and reconstruct the output as F(x) + x. The reference x is carried forward for free by a plain, parameter-free connection.

Now the honesty check — what would make this a real problem rather than a cute analogy. The anomaly people wave away as overfitting is actually the wrong diagnosis: the *training* error goes up when you add layers. That rules out the cheap explanation. And here is the sharp contradiction I can construct on paper: take a shallower network that works, then make it deeper by appending extra layers. There exists, by construction, a solution where the added layers are identity mappings — that deeper network can reproduce exactly the shallower one's function, so it should be *no worse*. Yet optimizers cannot find it. The conclusion I lean toward: the difficulty is not representational capacity but optimization — solvers struggle to drive a stack of nonlinear layers toward an identity mapping. I should note this is a hypothesis about optimization difficulty, not a proof that vanishing/exploding gradients are absent; normalization already helps signal propagation, so the residual reformulation is aimed at a distinct, harder-to-name difficulty.

This is exactly where my transferred residual mechanism earns its keep, and I can now say precisely *what transfers and what breaks*. What transfers: reformulating the target as reference-plus-correction makes the good solution trivially reachable. If the optimal H is near identity, then F need only be pushed toward zero — and pushing a pile of weights toward zero is something SGD does happily. What would break the transfer: if I implement the shortcut as anything other than a clean, unobstructed identity path — say a learned 1x1 projection everywhere, or a gate that can attenuate it, or dropout on the shortcut — then I risk destroying the very property that made this work: the free, undamaged flow of both the reference forward and the gradient backward. My intuition is that the mechanism's power is in the *parameter-free identity*, not in the general idea of a side branch — and that is a claim I need to test, not assume.

This cuts against a natural reflex, which is to add machinery. The most relevant contemporaneous relative is Highway Networks, which route information through learned gating functions on the shortcut. My bet is the opposite of adding gates: the winning move may be a zero-parameter connection. Less is more — but I have to earn that.

How I keep myself honest. I will not just claim the plain deep net is worse; I will hold everything fixed — depth, width, training budget, normalization — and change only whether the shortcut is present, so any difference is attributable to that one variable. My pre-hoc prediction: the plain deep stack shows higher *training* error than its residual counterpart at equal depth; if instead the plain net trains fine, my optimization story is wrong and I must abandon it. And I will ablate the shortcut form itself: identity vs. projection-everywhere vs. gated shortcut. Decision rule — if the plain identity form is not at least as good as the fancier shortcuts, I have no business adding parameters; parsimony wins unless complexity clearly and reproducibly pays. There will be edge cases — when the input and output dimensions differ, the identity shortcut cannot be literally parameter-free and needs a projection at those transitions, so the clean-identity claim holds within a block, not across every dimension change. And any number I report will be tied to an explicit test protocol, because a lone accuracy figure without its evaluation setting is not a claim anyone can check.

**Core idea:** Transfer the classical residual/iterative-refinement principle into network layers: have each block learn a residual F(x)=H(x)-x on top of a parameter-free identity shortcut, so the output is F(x)+x — turning the hard-to-optimize "learn an identity" into the easy "drive the residual toward zero," which makes very deep nets trainable.

**Non-trivial crux:** The benefit hinges on the shortcut being a *clean, parameter-free identity* that passes signal and gradient unobstructed; replacing it with learned projections everywhere, gates, or dropout on the shortcut risks breaking exactly the property that solves the optimization problem — with the necessary exception of dimension-change points, where a projection is unavoidable.
```

## fact-check changes

```plaintext
- (a) Softened the on-paper optimization diagnosis from a 'forced conclusion' to a leaned-toward hypothesis, and added that it is distinct from vanishing/exploding gradients rather than proven.
- (a) Reframed the 'breaks the property' claim about non-identity shortcuts as a risk to be tested, not an established fact, consistent with it being pre-experiment.
- (b) Named Highway Networks explicitly as the true contemporaneous gated-shortcut competitor instead of the vague 'more branches, more gates' reflex.
- (c) Marked the parameter-free-identity superiority as intuition/bet to be earned, not an established original argument.
- (d) Restored the real exception: identity shortcuts require a projection at dimension-change points, so the parameter-free claim is not absolute.
```