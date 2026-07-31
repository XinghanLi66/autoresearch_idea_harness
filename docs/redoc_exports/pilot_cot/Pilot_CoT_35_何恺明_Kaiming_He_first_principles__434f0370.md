# Pilot CoT 35 — 何恺明 Kaiming He · first_principles

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例5：面对"不可复现"质疑，用协议化的精度数字回应

- route: first_principles  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **173** · assistant(cot) **1043** · whole **1273**

- fact-check changes: 5

## user (setup)

```plaintext
It is 2015. Deep convolutional networks have gotten better as they got deeper, up to a point. But now the community is hitting a wall: when we stack plain convolutional layers past roughly twenty and beyond, accuracy does not just plateau — it degrades. The reflexive explanations are floating around: overfitting, vanishing/exploding gradients (which better initialization and normalization were supposed to have largely tamed), too little data, insufficient regularization. Everyone assumes that in principle a deeper network is at least as expressive as a shallower one, yet in practice deeper plain nets are worse. Suppose you want to attack this not by trial-and-error over activations, initializations, and normalization tricks, but by reasoning from what MUST be true. What is the argument, and what construction does it force? Propose a genuinely novel idea.
```

## assistant (cot)

```plaintext
Let me refuse, first, to file this under 'overfitting.' That word is doing too much work in everyone's mouth. So I go look at the one place that decides it: the training error. If a deeper net overfit, it would fit the training set better, not worse. But the deeper plain net has HIGHER training error. That single fact kills the overfitting story outright. This is not a generalization failure. It is an optimization failure. Good — now I have a real problem, not a tuning complaint.

Now I want to make this sharp enough to attack on paper, before I touch a GPU. Let me construct the solution that MUST exist and see whether reality contradicts it. Take a shallower network that trains fine. Build a deeper one by copying it exactly and appending extra layers on top. There is a trivially available parameter setting for those extra layers: make each of them compute the identity map. Then the deeper net computes precisely the same function as the shallower one. Therefore, by construction, the deeper net's training error should be no worse than the shallower net's — the shallower solution is a feasible point in the deeper net's parameter space.

So here is the contradiction, and it is a clean either/or. Such a solution provably exists. Yet the optimizer, given more layers, fails to find something even that good. I should be careful: this is a constructed-existence argument, not a proof that the optimizer cannot in principle find better — but empirically it does not. The most natural place the fault can live: it is not that the deeper model can't express the answer — it demonstrably can — it is that the optimizer struggles to drive a stack of nonlinear layers to realize the identity. That is my hypothesis about where the disease sits: not capacity, but the difficulty of learning identity-ish mappings with plain stacked nonlinearities.

Once the problem is stated that way, the fix is nearly forced. If the thing that's hard to learn is 'stay near identity,' then don't ask the layers to learn identity. Reparameterize so that identity is the DEFAULT — free, at zero cost — and let the layers only have to learn the deviation from it. Let the desired mapping be H(x). Instead of asking a block to produce H(x) directly, wire a parameter-free path that carries x forward unchanged and let the block produce F(x) = H(x) − x, so the output is F(x) + x. Now, whenever the optimal local mapping is close to identity, the optimizer just has to push F toward zero — pushing a residual toward zero via weights near zero should be easier than sculpting an identity out of nonlinear layers. The construction that MUST exist has become one that is, I conjecture, much EASIER to reach. I should note the contemporaneous cousin here: Highway Networks already carry information forward, but through learned gates that can attenuate the path; my move is to strip that down to a bare, ungated identity.

I want to keep myself honest here, because the elegant story could be self-flattering. The discipline is that the shortcut should add nothing — no parameters, no gate, no learned scaling — so that if it helps, the help is attributable to exactly this identity-preserving path and nothing else. That gives me falsifiable comparisons rather than a black box. Plan the ablations before believing anything: put the plain deep net and the residual deep net side by side under identical everything, and read the TRAINING error first — if my localization is right, the residual form should let the deep net at least match the shallow net's training error rather than degrade. Then vary only the shortcut: identity vs. replacing it with a learned projection everywhere vs. inserting a gate on the shortcut. My hypothesis is that anything that obstructs the clean identity path — extra parameters on the shortcut, a gate that can attenuate the signal — reintroduces the very difficulty I removed, so it should not help and may hurt. Decision rule: if the fancier shortcut variants don't clearly beat the bare identity, keep the bare identity on parsimony and cost grounds; and if identity shortcuts fail to fix the training-error degradation at all, then my localization of the disease was wrong and I go back to the construction.

This cuts against the room, where the instinct is to add machinery. The whole argument says the win may come from taking machinery away and making identity free.

**Core idea:** Because a deeper net can always emulate a shallower one by setting added layers to identity, higher TRAINING error in deeper plain nets points to an optimization difficulty (not overfitting or capacity); reparameterize each block to learn a residual F(x)=H(x)−x with a parameter-free identity shortcut, making the near-identity solution trivially reachable.

**Non-trivial crux:** Keep the shortcut exactly identity — parameter-free, ungated, unscaled — since any added machinery on that path risks reintroducing the identity-learning difficulty the construction was designed to eliminate; projections are used only where dimensions must change, not as a default.
```

## fact-check changes

```plaintext
- (a) Removed the 'pre-registered prediction' framing and the implied confirmation; recast the residual-is-easier claim and the ablation expectations as hypotheses to be tested with an explicit decision rule, since numeric outcomes were not yet run at that moment.
- (a) Softened 'the disease is located' / 'the construction that is EASY to reach' from asserted fact to hypothesis, noting the existence argument does not prove optimizer failure.
- (b) Added the true contemporaneous competitor (Highway Networks / learned gates) as the era-appropriate contrast, contextualizing the bare-identity choice.
- (e) Removed the English slogan-style 'Less is More' branding that overstated it as a decisive principle; reframed as ordinary parsimony/attribution discipline.
- (d) Added caveat that the constructed-existence argument shows feasibility, not impossibility of finding better solutions, and hedged the causal 'win comes from' to 'may come from'.
```