# Pilot CoT 34 — 何恺明 Kaiming He · empirical_anomaly

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例5：面对"不可复现"质疑，用协议化的精度数字回应

- route: empirical_anomaly  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **170** · assistant(cot) **1083** · whole **1310**

- fact-check changes: 3

## user (setup)

```plaintext
It is 2015. Convolutional networks have been getting deeper — VGG at ~19 layers, and everyone believes the trend is monotone: more depth, richer features, better accuracy, at worst diminishing returns held back by vanishing gradients (which normalized initialization and batch normalization already largely tame). You keep stacking plain conv layers well past 20, past 30, past 50, on a real image classification benchmark. But something is off: the deeper stack is doing WORSE. The common explanations on the table are 'overfitting' and 'vanishing/exploding gradients.' Both feel like reflexes rather than answers. You have the training curves in front of you. Find the real problem, and propose a genuinely novel idea for why depth stops helping and what to do about it.
```

## assistant (cot)

```plaintext
Start with the number that shouldn't exist. I stack the plain net deeper and the accuracy drops. The reflex in the room is 'overfitting.' So let me look at the training error, not the test error. If it were overfitting, the deeper net would fit the training set better and generalize worse — training error down, gap up. That is the opposite of what I see. The deeper net has HIGHER training error. It can't even fit the data it's shown. Overfitting is off the table. I want to be careful here: this is the whole ballgame, so I refuse to explain it away.

Next cheap suspect: vanishing gradients. But we already use normalized initialization and BN; I can check that the gradients and activation statistics are healthy — norms are sane, signals propagate. So the standard 'optimization is broken because signals die' story doesn't fit the measurement either. I'm now sitting in front of a contradiction that neither of the two comfortable explanations survives. Good. That's a real problem, not a tuning problem.

Let me force it into a sharp corner with a construction, on paper, before I touch a single hyperparameter. Take a shallower net that trains fine. Build a deeper net from it: copy the shallow net exactly, and append extra layers. Now, there EXISTS a solution the deeper net could adopt that is at least as good as the shallow one — let every appended layer compute the identity mapping. Under that construction the deeper net reproduces the shallow net's function exactly, so its training error should be no worse. That is a solution by construction. And yet SGD does not find it, or anything as good. So the network is not lacking capacity or representational power; the optimizer is failing to reach a solution we KNOW is inside the hypothesis space. This is not overfitting, not capacity, not dead gradients — it's a degradation of optimization: the solver struggles to drive a stack of nonlinear layers to approximate identity.

That reframes the whole target. The disease is: asking nonlinear layers to learn identity (or near-identity) mappings is hard. So don't ask them to. Make identity the DEFAULT, the thing you get for free, and let the layers spend their capacity only on the deviation from it. If the desired mapping is H(x), let the layers fit F(x) = H(x) - x, and add x back by a plain wire: y = F(x) + x. When the optimal mapping is close to identity, the optimizer just has to push F toward zero — driving a residual to zero is plausibly easier than driving a stack to reproduce identity. The shortest path from input to output is now zero layers deep, so gradient can flow straight through. I like this because it is a subtraction, not an addition: the shortcut carries no parameters, adds no cost. Less is more.

Now keep myself honest, because a clean story is exactly when I'm most likely to be fooling myself. Before running: if degradation is really an optimization pathology and not capacity, then the residual form should let training error keep decreasing as depth grows past where the plain net degrades, and I'd expect little difference at shallow depths where the plain net already trains fine. I'll test plain vs. residual at matched depth, parameters, and training budget, one variable at a time, and read off which behavior the training/test curves actually show. And I must resist decorating the shortcut. The temptation is to make it a learned gate — the contemporaneous idea here is the gated shortcut of Highway Networks — or a 1x1 projection everywhere 'to be safe.' Decision rule: the pure identity shortcut is the null; any added machinery on the shortcut (gating, projection where dimensions already match, dropout on the skip) must EARN its place by clearly beating identity under identical conditions. If it doesn't beat identity, I'll drop it on parsimony grounds and on the suspicion that it obstructs the unobstructed signal path that is the entire point — and note that a gate that can close is exactly what reintroduces the identity-learning burden I'm trying to remove.

One caveat I keep: this addresses the depth-degradation of trainability. It is not a claim that arbitrary depth is free, and dimension changes between stages still need SOME projection — the minimalist rule is to keep even that as small as possible, applied only where dimensions actually change.

**Core idea:** The failure of very deep plain nets is an optimization degradation, not overfitting or dead gradients — argued from the fact that an identity-preserving construction should match the shallow net yet SGD can't find it; so reformulate each block to learn a residual F(x)=H(x)-x with a parameter-free identity shortcut y=F(x)+x, making near-identity mappings easy to optimize.

**Non-trivial crux:** The decisive evidence is that the deeper net has higher TRAINING error (ruling out overfitting) while gradients are healthy (ruling out the vanishing story); the shortcut must stay a clean identity wire — added gates (as in Highway Networks) or projections on it should survive only if they clearly beat identity under matched conditions, since they otherwise obstruct the very path that makes it work, while dimension-change points still require a minimal projection.
```

## fact-check changes

```plaintext
- (a) Removed the pre-committed prediction framed as a fixed rule ('should (a)... (b)...') and softened 'far easier'/'always' to plausible expectations decided by reading the actual curves; kept it as a hypothesis with a decision rule rather than an asserted outcome.
- (b) Named the true contemporaneous competitor for the gated-shortcut alternative (Highway Networks) instead of leaving it generic/unnamed, and noted a gate that can close reintroduces the identity burden.
- (d) Softened 'gradient can always flow straight through' to 'can flow straight through'; changed 'infinite depth is free' to 'arbitrary depth is free' and clarified projections apply only where dimensions change.
```