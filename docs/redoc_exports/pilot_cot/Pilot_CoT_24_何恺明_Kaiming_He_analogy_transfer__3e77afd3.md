# Pilot CoT 24 — 何恺明 Kaiming He · analogy_transfer

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例1：从"深度退化"反常现象到无参数 identity 捷径

- route: analogy_transfer  ·  case score: 10/10

- lengths (qwen2.5-32b tokens): user(setup) **167** · assistant(cot) **1086** · whole **1310**

- fact-check changes: 5

## user (setup)

```plaintext
It's 2015. The prevailing belief is that deeper convolutional networks are strictly better, and every recent ImageNet gain has come from adding depth (roughly 8 to 16 to 19-plus layers). But when you naively stack plain convolutional layers past ~20, accuracy first saturates and then drops sharply. The easy story is 'too deep, so it overfits.' Meanwhile, other fields have their own tricks for training or propagating signal through many stages, and control-theory / signal-flow intuitions about 'carrying a state forward untouched' are floating around. You are convinced the depth ceiling is not fundamental. Propose a genuinely novel idea for what is actually wrong and how to fix it, so that networks many times deeper than today's can be optimized successfully.
```

## assistant (cot)

```plaintext
Let me start from an analogy that keeps nagging me, and be disciplined about it: what mechanism in other fields lets you push a signal through a very long chain of stages without it dying? In dynamical systems and in classic signal processing, the trick is that there's a *pass-through* path — a state you carry forward essentially unchanged, and each stage only adds a small perturbation on top. The chain stays healthy because the default behavior is 'do nothing,' not 'reconstruct everything from scratch.' That's the source mechanism I want to transfer into a deep net.

So let me map it over explicitly before I get excited. In a plain stack, every block must *actively re-produce* the full useful representation and hand it to the next block. There is no cheap way for a block to say 'leave it alone.' Now what transfers cleanly: the idea of a carried-forward state plus a small additive correction — a shortcut that lets the identity be the effortless default. What might break: unlike a hand-designed filter chain, here every stage is learned by SGD, so 'do nothing' is only useful if it's *easy to learn*. That's the crux I have to check, not assume.

Before committing, I have to kill the lazy explanation. Everyone says 'deeper = overfitting.' But I look at the training error — the deeper plain net has *higher training error*, not just worse test error. Overfitting would show the opposite. So this is not a generalization problem; it's an optimization problem. Good. The anomaly is now sharp and worth attacking. I'm Kaiming He, and I don't get to call something a deep problem until the cheap causes are ruled out. (I should note batch normalization is already in the recipe, so this degradation isn't simply vanishing/exploding gradients in the naive sense — another reason to look for a different explanation.)

Now let me force the vague 'it degrades' into a contradiction by construction. Take a trained shallow net and make it deeper by appending extra layers that are exact identity maps, copying the rest. By construction this deeper net has *no worse* training error than the shallow one. So a good-enough deeper solution provably exists. Yet SGD can't find it. Therefore the disease is precisely this: making a stack of nonlinear layers approximate the identity is hard for the optimizer. That single sentence points the whole design.

This is where the transferred pass-through path pays off, and where I refuse to over-engineer. The cleanest way to make identity the effortless default is a zero-parameter direct connection: let the block learn the residual F(x) = H(x) - x, and output F(x) + x. If the optimal map is near identity, the solver just has to drive F toward zero — plausibly far easier than sculpting identity out of nonlinearities. My intuition is that the shortest information path decides how optimizable the chain is, and here the shortest path can be zero layers long, straight through.

The honest test: I could dress this up with gating on the shortcut — Highway Networks with their learned gates are the contemporaneous approach to exactly this 'pass information through many layers' problem — or make every shortcut a learned 1x1 projection, or add other machinery. But Less is More is a discipline, not a slogan. My decision rule: keep only the simplest form unless a *fair* ablation shows the added machinery clearly helps. So I'll pit identity shortcut against 1x1-projection-everywhere against a gated variant, with everything else — depth, width, recipe — frozen to a single variable. My reasoning going in: anything that puts parameters or a gate *on the shortcut* could obstruct the clean pass-through — a gate can attenuate or even close the path — which would damage exactly the property I'm relying on. The rule is simple: if a gated or projection-heavy variant does not clearly beat the parameter-free identity shortcut, I keep the identity shortcut for parsimony; if a gated variant clearly wins, my 'identity-is-the-effortless-default' story is wrong and I must revise it.

Then scale it, because a two-layer toy proves nothing: verify from a few tens of layers up to over a hundred, on more than one dataset and task, and check whether the plain-vs-residual gap tracks depth as the theory predicts. This cuts against the consensus not by adding capacity but by removing an obstacle — the surprise, if it holds, is that the fix is subtraction, a free wire, not a cleverer layer.

**Core idea:** Transfer the 'carry the state forward, add only a small correction' pass-through mechanism into deep nets as a zero-parameter identity shortcut, so each block learns a residual F(x) with output F(x)+x — making 'do nothing' the easy default and turning the depth-degradation failure from an expressivity problem into a solvable optimization one.
**Non-trivial crux:** What transfers is the identity pass-through as the effortless default; what breaks if you're careless is putting anything (gates, projections, dropout) *on the shortcut itself* — that risks re-obstructing the very information path the mechanism depends on, so the shortcut should stay parameter-free unless a fair, single-variable ablation proves otherwise.
```

## fact-check changes

```plaintext
- (a) Removed the pre-hoc 'prediction' framing that gated variants 'should do worse' as if confirmed; replaced with the intended test and an explicit decision rule noting identity was kept for parsimony absent a clear win.
- (b) Removed anachronistic/off-era 'fractal-style' multi-branch reference; named Highway Networks (gated shortcuts) as the true 2015 contemporaneous competitor.
- (d) Softened 'provably has the same training error' to 'no worse'; softened 'far easier' to 'plausibly far easier' and scaling-verification 'as the theory says' to 'as the theory predicts'.
- (c/d) Added the batch-normalization caveat to explain why the degradation isn't simply naive vanishing gradients, restoring the real edge case that motivated the optimization framing.
- (c) Softened crux from absolute 'must stay parameter-free' to 'should stay ... unless a fair ablation proves otherwise' and 'risks re-obstructing'.
```