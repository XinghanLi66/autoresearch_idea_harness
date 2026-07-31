# Pilot CoT 32 — 何恺明 Kaiming He · first_principles

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例4："实验前先预测结果"——用 pre-hoc/post-hoc 对抗"研究的过拟合"

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **203** · assistant(cot) **1170** · whole **1430**

- fact-check changes: 6

## user (setup)

```plaintext
It's 2015. Convolutional networks have been getting deeper — from 8 layers to 16, 19 — and depth clearly helps, up to a point. But something strange is happening: when people stack plain convolutional layers much deeper (say beyond ~20-30 layers), accuracy does not keep improving. It gets worse. The easy story is 'too many parameters, it overfits' or 'vanishing/exploding gradients, fix it with better initialization and normalization.' Normalization layers already largely tamed the gradient-magnitude problem, yet the deeper networks still lose. You have a strong suspicion this is not a mere engineering nuisance but points at something the field has quietly accepted as inevitable. Reason it out on paper first: what MUST be true about a deeper network relative to a shallower one, and what does the failure to achieve that tell you? Propose a genuinely novel idea for how a network should be structured so that adding depth cannot hurt.
```

## assistant (cot)

```plaintext
Let me not touch a GPU yet. I want to reason from what MUST be true, because if I jump straight into swapping initializations and activations I'll be chasing symptoms in a fog.

First, name the phenomenon precisely, and refuse the cheap explanation. People say 'deeper nets overfit.' But overfitting means low training error and high test error. What I actually see is that the deeper plain net has *higher training error* too. That single fact kills the overfitting story dead. If it can't even fit the training set as well as its shallower sibling, this is not a generalization problem — it's an optimization problem. So I'll set the label straight: this degradation is not caused by overfitting. Good. Now the problem is honest and worth attacking.

Next, the obvious remedies. Better init, normalization between layers — these address gradient magnitude. Suppose gradients are already well-scaled (they largely are, with normalization). Then why does more depth still hurt? So the vanishing-gradient framing, while real historically, is not the whole disease here. I need a sharper argument, not another knob.

So let me construct the contradiction. Take a shallower network that trains to some error. Now build a deeper network by copying that shallower one exactly and appending extra layers on top. Here is the key move: what if I *force* those extra layers to compute the identity function — output equals input? Then the deeper network computes exactly the same function as the shallower one. Therefore, by construction, there EXISTS a solution in the deeper network's parameter space that is at least as good as the shallower one. Not 'probably', not 'roughly' — it provably exists. The deeper net's optimum can only be lower-or-equal training error than the shallow net's.

And yet, empirically, optimization does not find it — the deeper net lands higher. So the fault is not representational capacity, and not a fundamental impossibility. The fault is that the optimizer cannot easily *reach* that constructed solution. In particular, it struggles to drive a stack of nonlinear layers to approximate the identity mapping. That's a strong, falsifiable claim, and it points precisely where to act.

Now flip it into a design. If the trouble is that plain stacked layers find identity hard to represent and hard to learn, then I should change the parameterization so that identity is the *easy* thing — ideally free. Consider the layer's job as fitting a target map H(x). Instead of asking the layers to produce H(x) directly, let them produce the residual F(x) = H(x) - x, and add x back with a plain wire: y = F(x) + x. Now, if the optimal thing to do is 'do nothing,' the optimizer just has to push the weights toward zero — driving F to 0 is plausibly easier than sculpting a nonlinear stack into identity. The constructed-identity solution that I proved must exist is now sitting essentially at the origin of the parameter space, near where SGD starts. I have removed the obstacle I identified, and nothing else. I should note this is a hypothesis about optimization ease — a preconditioning intuition — not a proof that residual form is always easier to optimize.

One related idea already in the air is gated shortcut paths (Highway Networks), which route information through learned gates. My instinct is to go the opposite way: make the shortcut a pure identity — zero parameters, no gate, no transform — because the whole point is an unobstructed path that carries x through untouched so that adding depth cannot degrade the reachable optimum. This cuts against the reflex to make the shortcut 'learnable' and fancier; my bet is that here, less is more, and I want to justify that on paper before any run.

How do I keep myself honest? I'll write down predictions before I look at numbers. Prediction 1: plain deep net has higher *training* error than a shallow one; the residual form should remove that gap and let training error keep falling with depth. Prediction 2: replacing the identity wire with 1x1 projections everywhere, or inserting gating/dropout on the shortcut, should be no better and possibly worse, because those re-obstruct the clean path. Decision rule: I'll run these ablations across depths and datasets; if a more complex shortcut doesn't clearly beat the plain identity, I keep identity on parsimony grounds — and if identity fails to fix the training-error degradation at all, my optimization-not-capacity diagnosis is wrong and I go back to the drawing board. I'll only trust these as pre-hoc evidence if they hold across depths and datasets, not on one lucky configuration. (Caveat: where input and output dimensions differ, the shortcut can't be pure identity and will need a projection — that's an accepted exception at stage transitions, not the common case.)

**Core idea:** Since a deeper network can always match a shallower one by having its extra layers compute the identity, the degradation is an optimization failure, not a capacity failure; reparameterize each block to learn a residual F(x) with a parameter-free identity shortcut y = F(x)+x, so that 'do nothing' costs zero and identity is the easy default.

**Non-trivial crux:** The shortcut should be a pure zero-parameter identity — my hypothesis is that adding projections, gates, or dropout on it re-obstructs the unimpeded path and undermines the very property (identity-is-free) that I think makes deep nets trainable; the diagnosis only holds because the degradation shows up in *training* error, proving it's optimization, not overfitting — to be confirmed by the planned ablations.
```

## fact-check changes

```plaintext
- Softened fabricated certainty that driving F to 0 is 'trivial'/definitely easier than identity into a stated optimization hypothesis (hindsight-as-original / over-absolutized).
- Marked the 'identity is free' preconditioning framing as intuition/hypothesis rather than proven contemporaneous fact.
- Replaced generic 'reflex to make shortcut learnable' with the true contemporaneous competitor (Highway Networks / gated shortcuts) instead of implying no named prior existed.
- Removed pre-asserted claim that complex shortcuts would be 'likely worse'; reframed Prediction 2 as an ablation to be run with a decision rule.
- Restored the necessary caveat that identity shortcuts require a projection when input/output dimensions differ (dimension-change exception).
- Softened 'monotonically-ish with depth' and other confident outcome phrasing to reflect pre-experiment status.
```