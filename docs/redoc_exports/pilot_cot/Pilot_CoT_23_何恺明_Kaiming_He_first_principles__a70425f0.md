# Pilot CoT 23 — 何恺明 Kaiming He · first_principles

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例1：从"深度退化"反常现象到无参数 identity 捷径

- route: first_principles  ·  case score: 10/10

- lengths (qwen2.5-32b tokens): user(setup) **167** · assistant(cot) **994** · whole **1218**

- fact-check changes: 6

## user (setup)

```plaintext
It is 2015. The prevailing belief is that deeper convolutional networks are strictly more powerful. Yet in practice, once you stack a plain conv net past ~20 layers, accuracy first saturates and then drops sharply. The easy story is overfitting, but you observe something disturbing: the deeper net has HIGHER TRAINING error too. Better initialization and batch normalization already let very deep nets start converging, so this is not simple gradient vanishing at initialization either. You have several candidate architectures on the table, some with multi-branch or gated connectivity. You want to understand, from first principles, what MUST be true about the solution space of a deeper network relative to a shallower one — and let that reasoning force a design, before you trust any experiment. Propose a genuinely novel idea.
```

## assistant (cot)

```plaintext
Let me refuse the lazy label first. Everyone says "deeper is worse — must be overfitting." But overfitting means low training error, high test error. Here the training error itself is higher for the deeper net. So overfitting is the wrong bucket. I strike it off. That single distinction changes the whole problem: this is not a generalization failure, it is an optimization/representability failure. Now the question is sharp enough to attack.

Let me reason about what MUST be true, on paper, before touching a GPU. Take a shallower net that trains well. Construct a deeper net by copying those learned layers and appending extra layers. If — and this is a pure existence argument — I set the appended layers to compute the identity mapping, the deeper net computes EXACTLY the same function as the shallower one. Therefore a solution exists, by construction, whose training error is no worse than the shallower net's. This is not empirical; it is forced. So the deeper model should be able to match, not exceed in error, the shallower one's training error.

So the contradiction is exact: solution space of deeper ⊇ solution space of shallower (identity padding), yet SGD lands on something strictly worse. The extra representational capacity is real; the optimizer simply cannot find the good region. That pins the disease precisely — it is not that the network can't express what we want, it's that the current parameterization makes the good solution hard to reach.

Now, why is it hard to reach? The constructed good solution requires several stacked nonlinear layers to jointly approximate an identity function. Asking a pile of conv+ReLU to reproduce x is apparently a difficult ask for the optimizer — it has to conspire many weights to cancel into a near-identity. If the optimum is close to identity, we are forcing the solver to hunt for it from a bad reference point (near zero mapping).

Here is the leap, and it is a re-parameterization, not a new module. Change the reference frame. Instead of asking the stacked layers to fit the desired mapping H(x), ask them to fit the RESIDUAL F(x) = H(x) − x, and reconstruct the output as y = F(x) + x. Now if the optimal H is close to identity, the optimizer's job is trivial: push F toward zero. Driving a stack of layers to output zero is plausibly easier than driving them to output identity — weight decay already pulls that way. The pathological case of my construction becomes, on this hypothesis, the EASY case. The re-parameterization costs zero parameters and zero extra compute: the +x is just a wire. (Caveat: this exact identity wire only holds when input and output dimensions match; where they change, I'll need a projection on the shortcut.)

Why does this cut against the room? People are reaching for MORE structure — extra branches, gated shortcuts like Highway Networks — to make deep nets trainable. My argument suggests the answer may be a subtraction: the shortest path through the network plausibly governs how easy optimization is, and the shortest path here can be length zero — a bare identity wire straight through. Anything I put ON that wire (a 1×1 conv, a gate, dropout) reintroduces a transform that can obstruct the clean signal, and may re-create the very difficulty I just removed. My rule: use the plainest structure — identity — unless an ablation proves added machinery is necessary.

How do I keep myself honest? I will not trust one number. Prediction, written before running: plain and residual nets, identical in depth/width/recipe except the shortcut, should diverge exactly on the degradation — residual should NOT get worse with depth. And a fair ablation of the shortcut itself: identity vs. all-1×1-projection vs. gated shortcut. Decision rule — if the fancier shortcut does not clearly beat identity, I keep identity on parsimony grounds; if any variant obstructs the wire and degrades, that supports the "unimpeded information path" reading. And I'll push depth to the breaking point to test whether the effect scales, rather than resting on a lucky depth.

**Core idea:** A deeper net provably contains a solution no worse than a shallower one (append identity layers), so degradation is an optimization failure; re-parameterize each block to learn a residual F(x) with a zero-parameter identity shortcut y = F(x)+x, making the near-identity optimum easy to reach.

**Non-trivial crux:** The decisive move is the constructive existence argument that isolates optimization from representability; and the shortcut should be a bare identity — putting any transform or gate on it may re-obstruct the very path that made optimization easy, so simplicity is not just a preference but a hypothesis to be tested and defended by ablation.
```

## fact-check changes

```plaintext
- (b) Replaced anachronistic 'fractal-like / fractal topologies' (FractalNet, 2016) with contemporaneous competitors: multi-branch/gated connectivity and Highway Networks.
- (a) Removed the framed-as-proven 'zero is easier than identity' and 'less is more, forced not aesthetic' assertions; recast as hypotheses tested by the stated ablation with its parsimony decision rule.
- (a) Softened 'confirms' to 'supports' for the ablation reading, since the result is not yet run.
- (d) Added the necessary caveat that the identity shortcut requires matching dimensions, otherwise a projection is needed.
- (c) Marked the 'shortest path governs optimization ease' framing as intuition/hypothesis ('plausibly') rather than an established decisive argument.
- Minor: reworded 'lower bound' phrasing to avoid overstatement while preserving the existence argument.
```