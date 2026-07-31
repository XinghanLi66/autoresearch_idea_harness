# Pilot CoT 22 — 何恺明 Kaiming He · empirical_anomaly

- researcher: **何恺明 Kaiming He**  ·  style: LessIsMore·反常现象立题·极致控制变量

- case: 何恺明 (Kaiming He) 案例1：从"深度退化"反常现象到无参数 identity 捷径

- route: empirical_anomaly  ·  case score: 10/10

- lengths (qwen2.5-32b tokens): user(setup) **176** · assistant(cot) **992** · whole **1225**

- fact-check changes: 5

## user (setup)

```plaintext
It's 2015. The prevailing belief is that deeper convolutional networks are strictly better — more depth, more representational power, higher accuracy. Batch normalization and careful initialization have made it possible to train networks that were previously stuck. Yet when you actually stack a plain convolutional net past ~20 layers, something odd happens: accuracy saturates and then drops sharply. The convenient story on the table is 'overfitting — deeper models have more capacity, so of course they generalize worse.' You've measured carefully and something about that story doesn't sit right. You have a handful of candidate architectural fixes floating around (gated shortcut variants like Highway Networks, heavier initialization schemes, deeper-supervision tricks). Propose a genuinely novel idea that gets to the root of why deep networks degrade — and be precise about what you'd measure to know you're right.
```

## assistant (cot)

```plaintext
Start with the one number everyone is glossing over. When I push a plain net past ~20 layers, the accuracy degrades. The reflex explanation is overfitting. But I go look at the training error, not the test error — and the training error is *higher* for the deeper net. That single fact detonates the overfitting story. Overfitting means low training error, high test error. This is the opposite. So whatever this is, it is not a generalization problem; it is an *optimization* problem. Rule out the cheap explanation first, and only then do I believe I have a real problem worth my time.

Now I want to sharpen the fuzzy complaint 'deeper trains worse' into a contradiction I can attack. Let me build the offending solution by hand. Take a shallow net that trains fine. Make a deeper net by copying those layers and appending extra layers. If I set those extra layers to identity mappings, the deep net computes *exactly* the same function as the shallow one — same training error, by construction. So there always exists, in the solution space, a deep model that is no worse than the shallow model. And yet SGD can't find it in reasonable time. That's the contradiction, and it's sharp: the capacity is there; the optimizer just struggles to reach a solution it provably possesses. The disease looks like optimization difficulty, not lack of expressive power.

So where exactly is the difficulty? My working hypothesis: I'm asking a stack of nonlinear layers to *learn* the identity mapping, and that turns out to be hard — driving a pile of conv+ReLU to reproduce their input is an awkward target for the solver. So don't ask them to. Reparameterize. If the ideal function near optimum is close to identity, let the layers learn only the *deviation* from identity. Let them fit F(x) = H(x) − x, and add x back by a plain wire: y = F(x) + x. When identity is what's wanted, the solver just has to push F toward zero — and my intuition is that pushing a residual to zero is easier than synthesizing identity from nonlinearities.

Here's where I have to be disciplined and cut. There are fancier things on the table — gated or weighted shortcuts in the style of Highway Networks, learned 1x1 projections on every skip. My instinct is the opposite of adding: the shortest path through the network plausibly governs how easy it is to optimize, and the shortest path should be able to shrink to *zero* layers — a bare, parameter-free wire. Anything I put on that wire (a gate, a projection) re-introduces the obstruction I'm trying to remove: it stops information from passing through untouched. Highway's gates, for instance, can close and choke the shortcut. So the default answer is the identity shortcut: no extra parameters, no extra compute. Less is more here as a consequence of the argument, not just a slogan.

But I refuse to trust my own taste. This is where I keep myself honest with strict controls. Test one: plain net vs. residual net, identical in depth, width, and training recipe — the *only* difference is whether the shortcut exists. If the degradation vanishes when the wire is added, the optimization diagnosis is supported. Test two — the ablation that could embarrass me — put the variants side by side: identity wire vs. replacing every shortcut with a 1x1 projection vs. gated shortcuts. My hypothesis is that identity does at least as well and the extra machinery does not clearly help; the decision rule is parsimony — if the fancier shortcuts don't clearly beat the bare wire, I keep the bare wire, since I don't get to claim complexity earned its place unless the ablation forces me to. Test three — scaling: does the fix keep buying accuracy as I go far past where the plain net collapsed, and does it replicate across tasks and datasets, so it's not one lucky curve. If it holds only on my one setup, I don't believe it.

**Core idea:** Deep-net degradation is an optimization failure, not overfitting; since a deeper model provably *could* match a shallower one via identity layers but SGD struggles to find it, reparameterize each block to learn a residual F(x)=H(x)−x with a parameter-free identity shortcut y=F(x)+x, so approximating identity means merely driving F to zero.

**Non-trivial crux:** The whole argument rests on the shortcut being a clean, untouched identity path — the moment you add projections or gates onto it you reintroduce the very obstruction to information flow the residual reformulation was meant to remove; the necessary exception is dimension changes, where a minimal projection is unavoidable but should be used only there.
```

## fact-check changes

```plaintext
- (b) Removed anachronistic 'fractal-style multi-branch designs / FractalNet' (FractalNet is 2016) from both setup and cot; replaced with the true contemporaneous competitor, Highway Networks / gated shortcuts.
- (a) Removed the fabricated 'pre-registered prediction: identity is best, and adding machinery hurts' as an asserted outcome; reframed as a hypothesis with a parsimony-based decision rule, matching the historical rationale.
- (a) Softened 'SGD can't find it' / 'razor-sharp' to 'struggles to find it in reasonable time' to avoid over-absolutizing an empirical claim not proven.
- (c) Marked the 'residual easier than identity' framing as intuition/working hypothesis rather than an established fact at that moment.
- (d) Removed 'dropout on the shortcut' as a listed contemporaneous variant (not a standard shortcut variant of that era) to keep the ablation to the real alternatives.
```