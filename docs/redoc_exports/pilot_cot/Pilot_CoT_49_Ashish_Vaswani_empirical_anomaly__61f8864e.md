# Pilot CoT 49 — Ashish Vaswani · empirical_anomaly

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例5：把"通用计算原理"从架构延伸到公司 —— Essential AI 与开放科学

- route: empirical_anomaly  ·  case score: 6/10

- lengths (qwen2.5-32b tokens): user(setup) **174** · assistant(cot) **1059** · whole **1290**

- fact-check changes: 3

## user (setup)

```plaintext
It's early 2017. Sequence transduction for machine translation is dominated by recurrent encoder-decoders (LSTM/GRU), usually with an attention mechanism bolted on top of the recurrence. The field's collective effort is going into making these recurrent stacks faster and cheaper: better kernels, mixed precision, memory reuse, more clever gating. Attention is treated as a helpful accessory that lets the decoder peek at encoder states, not as the main engine. You have access to fast GPUs and a training pipeline that lets you iterate quickly. When you actually stare at wall-clock training curves and utilization, something nags at you: the hardware is not the wall. Propose a genuinely new way to think about, and possibly restructure, sequence models — driven by what the measurements are actually telling you rather than by the standard story that 'sequences need recurrence.'
```

## assistant (cot)

```plaintext
Start where the data refuses to cooperate. I keep looking at training throughput on these recurrent models and the GPU utilization is embarrassing — the chip is mostly idle inside a single training example. The standard story says: we're compute-bound, we're memory-bound, buy bigger cards, fuse the kernels. But the numbers don't fit that story. If I were truly compute-bound the utilization would be high and steady. Instead I see a machine that's starving. So the honest question isn't 'how do I feed the beast more,' it's 'what is keeping the beast from being fed?'

When I trace it, the answer is uncomfortable and structural: h_t depends on h_{t-1}. Every timestep waits on the previous one. Within one training example the sequence dimension is a strictly serial chain — O(n) sequential operations that no amount of kernel fusion can flatten. That is the wall. It's not the arithmetic; it's the *ordering* of the arithmetic. People have been polishing a wall instead of asking whether the wall needs to exist. I want to name this precisely, because I distrust aesthetic arguments: the bottleneck variable is the number of sequential steps, not FLOPs and not memory.

So let me quantify the choice instead of arguing taste. Lay the candidate primitives — recurrence, convolution, self-attention — on three hard, comparable axes: total compute per layer; the minimum number of sequential operations (parallelism); and the path length a signal must travel between any two positions (long-range dependency). Recurrence is O(n) sequential and O(n) path — both bad on exactly the axes I care about. Self-attention connects every position to every other in O(1) sequential steps and O(1) path length; its price is O(n²·d) compute per layer. And here's the thing the standard framing gets backwards: that O(n²·d) is not a penalty — it's dense matrix multiplication, which is *precisely* the shape the hardware is built to devour. I would be trading a serial chain the GPU hates for a batched matmul the GPU loves. On the axis that's actually limiting me, that looks like a clean win — though I should be honest that for very long n the quadratic term will eventually dominate, so the advantage is real in the regime where sequence lengths are moderate relative to d, which covers translation.

Now the leap, which cuts against everyone: if attention already gives O(1) mixing and full parallelism, what is the recurrence *for*? The instinct in the field is additive — keep the LSTM, add attention on top. My instinct is subtractive: throw the recurrence away entirely and see if it gets *worse*. If removing a load-bearing-by-assumption component doesn't hurt — or helps — then it was never load-bearing; it was overhead and inductive noise. This is my default move: delete a piece and watch the metric. 'Attention is all you need' is a subtraction claim, not an addition claim.

But deletion has a cost I must pay back honestly. Kill recurrence and the model becomes permutation-invariant — it literally can't see word order, because attention is a set operation. So the discipline is: identify exactly the capability I removed, and restore it with the *cheapest* possible mechanism, not a new complicated module. Inject position directly into the inputs — a fixed, parameter-free positional signal is the minimal patch. I'd also want to compare against a learned version; the plan is to run both and, if they perform about the same, keep the simpler fixed form for parsimony (and because it might extrapolate to longer sequences).

How I keep myself honest, without pretending I already know the outcome: I make ablation the daily action, not a pre-publication chore — which means first investing in an iteration loop fast enough to strip parts off one at a time. Compare the dot-product score (cheap, matmul-friendly) against additive attention; additive may look competitive on paper, but if dot-product is close I'd take it for the hardware win — and since large dimension can push the dot products into regions where the softmax saturates, add a 1/√d_k scaling to counter that. Test single-head versus multi-head, fixed versus learned positions, and the whole thing with and without recurrence. The rule throughout: prefer the smaller, faster, more parallel form unless a component *earns* its place with a number. Let the measured contradiction — idle hardware under a serial chain — do the forcing, not a theorem.

**Core idea:** The limiting variable in sequence models is sequential-operation count, not compute; so delete recurrence and convolution entirely, build the trunk from self-attention (dense, parallel matmuls giving O(1) path length), and pay back only the lost signal — word order — with a minimal positional encoding.

**Non-trivial crux:** Removing recurrence silently removes order-awareness and can push the attention softmax into saturation as dimension grows; the design only works if you explicitly restore position (cheaply) and rescale the dot-product scores — subtraction must be matched by the minimal necessary compensation, or the good idea dies looking like a failure.
```

## fact-check changes

```plaintext
- Removed the fabricated confirmed-outcome framing for the compute-axis comparison ('That's not a wash; on the axis... a clean win') by restoring the real caveat that self-attention's O(n²·d) advantage holds only when n is moderate relative to d (the translation regime), matching the actual paper's noted edge case.
- Softened the learned-vs-fixed positional decision from an asserted decision-rule outcome into a to-be-run comparison, noting the historical basis (they performed comparably, fixed kept for simplicity/extrapolation).
- Reframed the dot-product vs additive attention comparison from an anticipated result into a planned test with a stated decision rule, and tied the 1/√d_k scaling accurately to the saturation motivation rather than presenting it as a foregone conclusion.
```