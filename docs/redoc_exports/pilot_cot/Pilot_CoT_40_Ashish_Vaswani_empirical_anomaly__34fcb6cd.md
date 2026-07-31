# Pilot CoT 40 — Ashish Vaswani · empirical_anomaly

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例2：把注意力做成"可堆叠、可并行"的通用算子 —— scaled dot-product、multi-head 与 √d_k

- route: empirical_anomaly  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **195** · assistant(cot) **1204** · whole **1456**

- fact-check changes: 6

## user (setup)

```plaintext
It's early 2017. You're working on machine translation. The dominant recipe for sequence models is recurrent (LSTM/GRU), sometimes with attention bolted on top as an alignment mechanism between encoder and decoder. Attention is widely regarded as a useful *supplement* to recurrence, not a replacement. You've been running many model variants on your training infrastructure, and you keep noticing something odd in your logs and wall-clock numbers: the recurrent backbone dominates your training time in a way that doesn't track compute or memory the way people assume, and your attention-heavy ablations keep behaving surprisingly well even as you strip things out. Most colleagues explain this away ('attention just helps a bit', 'you still need recurrence for order and long-range memory'). Suppose you take the measurements at face value instead. What genuinely novel architectural idea would you propose, and how would you make it hold up as a reusable, high-throughput backbone rather than a fragile trick?
```

## assistant (cot)

```plaintext
Let me start from the thing I actually see, not the story everyone tells. I've been throwing bits of models away just to measure how much worse they get, and the recurrent variants keep losing on wall-clock in a way that people keep hand-waving as 'implementation'. So I profile it honestly. The GPU isn't compute-bound the way the folklore says. The wall is elsewhere: position t's hidden state literally cannot be computed until t−1 exists. That's not a memory problem or a kernel problem. It's the structure. The sequential dependency precludes parallelization *within* a single training example. I'm staring at a machine that is world-class at dense parallel matmul, and I'm feeding it a chain that forces it to wait.

So I refuse to explain that away. The bottleneck variable isn't FLOPs or VRAM — it's sequential path length. Once I name it that way, the obvious moves (mixed precision, operator fusion, memory reuse) all look like polishing a wall that never needed to be there. Those buy me a few percent and then cap out, because none of them touch the O(n) serial chain.

Now the subtraction instinct kicks in. Recurrence is treated as a load-bearing wall of sequence modeling. I want to mark it as a *hypothesis*, not a premise. What if I delete it entirely — and convolutions too — and keep only attention? Let me not argue this on taste. Let me put the candidate mechanisms on a table with three hard, comparable columns: total compute per layer, the minimum number of *sequential* operations, and the maximum path length any two positions must traverse to influence each other. Self-attention: O(1) sequential steps, O(1) path between any two positions, paid for with O(n²·d) compute per layer. Recurrence: O(n) sequential, O(n) path. That O(n²·d) I'm 'paying' is exactly the dense parallel matmul the hardware wants. The O(n) serial cost of recurrence is exactly what no hardware can parallelize away. The table forces the decision; I'm not overruling it with aesthetics. (I should note the caveat honestly: when the sequence length n exceeds the representation dimension d, self-attention per layer stops being cheaper than recurrence — for the sentence lengths in translation, n < d, so it holds.)

But deleting a wall means I've silently dropped whatever it was carrying. Two things fall out. First, without recurrence the model is permutation-invariant — it can't see order. I don't want to answer that with another heavy module; I want the cheapest patch that restores the capability. Add position information directly to the inputs — a fixed sinusoidal encoding — and keep it. My decision rule: compare a learned positional embedding against the fixed sinusoidal one; if they come out about the same, keep the fixed one for parsimony, and because it may extrapolate to sequence lengths longer than those seen in training. I won't claim in advance which wins — I'll let the comparison decide, and lean fixed on the tie.

Second, if attention is now the *backbone* — stacked many times — it has to be parallel, numerically stable, and safe to stack. So for the scoring function I deliberately pick the plainest, most matmul-friendly form: dot products, rather than additive attention. Additive attention has comparable theoretical complexity and can be competitive in quality, but dot-product runs on highly optimized matmul and that throughput compounds across every layer and every scale. For a component that will be run billions of times, I take the form that saturates the hardware.

Dot products have a known failure mode I can reason about: as the key dimension grows, the variance of the dot product grows, pushing softmax into regions with vanishingly small gradients. I don't want a fancier mechanism to fix a numerical bruise — I want a scalar. Divide by the square root of the key dimension to counteract that variance growth and pull it back into a healthy gradient regime. Cheap patch, right size.

Then expressivity: one attention averages all relation types together. Instead of one big expensive head, run several low-dimensional heads in parallel over projected subspaces and concatenate — the intuition being that different heads can attend to different kinds of relations, and they're independent, which is more parallelism, not less. Whether they actually specialize cleanly is something to inspect after the fact, not something I'll assert up front.

Every one of these gets a subtraction ablation, not a vibe. Vary the number of heads — does single-head hurt, and do too many heads also hurt? Vary key dimension — does shrinking it cost quality? Swap fixed vs learned positions. The rule is uniform: if removing machinery doesn't clearly hurt, I keep the simpler form; if 'throw it away' makes things *better*, that's the signal I chase, because it means I was carrying noise, not a wall. I'm Ashish Vaswani, and the title I'd want is a subtraction claim — that attention is all you need — which only earns itself once the ablations, not my taste, have nailed each choice down.

**Core idea:** Treat the sequential dependency of recurrence as the real performance bottleneck and delete it entirely, making attention the parallel backbone — dot-product scoring rescaled by √d_k for stability, multiple low-dimensional heads for expressivity, and a cheap position signal to restore word order.

**Non-trivial crux:** The win is chosen on parallel throughput and sequential-path-length, not per-metric optimality — you deliberately accept O(n²·d) matmul and dot-product scoring because they saturate hardware, and you must explicitly repay what deleting recurrence dropped (order, numerical stability) with the *cheapest* patches, validated by subtractive ablation rather than added on faith.
```

## fact-check changes

```plaintext
- (a) Removed asserted claim that additive attention is 'theoretically slightly better'/'a little better'; restated as comparable complexity and competitive quality (the paper's actual framing), and reframed the fixed-vs-learned position choice as a comparison with a tie-break rule rather than a pre-decided outcome.
- (a) Reframed ablations (single head, key dimension) as tests with a decision rule rather than implying known results; noted the actual finding that too many heads can also hurt as a thing to test.
- (d) Restored the real caveat that self-attention is only cheaper than recurrence per layer when n < d, which holds for translation-length sentences.
- (c) Marked the head-specialization framing as intuition/post-hoc inspection rather than a guaranteed contemporaneous argument.
- (a) Tightened the variance/√d_k rationale to the mechanism actually reasoned about (variance growth pushing softmax into small-gradient regions) without overstating.
- (e) Removed non-English fragment in the case title framing (kept English technical terms).
```