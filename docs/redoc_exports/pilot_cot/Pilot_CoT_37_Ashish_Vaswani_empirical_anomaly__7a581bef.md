# Pilot CoT 37 — Ashish Vaswani · empirical_anomaly

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例1：RNN 的"顺序瓶颈" → 整块删掉 recurrence，把并行化当第一性

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **173** · assistant(cot) **1075** · whole **1305**

- fact-check changes: 6

## user (setup)

```plaintext
It's early 2017. Sequence transduction for machine translation is dominated by RNN/LSTM/GRU encoder-decoders, usually with a Bahdanau-style attention layer bolted on between encoder and decoder to help alignment. The recurrent backbone is treated as non-negotiable: a sequence is a sequence, so of course you process it step by step, hidden state h_t depending on h_{t-1}. The community's optimization energy goes into memory tricks, better cells, kernel fusion, deeper stacks. You are staring at your own training logs on a GPU cluster and something bothers you: the hardware is barely breathing during the recurrent forward/backward, and the standard explanation ('sequences are inherently sequential, that's just how language is') feels like an excuse rather than an analysis. Propose a genuinely novel idea for the architecture.
```

## assistant (cot)

```plaintext
Let me start from the thing I can actually measure, not from a theory. I'm Ashish Vaswani, and what won't leave me alone is the GPU utilization trace. We have this enormous parallel matrix-multiply machine, and during recurrent training it sits mostly idle, stepping through time. The usual story people tell is 'it's memory-bound' or 'we need better kernels.' So we do the honest thing you do when you suspect the story: we attack the surface variables. Mixed precision, fused ops, smarter memory reuse. Throughput creeps up and then flattens. That flatline is the observation that won't fit. If the wall were memory or kernels, chipping at them should keep paying. It doesn't. Something structural seems to be capping us before the hardware does.

So I ask: what is actually forcing serialization? It's not the arithmetic. It's the dependency chain. h_t literally cannot be computed until h_{t-1} exists. That's the bottleneck variable. Not FLOPs, not GB, but the *minimum number of sequential steps* baked into the graph. And once I name it that way, a second known symptom lines up with it: long-range dependencies. For two words n positions apart to influence each other, signal has to walk O(n) hops down the recurrence, degrading along the way. Both my utilization problem and the long-distance-dependency problem trace back to the same culprit. That's the tell that I may have found the load-bearing wall, not a decoration.

Now, the default move here is additive: keep the recurrence, add gating variants, add more attention on top, add tricks. But attention is already in these models as a patch between encoder and decoder. What if the patch is the whole point? What if I do the subtractive thing that feels reckless — throw the recurrence away entirely and see if it gets *worse*? People will say a sequence model without recurrence can't possibly know order; the intuition that 'attention without recurrence is enough' is genuinely contentious in the room. Fine. Skepticism isn't a number. I want to convert this from an aesthetic argument into a table I can be wrong on publicly.

So I lay out three columns for any candidate layer: total compute per layer, the count of sequential operations (the parallelism metric I care about), and the maximum path length between any two positions. Self-attention: O(1) sequential steps, O(1) path length, at a cost of O(n^2 * d) compute. Recurrence: O(n) sequential, O(n) path. The trade is stark — self-attention pays in dense matrix multiplication, which is precisely what the GPU is starving for, and buys back both parallelism and short paths. And when n < d, as is common in translation, it isn't even more expensive per layer than recurrence. That table, not any theorem, is what would license deleting recurrence — the argument is empirical: my idle GPU plus this complexity accounting.

Deleting has a cost I must not ignore — pure attention is permutation-invariant, so I've thrown away word order along with the recurrence. The disciplined fix is the cheapest possible one that restores exactly the lost capability, not a new complex module. A positional signal added to the inputs does it. I'll also keep the scoring itself the dumbest parallelizable form (dot products, tamed with a single scaling scalar so the softmax doesn't saturate as dimension grows) because this operator is going to be stacked many times.

How do I keep myself honest? I run it as subtraction and let the numbers decide, with a fast iteration loop so 'change-run-compare' is cheap. Decision rules, no predicted outcomes: try learned positional embeddings vs. the fixed sinusoidal — if learned isn't clearly better, keep the simpler sinusoidal on parsimony (sinusoids also carry the hope, untested at this point, of extrapolating to longer sequences than seen in training). Try single-head vs. multi-head attention — keep extra heads only if they clearly help. And the master test: strip pieces out one at a time; anything whose removal doesn't clearly hurt is a candidate to drop. If removing recurrence makes translation quality collapse at matched compute, the bet is simply wrong and I revert. If the idle-GPU observation is really telling me the truth, the leaner model should train faster; whether it reaches at least as far in quality is exactly what the experiment has to decide.

**Core idea:** The real bottleneck in sequence models is the sequential dependency itself, not compute or memory; so delete recurrence and convolution entirely, build the backbone from self-attention (dense, parallel matrix multiplication with O(1) sequential steps and O(1) path length), and restore the one lost capability — word order — with a cheap positional signal.
**Non-trivial crux:** Removing recurrence silently removes order-awareness because attention is permutation-invariant; the deletion only works if that side effect is explicitly compensated by the simplest sufficient means, and the whole case rests on a measured complexity/parallelism trade-off (O(1) vs O(n) sequential steps), to be decided by subtractive ablations, not by a proof or by a predicted winning score.
```

## fact-check changes

```plaintext
- (a) Removed asserted numeric result 'Throughput creeps up a few percent' — softened to observed flattening without fabricated magnitude.
- (a) Reframed the closing claim from an implied confirmed 'should reach at least as far' into an open question the experiment must decide.
- (a) Changed 'That table... is what licenses deleting recurrence. The forcing is empirical' to conditional 'would license' to avoid narrating a settled outcome before the runs.
- (c) Downgraded 'attention without recurrence is enough' from a settled framing/anecdote (Jakob's linguist father) to marked contemporaneous intuition/contention, removing the unverifiable personal anecdote.
- (c) Marked the sinusoidal long-sequence-extrapolation rationale as an untested hope rather than established fact.
- (d) Added caveat 'as is common in translation' rather than absolute 'as in translation' for the n<d condition.
```