# Pilot CoT 41 — Ashish Vaswani · first_principles

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例2：把注意力做成"可堆叠、可并行"的通用算子 —— scaled dot-product、multi-head 与 √d_k

- route: first_principles  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **216** · assistant(cot) **1116** · whole **1389**

- fact-check changes: 0

## user (setup)

```plaintext
It is early 2017. You are building sequence models for machine translation. Every strong system you know of — RNN, LSTM, GRU, and their attention-augmented variants — computes the hidden state at position t as a function of the state at t-1. This sequential chain is treated as inevitable: sequences are ordered, so of course you process them in order. Attention has appeared as a helpful add-on that lets a decoder look back over encoder states, but nobody treats it as the load-bearing structure itself. Suppose you become willing to entertain removing recurrence entirely and letting attention carry the whole burden. Reasoning as much as possible from what MUST be true — complexity per layer, how many operations must happen in strict sequence, how far a signal must travel between two positions — argue your way toward the design of a general operator that can serve as the main trunk of such an architecture: one that is safe to stack deeply, saturates the hardware, and stays numerically stable. Propose the genuinely novel idea and show why it is almost forced.
```

## assistant (cot)

```plaintext
Let me not start from any experiment. Let me start from invariants — quantities that must hold no matter what implementation I choose.

Fix three measures for any candidate layer that maps a length-n sequence to a length-n sequence. First, total compute per layer. Second, the minimum number of operations that must occur strictly in sequence — the serial depth I cannot parallelize away. Third, the path length between any two positions, i.e. how many composed steps a signal must traverse for position i to influence position j. These are not aesthetic; they are forced by the dependency graph of the computation. If I write them down for the recurrent layer, the verdict is immediate and unavoidable: serial depth O(n), maximum path length O(n). The O(n) serial depth is the wall. It is not a memory problem, not a kernel problem — it is a property of the dependency graph. h_t needs h_{t-1}. No amount of mixed precision or operator fusion touches that.

Now ask: what is the minimum achievable serial depth for a layer that lets every position attend to every other? If each output position is a function of all input positions computed simultaneously, serial depth collapses to O(1), and path length collapses to O(1) — any two positions are one hop apart. The cost I must pay for this is that comparing all pairs is O(n^2 * d) compute per layer. So the trade is exact and provable: I buy O(1) serial depth and O(1) path length by spending O(n^2) dense compute. And O(n^2 * d) of pairwise interaction is precisely a matrix multiplication — the single operation modern accelerators are built to eat. So the limiting argument doesn't just permit removing recurrence; it says the replacement should be a dense all-pairs interaction, because that is the only shape that pushes serial depth to the floor while landing on hardware's best form. This is Ashish Vaswani's instinct: the bottleneck variable is the serial chain, so delete it, and let the trunk fall onto dense parallel matmul.

Given that the trunk MUST be an all-pairs interaction, what scoring function? Deduce, don't taste. I want the score between a query and a key to be a matmul, because I already argued matmul is the target form. A dot product is exactly that. An additive/MLP-style scorer might be marginally more expressive on paper, but it does not reduce to a single highly-optimized matmul — so on my own criterion of saturating hardware, the dot product is forced, and the burden of proof is on anything fancier.

But now a stacking invariant bites. If this operator is to be the main trunk stacked many times, it must stay numerically stable at scale. Consider the variance of a dot product of two vectors whose components are roughly independent unit-scale: the variance grows linearly with the dimension d_k. Large d_k pushes the pre-softmax scores into the saturated regime where gradients vanish — and a trunk with vanishing gradients cannot be stacked deep. This is deducible before any run. The minimal fix consistent with 'cheap scalar over new mechanism' is to rescale the scores by 1/sqrt(d_k) so their variance returns to order one. One scalar tames the pathology; I refuse to swap in a whole new scoring machinery to fix a variance problem.

Next, expressivity. A single all-pairs map averages every kind of relation into one weighting — a capacity ceiling I can foresee. The parallel-first way to add capacity is not a bigger, more expensive single operator, but placing several lower-dimensional copies in parallel — project into h subspaces, attend independently, concatenate. This is again forced by the parallel-first premise: independent heads are embarrassingly parallel and let different heads specialize.

Finally, deletion has a side effect I must pay back honestly. An all-pairs interaction with no recurrence is permutation-invariant — I have thrown away word order. Rather than reintroduce a sequential module (which would resurrect the O(n) wall), inject order additively at the input via a parameter-free positional signal.

How do I keep myself honest? Each of these is a claim, not a result. So I subtract: run with a single head vs many, shrink d_k, swap learned positional encodings for the fixed signal. Decision rule: keep a piece only if removing it clearly hurts; if learned and fixed positional encodings are indistinguishable, keep the simpler parameter-free one on parsimony grounds. I do not know the numbers yet — the argument only tells me the design is nearly forced and exactly what to falsify.

**Core idea:** Because the true bottleneck is serial depth, replace recurrence with an all-pairs dot-product attention (the only O(1)-serial, O(1)-path form that reduces to dense matmul), stabilize it for deep stacking with a single 1/sqrt(d_k) scale, and recover capacity by running several low-dimensional attention heads in parallel — with lost word order paid back by an additive positional signal.

**Non-trivial crux:** The dot product isn't chosen because it's more expressive (additive scoring is arguably slightly better) but because it lands on hardware's best form; and it is only viable as a stackable trunk once the linear-in-d_k variance blow-up is neutralized by the 1/sqrt(d_k) scaling — miss that scalar and the deep stack simply won't train.
```