# Pilot CoT 50 — Ashish Vaswani · first_principles

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例5：把"通用计算原理"从架构延伸到公司 —— Essential AI 与开放科学

- route: first_principles  ·  case score: 6/10

- lengths (qwen2.5-32b tokens): user(setup) **173** · assistant(cot) **1237** · whole **1467**

- fact-check changes: 4

## user (setup)

```plaintext
It is early 2017. Sequence transduction — machine translation above all — is dominated by recurrent encoder-decoders (LSTM/GRU), often with an attention module bolted on top of the recurrence to help with long-range alignment. Everyone treats the recurrent backbone as non-negotiable: position t's hidden state must wait for t-1. Improvements come from bigger nets, better regularization, mixed precision, kernel fusion — engineering around a structure nobody questions. You suspect the real ceiling is not FLOPs or memory but something structural. You have a fast experimentation stack and a habit of asking what MUST be true of any layer that maps a sequence to a sequence. Reason, on paper, from invariants and limiting arguments, toward a genuinely novel backbone for sequence modeling. Do not appeal to experiments until the design is nearly forced.
```

## assistant (cot)

```plaintext
Let me not start from any existing model. Let me start from what a sequence layer must, at minimum, accomplish, and then ask what the cheapest structure satisfying those constraints looks like. I'm Ashish Vaswani, and my instinct is to name the bottleneck variable first, not to optimize the surface.

Write down the invariants a sequence-to-sequence layer must have. (a) Every output position must be able to depend on every input position — otherwise long-range dependencies are unrepresentable. (b) Whatever we build gets stacked many times and trained at scale, so its cost must fall onto operations hardware is best at. (c) It must be numerically safe to stack deep.

Now the deductive move: characterize any candidate by three quantities that MUST be well-defined for any layer. First, total compute per layer. Second, the minimum number of sequential operations — the length of the dependency chain that cannot be parallelized, no matter how many processors. Third, the maximum path length between any two positions — how many composed steps a signal traverses to get from position i to position j.

Work out these three for recurrence, purely on paper. Recurrence composes a function n times along the sequence; h_t needs h_{t-1}. So its minimum sequential-operation count is Theta(n) — this is a hard lower bound, an invariant of the structure, not an implementation wart. And the influence path between two distant positions is also Theta(n): the signal must be relayed step by step, and along an n-step chain of nonlinearities the gradient signal generically attenuates. So recurrence pays twice: it forbids parallelization within a training example, AND it lengthens the credit-assignment path.

Convolutions shorten the sequential count but the path length between distant positions still grows with the kernel structure — you need a stack of them, logarithmic at best (with dilation) or linear otherwise, to connect the far ends. Neither structure hits the ideal.

So ask the limiting question: what is the best achievable on those two path-related quantities? The theoretical floor for sequential operations is O(1) — a layer that connects all positions in a single, constant-depth composition. The floor for pairwise path length is also O(1) — every position directly reachable from every other. If such a layer exists at all, it dominates recurrence on exactly the two axes that were killing us.

Does a layer meeting both floors exist? A direct all-pairs interaction — each position forms a weighted combination over all positions, weights computed from content — gives path length O(1) and sequential depth O(1). Its price is O(n^2 * d) compute per layer, because it touches every pair. Here is the crucial point: that price is a dense matrix multiply. It is precisely the operation modern accelerators are built to saturate. So the O(1) sequential/path structure I want and the hardware-friendly dense form are the SAME object. I should be honest that the quadratic cost is a real liability when n grows large relative to d — for the sentence-length sequences of translation it's a favorable trade, but it is a trade, not a free lunch. Still, the invariant argument doesn't just permit attention as the backbone — it nearly forces it, once you insist on the O(1) floors and on landing compute in dense parallel form.

So the leap is subtraction, not addition: dispense with recurrence entirely and let this all-pairs mechanism carry the sequence computation itself, rather than sitting on top of an RNN. That cuts hard against the 2017 consensus that sequence models require recurrence.

But a deletion exposes what the deleted part silently provided. An all-pairs weighted sum is permutation-invariant — it cannot see order. I must add order back with the cheapest possible device, not a new recurrent module (which would re-introduce the very chain I removed). A parameter-free, deterministic positional signal added to the inputs is one candidate, and being a fixed function of position it might extrapolate to longer lengths; a learned positional embedding is the alternative. I'll treat these as things to compare rather than assume the fixed one wins — the decision rule is whether they differ meaningfully, and if not, prefer the parameter-free form for the extrapolation intuition.

How do I keep myself honest? Two paper-level risks, each with a decision rule. (1) Dot-product scores could grow with dimension and push the softmax into a flat, low-gradient regime — the fix I'd try is a single scalar normalization by sqrt of the key dimension, a repair, not a new mechanism, checked against the unnormalized variant. (2) A single all-pairs map may be too low-rank in what relations it can express — run it in parallel subspaces (multiple heads) and concatenate; the plan is to sweep head count and, if a leaner variant is within noise of a richer one, keep the leaner one on parsimony grounds. For every removable piece — heads, the positional scheme, the depth — my default action is to throw it away and measure whether it gets worse; if removing it doesn't clearly hurt, it wasn't a load-bearing wall. These are experiments to run, not verdicts I already hold.

**Core idea:** From the invariants that a sequence layer must connect all positions with minimal sequential depth and minimal pairwise path length, the optimal structure is an all-pairs content-based attention with O(1) sequential and path length whose O(n^2*d) cost is exactly a dense matmul; this forces removing recurrence entirely and re-injecting only what deletion loses (order) via a cheap positional signal.

**Non-trivial crux:** The Theta(n) sequential/path cost of recurrence is a structural invariant, not an implementation flaw, so no engineering removes it — but attention's advantage is bought with quadratic compute, which is only a win because that compute is a hardware-optimal dense multiply (and only when n is not too large relative to d) and because order, silently supplied by recurrence, must be explicitly restored.
```

## fact-check changes

```plaintext
- Toned down absolute framing of convolution path length to note dilation gives log and plain conv gives linear.
- Added the necessary caveat that quadratic compute is a real liability for long sequences, not an unconditional win (restored in both body and crux).
- Converted the positional-encoding choice from an asserted 'parameter-free suffices' claim into a comparison to be tested (fixed vs learned), reflecting that the paper found them near-equivalent and chose sinusoidal for the extrapolation intuition rather than a demonstrated win.
- Reframed the sqrt-normalization and head-count as experiments/decision rules to run rather than settled repairs, and made explicit these are tests not held verdicts (category a).
```