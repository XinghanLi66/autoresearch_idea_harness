# Pilot CoT 47 — Ashish Vaswani · first_principles

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例4：SMT 出身的"简单快准"品味 + 一次走廊对话，逼出跨界组队

- route: first_principles  ·  case score: 7/10

- lengths (qwen2.5-32b tokens): user(setup) **174** · assistant(cot) **1064** · whole **1295**

- fact-check changes: 4

## user (setup)

```plaintext
It's early 2017. Every strong sequence transduction model in machine translation is built on recurrence — LSTMs and GRUs, often with attention bolted on top as a helper. The recurrent backbone computes hidden states step by step: the state at position t depends on the state at t-1. Everyone treats this ordering as intrinsic to sequences. The community's optimization effort goes into memory reuse, kernel fusion, better gating variants, and mixed precision to squeeze throughput out of these serial models. You come from a statistical machine translation background, trained to prize models that are smaller, faster, and still accurate. You suspect the real ceiling isn't compute or memory at all, but something structural. Before running a single large experiment, reason on paper about what MUST be true of a sequence architecture, and propose a genuinely novel design principle for the backbone.
```

## assistant (cot)

```plaintext
Let me not start from what works empirically. Let me start from what must be true — reason from invariants until the design is nearly forced.

I'll name three quantities that any layer over a length-n sequence must pay, and treat them as a table I can compare candidates on, not as matters of taste. First: total computation per layer. Second: the number of operations that MUST happen in sequence — the serial depth that no amount of hardware can collapse. Third: the maximum path length over which any two positions must communicate to influence each other. These three are the load-bearing invariants. Whatever architecture I pick, it lives or dies on where it lands in that table.

Now put recurrence into the table. A recurrent layer has serial depth O(n) by construction — h_t provably cannot be computed until h_{t-1} exists. That's not an implementation wart I can engineer away; it's a data-dependency, an invariant of the definition. And the path between position 1 and position n is also O(n) hops, so a gradient or a signal between distant tokens must survive a chain that long. So the thing people spend months optimizing — memory, kernels — is polishing a wall while the real wall, the O(n) serial chain, is never touched. I refuse to optimize the surface quantity. Lock onto the true bottleneck: the sequential dependency itself. (I should be honest that recurrence isn't the only game here — convolutional sequence models exist too, and they already cut serial depth; but their path length between distant positions still grows with distance, so they don't escape the third invariant cleanly.)

Here is the deductive leap. Ask: is there an operation that lets every position look at every other position with serial depth O(1) and path length O(1)? There is — if a position's new representation is a weighted combination over ALL positions computed at once, the mixing is a single dense operation, not a chain. That's attention used as the backbone, not as a helper. The price the table shows is honest: O(n^2 · d) computation per layer instead of O(n · d). But that price is exactly a dense matrix multiply — the one shape modern accelerators are built to eat. So I'm trading O(n) unavoidable serial steps for O(n^2) parallelizable arithmetic. On paper that is the trade you WANT: convert an irreducible serial cost into a parallelizable one. So the recurrence isn't just replaceable — the invariants say it's the thing standing in the way. The intuition pushing me is to dispense with it entirely, though I have to concede the quadratic term is a real liability for very long sequences.

Now, the discipline of deletion: what capability leaves with recurrence? A pure set-of-positions operation is permutation-invariant — it provably cannot see order. That's a real subtraction, not an inconvenience. The minimal repair is to inject position as a signal added to the inputs — a function of position added to the embeddings — so order is restored without adding a serial component. I can try a fixed, parameter-free encoding or a learned one; the cheapest fix that preserves the O(1) properties is what I want, and I shouldn't reintroduce a chain to patch a chain.

One more invariant to respect if this is to be a backbone stacked many times: numerical stability under scaling. A raw dot-product score has variance growing with dimension, which pushes softmax toward saturated regions and shrinks gradients. That points to a single scalar normalization rather than a fancier scoring mechanism. And where a single mixing may lack expressive capacity, the parallel-first move is to run several low-dimensional mixings side by side — still one parallel op, no added serial depth.

How do I keep myself honest before believing any of this? I don't get to claim it wins. I commit to subtractive ablations: build it, then remove or vary pieces — the scaling scalar, the number of parallel heads, the learned-vs-fixed position signal — and measure. Decision rule: if a piece can be deleted without clear degradation, delete it; if two variants tie, keep the simpler, parameter-free one on parsimony grounds. The title-worthy claim, if it holds, is a subtraction: not 'we added attention' but 'attention is the only backbone you need.'

**Core idea:** Treat serial depth and inter-position path length as the true bottleneck invariants; since recurrence forces both to O(n) while all-to-all attention makes both O(1) at the cost of a hardware-friendly O(n^2·d) matmul, delete recurrence entirely and build the backbone from attention, repairing only the provably-lost capabilities (order via added positional signal, stability via scalar scaling) with the cheapest possible fixes.
**Non-trivial crux:** The O(n^2) cost is not a defeat but the point — it converts an irreducible serial dependency into parallelizable dense arithmetic (with the caveat that the quadratic term itself becomes the limiting cost at long sequence lengths); and each deleted-or-added piece (scaling scalar, multi-head, positional encoding) must earn its place by subtractive ablation, keeping the simpler variant on ties rather than by aesthetic assertion.
```

## fact-check changes

```plaintext
- (c)/(d) Marked 'dispense with recurrence entirely' as intuition and restored the caveat that quadratic cost is a real liability for long sequences.
- (b)/(d) Added contemporaneous competitor class (convolutional sequence models) and noted their path-length limitation, avoiding the false framing that recurrence was the only alternative.
- (a) Softened deterministic claims about positional-encoding and stability fixes ('the fix is'/'cheapest possible') into what would be tried and tested, since the fixed-vs-learned choice was decided by ablation/parsimony, not asserted in advance.
- (d) Added quadratic-cost caveat to the Non-trivial crux.
```