# Pilot CoT 44 — Ashish Vaswani · first_principles

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例3：消融即减法 —— "把部件一件件扔掉看会不会更差，结果反而更好"

- route: first_principles  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **183** · assistant(cot) **1120** · whole **1360**

- fact-check changes: 4

## user (setup)

```plaintext
It is early 2017. The dominant recipe for sequence transduction (machine translation, language modeling) is an encoder-decoder built on recurrent networks (LSTM/GRU), often augmented with an attention mechanism that lets the decoder look back at encoder states. The research momentum is toward adding: stacking more recurrent layers, gating tricks, convolutional encoders as faster alternatives, better regularization. Everyone treats the recurrent backbone as the load-bearing structure and treats attention as a helpful add-on. You suspect the real constraint on scaling these models is not FLOPs or memory but something structural in how information is forced to travel through the network. You are asked to design a sequence architecture and to justify your structural choices not by taste but by what MUST be true about computation and information flow. Propose a genuinely novel backbone and argue for it from first principles, largely on paper, before running the decisive experiments.
```

## assistant (cot)

```plaintext
Let me refuse, for a moment, to ask 'what should I add?' and instead ask what is forced by the structure I already have. I am Ashish Vaswani, and I want to reason about the recurrent backbone as a hypothesis to be checked, not a premise.

Start from an invariant. In any recurrent layer, the hidden state at position t is a function of the state at t−1. That is not an implementation detail—it is a hard dependency in the computation graph. So the minimum number of sequential operations to process a length-n sequence is Θ(n), no matter how much hardware I throw at it. This is a lower bound I cannot engineer away with mixed precision or kernel fusion. The GPU is a machine that is spectacular at dense parallel matrix multiplication, and I am handing it a computation whose critical path is inherently serial in n. The bottleneck is not compute; the bottleneck is the serial dependency itself.

Now a second invariant, about information, not time. Ask: for two positions i and j, how many composed operations must a signal traverse to let i influence j? In a recurrent chain it is Θ(|i−j|), up to Θ(n). Every hop is a nonlinear squashing; gradients and signal attenuate along an O(n) path. So long-range dependency is fighting the same enemy as parallelism—path length. Both pathologies have the same root: the sequential chain.

So let me set up desiderata I can actually put in a table and compare candidates on, deductively, before any BLEU number exists: (1) total computation per layer; (2) the minimum number of sequential operations, i.e. parallelizability; (3) the maximum path length between any two positions. Convolution shortens path length—dilated stacks give O(log_k n)—but still require stacking depth. Recurrence is O(n) on both the serial and path axes.

What operator makes columns (2) and (3) both O(1)? An operator where every position attends directly to every other position in a single step. Self-attention: each output is a weighted combination over all inputs, computed as one dense matmul of scores, one softmax, one dense matmul of values. Sequential depth O(1). Path length between any two tokens O(1). The price is O(n²·d) computation per layer—but that price is paid precisely in the coin the hardware is richest in: dense parallel matrix products. I am trading a serial chain I cannot parallelize for a quadratic amount of the exact work GPUs devour. That trade is the whole argument, and it is derivable on paper.

Given that, the creative leap is almost forced: dispense with recurrence entirely rather than bolt attention onto it. Keep attention as the load-bearing wall, not the decoration. That cuts against the 2017 consensus, where attention is an accessory to the recurrent core.

But deletion has consequences I must derive too. Remove recurrence and the operator becomes permutation-invariant—it can no longer see order. I must add order back with the cheapest possible mechanism, not a new complex module: inject a position signal into the inputs so the model can recover word order. A sinusoidal encoding of varying frequency is one candidate that costs no parameters; a learned embedding table is the other. Second, the dot-product scores: as key dimension grows, their variance grows with dimension, pushing softmax into saturation and shrinking gradients—so scale by a single scalar, √d_k, to hold the operator in a healthy regime. A scalar patch, not a new machine. Third, a single attention may lack representational capacity, so run several low-dimensional attentions in parallel—naturally parallel, and cheap.

How do I keep myself honest? By subtraction, not addition. Take each added piece—multi-head vs single-head, learned vs sinusoidal position encoding, the scaling—and remove or swap it, measuring the cost on a fast iteration loop. Decision rule: if a piece doesn't clearly earn its keep, drop it and keep the simpler form; if learned and sinusoidal position encodings come out effectively indistinguishable, I would prefer the parameter-free sinusoidal one on parsimony grounds (and because it might extrapolate to longer sequences), but which I keep is for the ablation to decide, not for me to assert now. The dot-product choice over an additive scorer I make on a stated criterion too: additive attention may be comparable or slightly better in quality, but dot-product runs as optimized matmul, so unless the quality gap is large, the parallel form wins.

**Core idea:** The recurrent chain is the true bottleneck—it forces Θ(n) sequential operations and Θ(n) path length between tokens—so delete recurrence entirely and make self-attention the backbone: O(1) sequential depth and O(1) path length paid for in dense, GPU-friendly matmuls, with word order, gradient stability, and capacity restored by the cheapest possible patches (positional signal, √d_k scaling, parallel heads).

**Non-trivial crux:** The justification is a two-axis argument—parallelizability AND maximum path length share the same root cause, the serial chain—and the quadratic O(n²·d) cost is not a flaw but the point, since it converts an unparallelizable serial dependency into exactly the dense parallel work hardware is best at; the caveat is that this favorable trade holds while sequences are not so long that n² dominates d.
```

## fact-check changes

```plaintext
- (a) Removed asserted parsimony outcome framing that implied the ablation result was known; reframed the learned-vs-sinusoidal preference as a stated tie-break rule to be adjudicated by the ablation, keeping the first-person 'I won't claim which wins' honesty.
- (a) Softened 'additive may be marginally better in theory' to 'may be comparable or slightly better in quality', matching the actual contemporaneous rationale (dot-product chosen for speed via matmul).
- (c) Marked the extrapolation-to-longer-sequences motive for sinusoidal encoding as an intuition ('might'), rather than a decisive contemporaneous argument.
- (d) Clarified the sinusoidal encoding as one of two candidates rather than presupposing the parameter-free choice; restored that the n² caveat is relative to d.
```