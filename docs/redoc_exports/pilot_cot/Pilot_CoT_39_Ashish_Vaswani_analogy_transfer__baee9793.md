# Pilot CoT 39 — Ashish Vaswani · analogy_transfer

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例1：RNN 的"顺序瓶颈" → 整块删掉 recurrence，把并行化当第一性

- route: analogy_transfer  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **162** · assistant(cot) **1034** · whole **1253**

- fact-check changes: 5

## user (setup)

```plaintext
It is early 2017. Sequence transduction — machine translation especially — is dominated by recurrent encoder-decoder models (LSTM/GRU), usually augmented with an attention mechanism that lets the decoder look back over encoder states to help alignment. These models work well but train slowly: because the hidden state at position t depends on position t-1, computation marches step by step along the sequence. People are pouring effort into making recurrence faster — mixed precision, kernel fusion, better memory reuse. You have been staring at throughput on modern accelerators that thrive on large dense matrix multiplications, and it bothers you that these chips sit half-idle. Attention already exists, but only as a bolt-on helper riding on top of recurrence. Propose a genuinely novel idea for how to build the model.
```

## assistant (cot)

```plaintext
Let me name the mechanism I keep coming back to: attention itself. Right now it lives as a subordinate module — Bahdanau-style content-based lookup between a decoder state and a bag of encoder states. It's an associative retrieval: given a query, take a similarity-weighted average over a set of keyed values. That's the source mechanism. My instinct, coming from statistical MT rather than deep learning orthodoxy, is to ask a transfer question I don't think the field is asking: what if this retrieval operation is not a helper but the *entire* computation of the layer? Not attention over an RNN — attention instead of the RNN.

So let me map it over explicitly and be honest about what transfers and what breaks. What transfers beautifully: the retrieval operation connects any two positions in a *single* step. In recurrence, for word i to influence word j, the signal must travel O(n) sequential hops and decays along the way. In an attention layer, every position queries every other position directly — path length O(1). And the operation is, at its core, a matrix of dot products followed by a softmax followed by a weighted sum. That is exactly the dense, highly parallel matrix multiplication the hardware is built for. This is the part I trust most, because it isn't aesthetic — I can write it down as three comparable numbers: total compute per layer, the minimum number of *sequential* operations, and the maximum path length between any two positions. Recurrence is O(n) sequential and O(n) path. Self-attention is O(1) sequential and O(1) path, paying O(n^2 * d) compute — but that compute is parallelizable, and when n is smaller than d, as in translation, it's often cheaper anyway. That table is my design instrument; I'll let the measured comparisons, not taste, settle it.

Now, what breaks when I lift retrieval out of its recurrent host. Two things the RNN was quietly doing get thrown away. First: order. A pure set-retrieval is permutation-invariant — it literally cannot see word order, and word order is most of what language is. Second: the recurrence gave the model implicit depth and nonlinearity between positions; a single averaging step is weak. I refuse to solve either by dragging recurrence back in through the side door — that would defeat the whole bet. For order, the cheapest possible patch: inject position directly into the inputs as a fixed, parameter-free signal — distinct multi-frequency sinusoids so each position gets a unique multi-scale fingerprint and, I suspect, some hope of extrapolating past training lengths. For weakness of a single retrieval, don't build one giant clever attention; run several low-dimensional retrievals in parallel — multiple heads in different subspaces — and stack many such layers with position-wise transformations in between. Everything I add must itself be parallel and stackable, because this operator is meant to be the load-bearing wall repeated many times.

One numerical worry: as the key dimension grows, dot products grow in variance and shove the softmax into saturation where gradients vanish. I'm reluctant to switch to an additive scoring function — additive may be marginally better in quality, but dot-product maps onto optimized matmul and that practical speed is the whole point. Tame it with one cheap scalar: divide the scores by the square root of the key dimension. A structural fix, not a new mechanism.

This cuts hard against the 2017 consensus that sequences *require* recurrence (or, in the convolutional line of work, still need many stacked layers to grow the receptive field) — a premise so ingrained that respected people, even computational linguists, doubt attention alone can carry translation. My discipline for staying honest is subtraction, not addition: build it with recurrence fully removed, then keep throwing components away to see how much worse it gets. The decision rule is strict — compare learned position embeddings against the fixed sinusoids and, if they're roughly matched, keep the simpler sinusoids (which also hold the promise of longer-sequence extrapolation); if extra machinery doesn't clearly help, delete it on parsimony grounds; and test single-head versus multi-head to see whether the heads are actually carrying weight. I, Ashish Vaswani, want every design choice checked against an ablation and every serial dependency compressed to O(1).

**Core idea:** Promote the attention retrieval mechanism from a bolt-on aligner to the sole computation of the layer — dispense with recurrence entirely and build the sequence model from stacked, multi-head, parallelizable self-attention — recovering the discarded word-order information with a cheap parameter-free positional signal added to the inputs.

**Non-trivial crux:** The winning move is deleting recurrence (turning O(n) sequential path into O(1) dense matmul the hardware loves), but it only works if you explicitly pay back what recurrence silently provided — position information via added encodings and depth via stacked multi-head layers — and stabilize the dot products with a simple scaling scalar rather than reintroducing any sequential machinery.
```

## fact-check changes

```plaintext
- (a) Changed 'The table decides this, not taste' to frame the compute/path table as a design instrument whose conclusions await measured comparison, not an already-decided verdict.
- (a) Rewrote the ablation decision rules from asserted outcomes ('if learned embeddings don't clearly beat', 'if single-head is nearly as good') into genuine tests-to-be-run with decision rules, since these results were not yet obtained at reasoning time.
- (a) Softened 'additive may be marginally better on paper' to reflect it as an expectation about quality vs. speed tradeoff rather than an established result.
- (b) Added the contemporaneous convolutional sequence-modeling line as an era-appropriate competing approach to receptive-field growth, alongside recurrence.
- (e) Minor: kept all text English (no non-English words present in draft body).
```