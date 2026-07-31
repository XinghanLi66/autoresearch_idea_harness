# Pilot CoT 46 — Ashish Vaswani · empirical_anomaly

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例4：SMT 出身的"简单快准"品味 + 一次走廊对话，逼出跨界组队

- route: empirical_anomaly  ·  case score: 7/10

- lengths (qwen2.5-32b tokens): user(setup) **188** · assistant(cot) **1127** · whole **1372**

- fact-check changes: 4

## user (setup)

```plaintext
It's early 2017. Sequence transduction for machine translation is dominated by recurrent encoder-decoders (LSTM/GRU), often with an attention mechanism bolted on top to help the decoder look back at the source. Everyone agrees attention helps. But the backbone is still recurrent: hidden state at position t depends on t-1, so training must march through the sequence step by step. The standard engineering effort goes into making that recurrence cheaper — better kernels, mixed precision, memory reuse — while attention stays an auxiliary. You have access to fast accelerators that love large dense matrix multiplies, and you keep noticing your GPUs sit underutilized during training on long sequences. The community's story is: recurrence is how you model order and long-range dependency; attention is a helpful add-on. You suspect that story hides a measurable contradiction. Propose a genuinely novel architecture idea for sequence transduction and justify it.
```

## assistant (cot)

```plaintext
Let me start not from a theorem but from a number that won't sit still: our accelerators can do enormous dense matmuls per second, yet during training our recurrent translation models leave them starved. People explain this away — "long sequences, memory-bound, that's just how RNNs are" — and go optimize kernels. I keep looking at the utilization trace and the wall-clock-per-epoch and thinking: that gap isn't an implementation wart. It's the structure telling on itself.

So I ask the question I was trained to ask by years of statistical MT, where I chased smaller, faster, accurate models against a jungle of hand-built pieces: what is the *bottleneck variable* here? Not FLOPs, not memory. It's the sequential dependency itself — h_t waits for h_{t-1}. Within a single training example, that ordering *precludes* parallelization along the sequence dimension. You can throw all the hardware you want at it; the critical path is O(n) sequential steps no matter what. That's the measured contradiction: the machine can parallelize, the model forbids it.

And there's a second symptom that the standard story also waves off. For two words far apart to influence each other, the signal has to travel O(n) recurrent hops, attenuating along the way. We add gating and attention to patch this, but we're patching a path-length problem we created.

Now, the default move is to keep recurrence as load-bearing and make attention the helper. Let me invert it and just *stare at the ledger*. Put the candidate operations on a table with three honest, comparable columns: total compute per layer, number of sequential operations (minimum, because that's what can't be parallelized away), and the maximum path length between any two positions. Self-attention: sequential steps O(1), path length O(1), at a cost of O(n²·d) compute. Recurrence: sequential O(n), path O(n), compute O(n·d²). The recurrent O(n·d²) looks cheaper on paper, but it's O(n) *serial* — exactly the shape the hardware cannot eat. The self-attention O(n²·d) is a big dense matmul — exactly what the hardware is best at. So the quadratic isn't a liability; for the sequence lengths that matter — where n is smaller than d — it's compute the machine actually wants. (I should be honest that when n grows much larger than d this crossover reverses; for typical translation sentence lengths it holds.)

That table is the forcing function. It says: the recurrence isn't a load-bearing wall, it's the wall blocking the door. So do the subtraction — dispense with recurrence entirely, and convolutions too (the contemporaneous convolutional seq2seq work being the main non-recurrent alternative), and see if it gets *worse*. My instinct, from being burned by complexity before, is that throwing pieces away often makes things better because you were carrying noise.

But deletion has a price I must pay back honestly. Kill recurrence and the model becomes permutation-invariant — it can no longer see word order. I refuse to reintroduce a complicated module to fix this. The minimal repair: inject position directly into the inputs with a fixed, parameter-free positional signal (I'll try sinusoids of varying frequency). Cheapest thing that restores the deleted capability. Decision rule: I'll also try a *learned* positional embedding and compare quality; if learned gives essentially the same quality, I keep the simpler parameter-free version on parsimony grounds — and because fixed sinusoids might extrapolate to longer sequences than seen in training. No pre-decided winner; prefer simplicity when the numbers don't clearly separate.

On the scoring function: additive attention may be marginally better in theory, but dot-product attention is a matmul — faster, more memory-efficient, more parallel. For a backbone I'll stack many times, parallelizability wins unless quality clearly suffers. One numerical failure mode I anticipate: as key dimension grows, dot products inflate variance and push softmax into saturation, killing gradients. The reasoning suggests fixing it with a single scalar — scale by √d_k — not a new mechanism, and I'll verify the ablation shows scaling helps at larger d_k. And to recover expressiveness lost by simple dot products, run several low-dimensional attentions in parallel and concatenate; that's more heads, still one big parallel op.

How I keep myself honest: subtraction ablations as the daily move — remove heads, remove positional encoding, remove scaling — and watch which removals hurt. The load-bearing pieces announce themselves by making things worse when gone. Whatever survives every deletion is the essence.

**Core idea:** The bottleneck in sequence models isn't compute or memory but sequential dependency itself; delete recurrence and convolution entirely, build the backbone from parallelizable self-attention (dot-product, √d_k-scaled, multi-head), and pay back the lost word-order with a minimal positional signal added to the inputs.

**Non-trivial crux:** The O(n²) cost of self-attention is a *feature*, not a defect for typical translation lengths — it trades an O(n) serial critical path for O(1), turning the main computation into the dense matmul hardware most wants — but only if you explicitly restore the capabilities deletion removes (order via positional encoding, stable gradients via √d_k scaling), using the simplest repair the ablations justify.
```

## fact-check changes

```plaintext
- (e) Replaced Chinese phrase '病灶' with English 'failure mode'.
- (a) Removed the pre-decided framing on positional encoding/scaling so the ablation outcomes are described as tests with decision rules, not asserted results; added that scaling would be verified by ablation and noted the learned-vs-fixed comparison plus the extrapolation rationale.
- (b) Named the contemporaneous non-recurrent alternative (convolutional seq2seq) rather than leaving convolution deletion unmotivated.
- (d) Added the caveat that the n²·d vs n·d² crossover reverses when n greatly exceeds d, restoring the real edge case.
```