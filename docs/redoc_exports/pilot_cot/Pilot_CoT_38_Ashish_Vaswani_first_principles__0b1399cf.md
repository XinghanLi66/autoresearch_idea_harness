# Pilot CoT 38 — Ashish Vaswani · first_principles

- researcher: **Ashish Vaswani**  ·  style: 删recurrence·并行化第一性·消融即减法

- case: Ashish Vaswani 案例1：RNN 的"顺序瓶颈" → 整块删掉 recurrence，把并行化当第一性

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **178** · assistant(cot) **1069** · whole **1304**

- fact-check changes: 3

## user (setup)

```plaintext
It's early 2017. Sequence transduction — machine translation especially — is dominated by recurrent encoder-decoders (LSTM/GRU), usually augmented with an attention mechanism bolted between encoder and decoder to help alignment. The recurrent backbone computes hidden states step by step: the state at position t depends on the state at t-1. Everyone treats this sequential recurrence as the natural substrate for sequences, and most engineering effort goes into squeezing more out of it: better cells, mixed precision, kernel fusion, memory reuse. Yet training throughput on modern accelerators keeps disappointing, and very long-range dependencies remain hard to learn. You suspect the ceiling isn't compute or memory — it's something more structural. Reason, as much as possible on paper, from what MUST be true about these architectures, and propose a genuinely novel design direction. Do not assume the answer is known.
```

## assistant (cot)

```plaintext
Let me not start from experiments. Let me start from what is provably forced by the structure of these models, and see how far a deductive argument carries me before I ever touch a GPU.

I'll name the real variable. In an RNN, state h_t is a function of h_{t-1}. That is not an implementation detail — it is a data dependency. It *precludes* parallelization within a single training example along the sequence axis. No amount of kernel fusion or memory tricks can remove a genuine dependency in the computation graph; you can only make each serial step cheaper, never fewer. So if I optimize memory or operators, I'm polishing a wall I haven't questioned. The bottleneck is the serial chain itself, not the arithmetic. That's the variable to lock onto.

Now let me make the comparison purely quantitative, so this stops being a matter of taste. For any candidate layer I care about exactly three invariants: total compute per layer, the *minimum number of sequential operations* it forces (its serial depth, which is what hardware parallelism cannot escape), and the maximum path length between any two positions (how many hops a signal must travel for one token to influence another — because gradient signal degrades along long paths). These three numbers decide everything a stacked, reused operator must be good at.

Run the accounting. A recurrent layer: serial depth O(n), and path length between two arbitrary positions O(n) — every distant pair must relay through the whole chain. A convolutional layer shortens the path — stacked/dilated convolutions (the kind ByteNet and the recent convolutional sequence work explore) reach O(log_k n) — but still doesn't give you O(1) reach. Now imagine a layer where every position attends directly to every other position in one shot. Serial depth collapses to O(1). Path length collapses to O(1) — any two tokens are neighbors. The price: O(n^2 · d) compute per layer, because it's all pairs. But — and this is the crux of the ledger — that O(n^2 · d) is a dense, all-to-all operation, i.e. matrix multiplication, which is exactly the shape modern accelerators are built to devour. And when n < d, which is the common regime for translation, that per-layer cost is actually *below* the recurrent one. So I'm not even paying a real penalty in the regime I care about, and I'm buying O(1) serial depth and O(1) path length.

Once the table reads that way, the conclusion is nearly forced: the recurrence is not a load-bearing wall. It's the thing capping parallelism and lengthening every dependency path. So the bold move isn't to add a cleverer cell — it's to *delete recurrence entirely* and let attention be the backbone, not a patch. This cuts hard against the 2017 consensus that 'sequences require recurrence' — a premise even serious language people doubt you can drop. But my three invariants stand on its side, not the consensus's.

Deletion is only honest if I account for what the deleted part silently carried. Recurrence encoded order implicitly through its step index. Pure attention is permutation-invariant — it literally cannot see word order. So I must add order back, and by the same taste, with the *simplest* thing that works: inject a position signal directly into the inputs. A parameter-free family (multi-scale sinusoids) is the minimalist candidate; a learned table is the obvious alternative. I won't assert which wins — my rule is: try both, and if the learned version isn't clearly better, keep the parameter-free one on parsimony grounds, since it also plausibly extrapolates to longer sequences.

Two more paper-level checks before code. The all-pairs score via dot products will grow in magnitude as dimension grows, pushing softmax into saturation and shrinking gradients — so I expect to need a cheap scalar rescale (divide by sqrt of the key dimension), not a new mechanism. And a single attention may lack expressivity — so run several low-dimensional attentions in parallel and concatenate; it stays perfectly parallel. Decision rule throughout: throw a piece away and see if it gets worse. If removing something doesn't clearly hurt, it goes.

**Core idea:** Treat sequential recurrence as the true bottleneck (it fixes serial depth and dependency path length at O(n)) and delete it entirely, replacing the backbone with all-pairs self-attention that gives O(1) serial depth and O(1) path length at the cost of dense O(n^2·d) matrix multiplication — precisely the operation hardware parallelizes best — while restoring the deleted order information via an explicit, minimal position signal.

**Non-trivial crux:** The justification is a paper-level invariant comparison (compute / serial depth / path length), not an empirical hunch — but the O(1)-path advantage is only a net win when you (a) budget the O(n^2·d) all-pairs cost, tolerable chiefly when n is not much larger than d, and (b) explicitly add back the order information that pure attention, being permutation-invariant, cannot see.
```

## fact-check changes

```plaintext
- (b) Replaced the anachronistic/vague 'convolutional layer ... with dilation' framing with the true contemporaneous convolutional-sequence competitors (ByteNet / convolutional seq2seq), which are the era's O(log n)-path alternatives.
- (c) Clarified the softmax saturation reasoning: scores grow in magnitude (not just 'inflate in variance') and named the actual fix (divide by sqrt of key dimension) as intuition/anticipated need rather than a proven result.
- No fabricated experimental outcomes were present; the parsimony-based decision rule for position encodings was already correctly framed and preserved.
```