# Pilot CoT 57 — 苏剑林 Jianlin Su · analogy_transfer

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例2：追问"Attention 一定要有 Softmax 吗" → 用结合律把 O(n²) 降成 O(n)

- route: analogy_transfer  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **195** · assistant(cot) **1363** · whole **1615**

- fact-check changes: 4

## user (setup)

```plaintext
It's mid-2020. Transformers are everywhere but the O(n^2) cost of self-attention in both time and memory blocks long sequences (recall that for BERT-base the head dimension d is tiny, e.g. 64, while sequence length n is often much larger). Several people are attacking this: sparse-attention schemes that hand-design sparsity patterns and need custom kernels, LSH-based hashing that gets to O(n log n) with a complicated implementation, and low-rank projection tricks that squeeze the key/value length down to a fixed m. Each buys speed but sacrifices something—general token-to-token attention, or the ability to do autoregressive (causal) generation, or robustness when the sequence really is long. You want a genuinely novel angle: not another approximation heuristic, but a principled reformulation that gets attention to true linear complexity while keeping full token-token interaction and causal generation. What is the idea?
```

## assistant (cot)

```plaintext
Let me start where I always start: not with "which trick lowers the cost", but with "where exactly is the cost coming from, written as math". Everyone says the bottleneck is computing QK^T. I don't quite believe the framing. Look at Attention = softmax(QK^T)V. If there were no softmax, QK^T V is just three matrices multiplied, and matrix multiplication is associative. Q(K^T V): K^T V is a d×d object, tiny, computed once; then left-multiply by Q. That's O(n·d²), linear in n. So the O(n²) is not intrinsic to "attention"—it is forced on us by the softmax, because softmax acts on the full n×n score matrix and therefore *demands that we materialize* that n×n matrix. The real culprit is the softmax. Once I see that, the whole problem changes shape.

Now, I won't just crudely delete the softmax—inner products can be negative, and a weighting scheme with negative weights isn't a meaningful average. So let me reformulate the object itself. Write each output row honestly as a weighted average: output_i = (Σ_j w_ij v_j)/(Σ_j w_ij), where in standard attention w_ij = exp(q_i·k_j). The only property of exp I truly need for this to be a valid distribution is *non-negativity*. So replace exp by any non-negative similarity sim(q_i, k_j) ≥ 0. That's the reformulation: attention is "non-negative-similarity-weighted averaging". Nothing lost yet, and I've turned a fixed formula into a design space.

Here is the transfer I want to name explicitly. This is exactly the kernel trick from kernel methods / SVMs. In kernel methods a similarity that can be written as an inner product of feature maps, sim(x,y)=⟨φ(x),φ(y)⟩, is what lets you avoid forming the full Gram matrix. Map that over here: if sim(q_i,k_j) = φ(q_i)^T ϕ(k_j), then Σ_j φ(q_i)^T ϕ(k_j) v_j^T = φ(q_i)^T (Σ_j ϕ(k_j) v_j^T). The parenthesized Σ_j ϕ(k_j)v_j^T is d×d, computed once, reused for every i. Associativity fires—exactly the same associativity I lost to softmax, now recovered. So the exponential kernel was the thing standing between me and linearity, and a *factorizable* kernel restores it.

What transfers cleanly: the associativity payoff, the O(n) cost, the "feature map" viewpoint. What breaks / needs care: in kernel methods you're free to pick any PSD kernel, but here φ must map into the *non-negative* orthant, otherwise the weights aren't a valid average. That's the constraint the transfer imposes. So the design question becomes: choose non-negative feature maps. One natural choice keeps values positive with something like elu(x)+1 (this is the form proposed in the contemporaneous linear-attention-as-RNN work). Another choice is to (ab)use softmax itself but along *different axes* for Q and K so the product is automatically normalized—I've seen this floated as well.

But I'm not satisfied with "pick any non-negative kernel"—that's arbitrary, and arbitrary choices don't generalize or explain themselves. Let me derive one that is *faithful to the original softmax attention*. Taylor-expand: exp(q_i^T k_j) ≈ 1 + q_i^T k_j. To make 1 + q_i^T k_j ≥ 0 hold robustly, just ℓ₂-normalize q and k, so the inner product lives in [-1,1] and 1 + (q̂^T k̂) ≥ 0 always. This isn't literally of the form φ(q)^Tϕ(k) with a single non-negative map, but it is a first-order approximation to the true exponential kernel—so my intuition is it should sit closer to real scaled-dot attention than a hand-picked kernel. Derived, not guessed. Whether it actually helps in practice is an empirical question to settle later, not something I can assert here.

One more consequence I want to keep honest about generality: the low-rank/projection approaches can't do causal masking and their fixed m may not survive truly long sequences. My form does—just replace Σ_{j=1}^n with Σ_{j≤i}, and maintain running states S_i = S_{i-1} + ϕ(k_i)v_i^T, z_i = z_{i-1} + ϕ(k_i). Then output_i = φ(q_i)^T S_i / φ(q_i)^T z_i. That's a recurrence—linear attention *is* an RNN: O(1) memory per decode step, and trainable via cumsum. One caveat: the cumsum form makes causal training's memory grow with the S_i states, so the elegant O(1) really refers to inference-time decoding. Still, I haven't sacrificed autoregression.

How do I stay honest? Two decision rules. First, faithfulness: whatever φ I pick must keep full token-token attention and causal generation—if a variant quietly drops either, reject it regardless of speed. Second, on choosing among kernels: compare the 1+q̂^Tk̂ form against the elu+1 and axis-wise-softmax forms on the same task; if the more principled Taylor form doesn't clearly win, I don't get to claim it's "better", only that it's more interpretable—and I'd report that plainly.

**Core idea:** The O(n²) cost of attention comes from softmax forcing materialization of the n×n score matrix, not from attention itself; reformulate attention as non-negative-similarity-weighted averaging and, by the kernel trick, write the similarity as an inner product of non-negative feature maps so matrix-multiply associativity yields O(n)—with a Taylor-derived 1+q̂^Tk̂ similarity meant to stay faithful to the original softmax and a running-state recurrence showing linear attention is an RNN.
**Non-trivial crux:** The kernel transfer only works if the feature map is non-negative (so the weights remain a valid average)—that constraint is what the transfer costs you, and it's what separates a principled linear attention from a naive softmax-deletion that produces meaningless negative weights.
```

## fact-check changes

```plaintext
- (a) Removed the implied claim that the Taylor 1+q̂^Tk̂ form 'should be closer to real Scaled-Dot attention than any hand-picked kernel' as an established fact; reframed as intuition and added that its practical benefit is an empirical question to be tested later.
- (b/c) Attributed the elu(x)+1 non-negative feature map to the contemporaneous 'linear attention as RNN' work (Katharopoulos et al., 2020) rather than leaving it as a vague 'seen floated for vision' framing; softened the vision attribution.
- (d) Added the caveat that the O(1) memory claim applies to inference-time decoding, since causal-training cumsum requires maintaining the running states (memory grows accordingly).
- Minor: softened 'I don't believe the framing' to 'I don't quite believe' to avoid over-absolutizing.
```