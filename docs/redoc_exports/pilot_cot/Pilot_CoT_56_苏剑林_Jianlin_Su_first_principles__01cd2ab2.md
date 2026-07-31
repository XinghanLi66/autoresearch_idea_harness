# Pilot CoT 56 — 苏剑林 Jianlin Su · first_principles

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例2：追问"Attention 一定要有 Softmax 吗" → 用结合律把 O(n²) 降成 O(n)

- route: first_principles  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **198** · assistant(cot) **1249** · whole **1504**

- fact-check changes: 3

## user (setup)

```plaintext
It's mid-2020. Transformers scale beautifully in quality but the standard attention Attention(Q,K,V)=softmax(QKᵀ)V costs O(n²) in both time and memory, since Q,K,V are n×d with n far larger than d (a BERT-base head has d=64). Several groups are attacking long sequences: sparse-attention patterns needing hand-designed masks and custom kernels; LSH-based hashing schemes with O(n log n); low-rank projections that compress K,V to a fixed length m. Each buys speed by giving something up — restricted attention patterns, complex implementations, or the loss of full token-to-token interaction and autoregressive decoding. You are convinced most of these treat a symptom rather than the disease. Reason from first principles: what EXACTLY forces the quadratic cost, and can you derive an attention mechanism that is provably O(n) while keeping full token-token interaction and causal generation? Propose a genuinely novel construction.
```

## assistant (cot)

```plaintext
Let me refuse to start from "which trick makes it cheaper." That's the trap everyone falls into. Let me instead ask the sharp question: where is the O(n²) *forced*? Write out the shapes. Q,K,V are n×d. The output is n×d. Nothing about the input or output is quadratic. The only n×n object in the whole computation is QKᵀ. So the quadratic cost is not intrinsic to "attention" — it is intrinsic to *materializing that n×n matrix*.

Now, why must I materialize it? Because a softmax sits on top of it. softmax acts row-wise on the full n×n scores; I cannot apply it without the scores existing. But matrix multiplication is associative. If the operation were just Q(KᵀV), I could compute KᵀV first — a d×d object, tiny — and then left-multiply by Q, and the whole thing is O(nd²)=O(n). The associativity is the escape hatch, and the softmax is the padlock on it. So the true culprit is not "computing attention," it is the softmax nonlinearity that stands between the three matrices. That inversion is the whole game: the enemy is the softmax, not the matmul.

But I refuse to just delete the softmax. If I drop it, the weights become q_i·k_j, which can be negative — that is no longer a weighted average, it's not a valid attention. I want to keep what attention *is*, not mutilate it. So let me re-formalize the object rather than hack it. Rewrite one row honestly: the output at position i is a convex combination of the v_j, with weights proportional to exp(q_i·k_j). That is the invariant I must preserve: a *non-negative* similarity, normalized to a distribution, applied to V. Every token still attends to every token.

So abstract exp(q_i·k_j) to a general similarity sim(q_i,k_j), with the single hard constraint sim≥0, so the row still normalizes to a legitimate distribution. Now the deductive step: when can I recover associativity? Precisely when sim factorizes as an inner product of two non-negative feature maps, sim(q_i,k_j)=φ(q_i)ᵀϕ(k_j). Then Σ_j φ(q_i)ᵀϕ(k_j)v_j = φ(q_i)ᵀ(Σ_j ϕ(k_j)v_jᵀ), and the parenthesized d×d block is computed once and reused for all i. Linear again — forced, not fitted. And I notice: with φ=ϕ this is exactly a kernel, ⟨φ(q),φ(k)⟩ a positive-definite kernel. So "linear attention" is "kernelized attention with a factorizable non-negative kernel." That is the clean statement.

Now which φ? A contemporaneous line (Katharopoulos et al.'s "Transformers are RNNs") picked a convenient non-negative map, elu+1, to force positivity. That works, but it is somewhat arbitrary — pick a kernel and hope. My instinct says: don't pick a kernel, *approximate the real thing*. Taylor-expand exp(q_i·k_j) ≈ 1 + q_i·k_j. To keep it non-negative it suffices to ℓ₂-normalize q and k, so 1+q̂·k̂ ∈ [0,2]≥0. This is not literally of the form φᵀϕ, but it stays faithful to the original scaled-dot attention rather than substituting a foreign kernel — that faithfulness is the appeal, though I should be honest that it isn't an exact factorization the way elu+1 is.

One more thing I must not lose: causal generation. Sparse and low-rank tricks stumble here. But in my form, replace Σ_{j=1}^{n} by Σ_{j≤i}. Define a running state S_i=Σ_{j≤i}ϕ(k_j)v_jᵀ and z_i=Σ_{j≤i}ϕ(k_j); then output_i=φ(q_i)ᵀS_i / φ(q_i)ᵀz_i, with S_i=S_{i-1}+ϕ(k_i)v_iᵀ. That is a recurrence with fixed-size state — linear attention *is literally an RNN*: O(1) memory per decode step, parallelizable in training via cumulative sums. (This RNN framing is exactly the observation the Katharopoulos work makes too.)

How do I keep myself honest? I have derived the complexity, not measured it — that part is a theorem. What must be tested is quality: does 1+q̂·k̂ hold accuracy against softmax attention, and against the elu+1 variant? Decision rule: I don't claim a winner in advance; if my Taylor-motivated map isn't clearly better, I keep it only as the theoretically most faithful option and report the trade-off, preferring whichever preserves full token-token attention and causal decoding without extra machinery.

— this is the kind of thing I, Jianlin Su, would write up as motivation → derivation → code → experiment.

**Core idea:** The O(n²) cost comes from the softmax forcing you to materialize QKᵀ; abstract attention to a normalized non-negative similarity, require that similarity to factor as φ(q)ᵀϕ(k), and matrix-multiplication associativity yields exact O(n) attention that also unrolls into an RNN recurrence for causal decoding — with 1+q̂·k̂ (a first-order Taylor of exp) as the most softmax-faithful choice.

**Non-trivial crux:** The bottleneck is the softmax, not the matmul; linearity is recoverable only if the replacement similarity is both non-negative (to stay a valid distribution) and inner-product-factorizable (to invoke associativity), and one must verify quality empirically since the linear form is exact in cost but only approximate in expressivity.
```

## fact-check changes

```plaintext
- (b/c) Removed the vague 'others in vision picked... or softmax along different axes' framing; named the true contemporaneous competitor (Katharopoulos et al. 2020, 'Transformers are RNNs', elu+1 feature map) and noted the RNN framing came from that work too, rather than presenting it as solely the researcher's own.
- (a) Trimmed the ablation comparison to reference only the real contemporaneous variant (elu+1), removing the fictitious 'axis-softmax' competitor from the decision rule.
- (d) Added the caveat that 1+q̂·k̂ is not an exact φᵀϕ factorization, unlike elu+1, restoring the real edge case.
```