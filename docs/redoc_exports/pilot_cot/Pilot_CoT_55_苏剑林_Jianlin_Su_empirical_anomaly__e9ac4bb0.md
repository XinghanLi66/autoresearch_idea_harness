# Pilot CoT 55 — 苏剑林 Jianlin Su · empirical_anomaly

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例2：追问"Attention 一定要有 Softmax 吗" → 用结合律把 O(n²) 降成 O(n)

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **211** · assistant(cot) **1234** · whole **1502**

- fact-check changes: 5

## user (setup)

```plaintext
It's mid-2020. Standard self-attention Attention(Q,K,V)=softmax(QKᵀ)V has become the workhorse of Transformers, but its O(n²) time and memory cost in sequence length n is a hard wall for long documents. Several approaches to cheapen it have appeared: sparse attention patterns (needing hand-designed masks and custom kernels), LSH-based bucketing achieving O(n log n) with complicated implementation, and low-rank projections of the key/value sequence to a fixed length m. A common assumption in the community is that the cost is intrinsic to 'computing the attention matrix' and that we must approximate or sparsify that n×n matrix. You have BERT-base numbers in front of you and notice something concrete: the head dimension d=64 is tiny, while n runs into the hundreds or thousands. Propose a genuinely novel way to think about, and cut, the cost — without giving up the full token-to-token expressiveness or the ability to do autoregressive generation.
```

## assistant (cot)

```plaintext
Let me start from the arithmetic that's literally sitting in front of me, not from anyone's story about what attention 'is'. BERT-base: d=64, but n is in the hundreds to thousands. So d≪n, and yet everyone reports the cost as O(n²). That mismatch bothers me. If the head dimension is that small, where does the quadratic actually come from? People keep saying 'computing QKᵀ is the bottleneck' and then go off designing sparse patterns or LSH buckets to avoid building that matrix. But that explanation doesn't survive a careful look at the expression.

Write it out honestly. QKᵀV is a product of three matrices. Matrix multiplication is associative. If I were allowed to compute KᵀV first — that's a d×d object, tiny — and then left-multiply by Q, the whole thing is O(nd²)=O(n) in sequence length. So the quadratic is NOT forced by 'attention' at all. The only thing standing between me and the associativity trick is that softmax sits in the middle: softmax(QKᵀ) is a nonlinear function of the full n×n matrix, so I'm forced to instantiate all n² entries before I can apply it. That's the real culprit. The bottleneck isn't QKᵀ, it's the Softmax that pins me to an explicit n×n matrix. That's the measured contradiction I trust more than the folklore.

So the question is not 'how do I approximate the n×n matrix' but 'what do I actually need softmax for, and can I get it a different way?' Let me refuse to just crudely delete softmax — raw inner products q·k are signed, and a weighted average with negative weights isn't a valid distribution, so 'no softmax' naïvely gives nonsense. I need to reformulate the object first. Rewrite one output row as a weighted average: output_i = Σ_j w_ij v_j with w_ij = e^{q_i·k_j}/Σ. The essential content is 'a normalized average with nonnegative weights.' Replace e^{q_i·k_j} by any similarity sim(q_i,k_j) with the single requirement sim≥0. Now the whole design collapses to one algebraic wish: I want the associativity back. That happens exactly when sim factorizes, sim(q_i,k_j)=φ(q_i)ᵀϕ(k_j) with φ,ϕ nonnegative feature maps. Then Σ_j φ(q_i)ᵀϕ(k_j)v_j = φ(q_i)ᵀ(Σ_j ϕ(k_j)v_jᵀ): the bracket is d×d, computed once, reused for every i. O(n). And note — when φ=ϕ this is just a kernel; softmax attention is one particular (non-factorizable) similarity. So I haven't invented a hack, I've located softmax inside a family.

Which feature map? I could grab any nonnegative map, e.g. elu(x)+1 — this is essentially the choice made in the contemporaneous linear-attention work (Katharopoulos et al., 2020), which frames attention as a kernel and uses that feature map. But 'pick a kernel and hope' is exactly the arbitrariness I distrust. Let me instead try to derive one by staying faithful to the true softmax. Taylor: e^{q·k}≈1+q·k. To keep 1+q·k≥0 I ℓ₂-normalize q and k, so q·k∈[-1,1]. That gives sim = 1 + q̂·k̂. It isn't literally of the φᵀϕ form for arbitrary sign, but with normalized inputs it's a low-order approximation of the real thing rather than a random kernel — so my intuition is it should track scaled-dot attention more honestly. Decision rule: I'll compare the 1+q̂·k̂ form against the borrowed elu+1 kernel on the same task; if the principled form doesn't clearly help, keep whichever is simpler, but in any case report it as an approximation of softmax, not a truth. I won't claim a winner before running it.

One more thing I refuse to give up: full token-to-token coupling and autoregressive generation. Low-rank projection of K,V to fixed length m tends to destroy causality and I doubt m stays effectively constant for genuinely long sequences. But my factorized form keeps causality: restrict the sum to j≤i and carry a running state S_i=S_{i-1}+ϕ(k_i)v_iᵀ, z_i=z_{i-1}+ϕ(k_i). Then output_i=φ(q_i)ᵀS_i/φ(q_i)ᵀz_i. That recurrence is an RNN — O(1) memory per step at inference, though training loses the fully-parallel matrix form unless done carefully. Linear attention in causal mode just is a linear RNN. I'll write the derivation, the code, and the experiments out in the open so every 'I think this works' becomes 'I can show it.'

**Core idea:** The O(n²) cost of attention comes not from computing QKᵀ but from the Softmax that forces an explicit n×n matrix; reformulate attention as a nonnegative-similarity weighted average and, whenever the similarity factorizes as φ(q)ᵀϕ(k), use matrix-multiplication associativity to reach O(n) — with a Taylor-derived 1+q̂·k̂ similarity meant to stay faithful to softmax, and a causal running-state form that reveals linear attention as an RNN.

**Non-trivial crux:** The trick works only if the similarity is both nonnegative (so the weighted average stays a valid distribution) AND factorizable into feature maps; without the nonnegativity constraint the 'attention' is ill-posed, and the linear form is a genuine approximation of true softmax attention, not an identity.
```

## fact-check changes

```plaintext
- (a) Removed asserted claim that the principled 1+q̂·k̂ form would 'track softmax more honestly' as a settled conclusion; reframed as intuition/hypothesis and made the decision rule explicit that no winner is claimed before running.
- (b)/(c) Named the true contemporaneous competitor (Katharopoulos et al. 2020 linear transformers with elu+1 feature map) as the source of the 'borrowed kernel' rather than treating it as an anonymous arbitrary choice.
- (c) Softened 'softmax is one particular kernel choice' to note softmax similarity is non-factorizable, avoiding a hindsight over-clean framing.
- (d) Added caveat that causal linear attention loses the fully parallel matrix form during training unless handled carefully; softened absolute claims about low-rank methods ('tends to destroy causality', 'effectively constant').
- (e) No non-English words required changing (name preserved as given).
```