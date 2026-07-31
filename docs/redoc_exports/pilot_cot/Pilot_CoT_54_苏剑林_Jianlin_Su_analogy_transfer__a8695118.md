# Pilot CoT 54 — 苏剑林 Jianlin Su · analogy_transfer

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例1：为把"相对位置"塞进注意力内积 → 解一个函数恒等式，推出旋转位置编码 RoPE

- route: analogy_transfer  ·  case score: 10/10

- lengths (qwen2.5-32b tokens): user(setup) **214** · assistant(cot) **1214** · whole **1485**

- fact-check changes: 2

## user (setup)

```plaintext
It's early 2021. Transformers need position information because pure attention is permutation-invariant. Two families dominate. Learnable absolute position embeddings (as in BERT/GPT) are simple and fast, but a model pretrained at max length 512 can only ever handle 512 tokens — no extrapolation. Sinusoidal absolute encodings can implicitly express relative offsets through trig identities, yet they're oddly underused and don't feel principled. On the other side, classical relative position encodings (Shaw-style, Transformer-XL, XLNet) work well but all operate directly by surgery on the attention matrix q_i k_j^T — which means they're incompatible with any attention variant that never materializes that matrix (e.g. linear attention). The wish is clear: get the extrapolation-friendliness of relative encoding while keeping the simplicity of absolute encoding, and ideally stay usable even when the attention matrix is never formed. Propose a genuinely new mechanism that achieves this, and be precise about what makes it work.
```

## assistant (cot)

```plaintext
Let me not start by piling up tricks — add a learnable bias here, concat a distance embedding there, tweak the mask. That's how most people approach position encoding, and the result always 'works' without anyone being able to say why, and it collapses the moment you change dimension or switch to linear attention. I want to solve, not to guess. So first: what do I actually want, written as an equation?

I want to apply a transform to each query and key that depends only on the token and its *absolute* position — f(q,m), f(k,n) — such that when I take their inner product, the result depends only on the *relative* position m-n. Write it down as the North Star identity: ⟨f(q,m), f(k,n)⟩ = g(q,k, m-n), with f(q,0)=q. Everything else is just solving this functional equation. If I can't reduce a requirement to a clean identity like this, I don't yet understand what I want.

Now the transfer move. Where have I seen 'inner products that only care about a difference'? In the geometry of the plane, via complex numbers. There is a clean bridge I trust: for 2D vectors, ⟨q,k⟩ = Re[q k*]. That's the mechanism I'm importing — treat a 2D slice of the feature vector as a complex number, and use the algebra of complex multiplication. What transfers is precisely this: multiplying a complex number by a unit-modulus phase e^{iθ} rotates it without changing its length, and the product of two such phases combines *additively* in the exponent. What I must be careful about — what could break — is that this identity Re[q k*] is exact only for 2D; for full d-dimensional vectors I can't just wave hands, I'll have to build the d-dim case explicitly. And the phase must be modulus-1 (a rotation, not a scaling), otherwise lengths drift and stability suffers.

So try the transferred mechanism: multiply q at position m by e^{imθ} and k at position n by e^{inθ} — rotate each proportionally to its absolute position. Then ⟨q_m e^{imθ}, k_n e^{inθ}⟩ = Re[q_m k_n* e^{i(m-n)θ}]. The e^{imθ} and e^{inθ} — the two absolute positions — fuse into a single e^{i(m-n)θ}: the inner product sees only m-n. That's exactly the identity I wanted, and it fell out of the analogy rather than from fitting.

But I shouldn't stop at 'here's something that satisfies it.' Let me actually solve the functional equation to be sure the rotation is *the* solution and not one lucky instance. Write f(q,m) = R(q,m) e^{iΘ(q,m)}. The identity splits into a modulus equation and a phase equation. Setting the two positions equal and using f(q,0)=q forces the modulus to be position-independent — R = ‖q‖. The phase equation, taken at adjacent positions, gives φ(m)-φ(m-1) = constant, i.e. φ(m) is arithmetic, φ(m)=mθ. So the closed-form solution is: rotate the vector by mθ. Nothing extra. It is derived, not tuned — which is why I expect it to be simple, interpretable, and generalizable.

Generalize to d dimensions by pairing coordinates and using a block-diagonal rotation with per-block angle mθ_i. Orthogonality guarantees (R_m q)^T(R_n k) = q^T R_{n-m} k — again only relative position — and preserves norms, so it shouldn't disturb model stability. For the frequencies I'll reuse θ_i = 10000^{-2i/d} from the sinusoidal tradition, and I can check that inner products decay with distance (an Abel/summation-by-parts bound makes this plausible) — though I should stay honest: almost any smooth monotone frequency schedule would give some decay, so this specific choice isn't sacred.

Why this cuts against consensus: everyone injects relative position by editing the attention matrix. My mechanism injects it by rotating q and k *before* the dot product — so it never touches the matrix, which means it's a relative encoding that also works for linear attention where the matrix is never formed. That's the payoff of the transfer.

How I keep myself honest: I'll swap absolute encoding for this in a real pretrained model and test long-sequence behavior; the decision rule — since the derivation already gives φ(m)=mθ with fixed frequencies, I'll keep them fixed as the default (no free parameters unless experiments demand them); and if the derived form doesn't at least match on training length while helping extrapolation, I won't trust the story regardless of how elegant the algebra looks. Then write the whole motivation→derivation→code→experiment up publicly; the paper is a byproduct.

**Core idea:** Impose the identity ⟨f(q,m),f(k,n)⟩ = g(q,k,m-n) and solve it via the complex-number identity ⟨q,k⟩=Re[q k*]; the closed-form solution is to rotate each query/key by an angle proportional to its absolute position, so inner products depend only on relative position — absolute-encoding simplicity with relative-encoding behavior, and no attention-matrix surgery.
**Non-trivial crux:** The transfer works because the phase is unit-modulus (a norm-preserving rotation whose phases add), so absolute positions fuse into a single m-n; the Re[q k*] bridge is exact only in 2D, so the d-dim case must be built explicitly as block-diagonal rotations, and the frequency schedule is a reasonable default, not the unique source of distance decay.
```

## fact-check changes

```plaintext
- (a) Replaced the fabricated decision rule about testing learnable θ_i ('if making θ_i learnable doesn't move them meaningfully, fix them') with the historically accurate framing: the derivation itself yields fixed frequencies, kept fixed by default unless experiments demand otherwise — avoiding a narrated ablation not actually central to the derivation.
- (d) Softened 'it's the one relative encoding that also works for linear attention' to 'a relative encoding that also works', removing the over-absolutized uniqueness claim.
```