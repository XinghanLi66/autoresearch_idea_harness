# Pilot CoT 53 — 苏剑林 Jianlin Su · first_principles

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例1：为把"相对位置"塞进注意力内积 → 解一个函数恒等式，推出旋转位置编码 RoPE

- route: first_principles  ·  case score: 10/10

- lengths (qwen2.5-32b tokens): user(setup) **223** · assistant(cot) **1138** · whole **1418**

- fact-check changes: 2

## user (setup)

```plaintext
It's early 2021. Transformer attention is permutation-invariant, so position encoding is mandatory. Two families dominate, each with a real defect. Learnable absolute position embeddings (BERT/GPT-style) are simple and fast but don't extrapolate: pretrain at length 512 and you're stuck at 512. Sinusoidal absolute encodings can *implicitly* express relative offsets via trig identities, but somehow they're rarely used directly and feel like 'an absolute encoding that wants to be a relative one, but isn't good enough'. The classical relative encodings (Shaw / Transformer-XL / XLNet style) work well but all operate by surgically editing the attention matrix q_i·k_j^T — which means they can't be used with attention variants that never materialize that matrix. You want the best of both worlds: something as clean and drop-in as an absolute encoding, that nonetheless injects genuinely *relative* information, and that doesn't require touching the attention matrix. Propose a genuinely novel construction and show why it is almost forced.
```

## assistant (cot)

```plaintext
Let me refuse, on principle, to start by listing tricks. Every existing method — learnable bias, distance embeddings, attention-matrix surgery — is somebody's *guess* at a scheme. I don't want a scheme I can tune into working; I want a scheme I can *derive*, so that its simplicity and generalizability come for free. So first I make myself write down exactly what I want as a mathematical target, and then I only allow myself to solve it.

What do I actually want? I want to apply, to the query and key separately, a transformation that depends only on the vector itself and its *absolute* position — call them f(q,m) and f(k,n) — such that when I take their inner product, the position-dependence collapses to depend only on the *relative* offset m−n. That is the whole thing:

  ⟨f(q,m), f(k,n)⟩ = g(q, k, m−n),  with f(q,0)=q, f(k,0)=k.

This single identity is my North Star. Notice what it buys me before I've solved anything: because f acts on q and k individually *before* the inner product, whatever I find will be applied token-wise, never to the assembled attention matrix. That is precisely the property the classical relative encodings lack. So if a solution exists at all, it automatically inherits the drop-in simplicity of absolute encoding while carrying relative information. Now I just have to solve the functional equation.

Start in 2D, the smallest nontrivial case, and use the one identity that turns 2D geometry into algebra: a 2D inner product is the real part of a complex product, ⟨q,k⟩ = Re[q k*]. Write f in polar form, f(q,m) = R(q,m)·e^{iΘ(q,m)}. Substitute into the identity and split it into two coupled equations — a modulus equation R(q,m)R(k,n) = R_g(m−n) and a phase equation Θ(q,m) − Θ(k,n) = Θ_g(m−n). This is the move I trust: don't guess the function, constrain it until it has no freedom left.

Now squeeze. Set m=n in the modulus equation and use the initial condition f(·,0)=·: the modulus must be independent of position, R(q,m)=|q|. That kills one whole degree of freedom. For the phase, set n = m−1: the phase increment φ(m) − φ(m−1) must be a constant. A sequence with constant successive difference is an arithmetic progression, so φ(m) = mθ, forced. There is essentially nothing else it *can* be. And an argument shifting by mθ with fixed modulus is, by the geometric meaning of complex multiplication, exactly a rotation of the vector by angle mθ. The closed form isn't invented; it fell out.

Lift to d dimensions the obvious way — pair the coordinates and rotate each pair by its own angle mθ_i, i.e. a block-diagonal rotation R_m. Then R_m^T R_n = R_{n−m}, so q^T R_m^T R_n k depends only on n−m. Two bonuses I get for nothing: R_m is orthogonal, so it preserves norms and shouldn't perturb training stability; and since it's just an elementwise rotate applied per token, it slots straight into linear attention — the one place the attention-matrix methods can never reach. That's the unique selling point, and it's a *consequence*, not a design choice.

How do I keep myself honest rather than falling in love with the algebra? Three checks. (1) Frequencies: I'll reuse the sinusoidal schedule θ_i = base^{−2i/d} and try to prove long-range decay of the inner product — Abel summation / summation-by-parts should bound it — but I must state honestly that many smooth monotone schedules would also give decay; this choice is not sacred. (2) Learnable vs fixed θ_i: I could let θ_i be trainable and watch whether it actually moves; decision rule — if it doesn't update meaningfully, fix it, because I don't add parameters for decoration. (3) The real test: swap absolute encoding for this in a model and pretrain across varying lengths. Decision rule — it must not lose accuracy at training length, and I want to see whether it yields an extrapolation gain; if the rotation buys nothing over absolute encoding, I keep the simpler baseline. Then I write the whole thing up publicly: motivation → derivation → code → experiment. Writing it for others is what forces every 'seems right' into 'I can prove it or I measured it'.

**Core idea:** Demand that a per-token, position-dependent transform of q and k make their inner product depend only on m−n; solving that functional equation on the complex plane forces the transform to be a rotation by angle mθ — position injected by rotating each coordinate-pair, giving relative information with absolute-style, drop-in simplicity that even works for linear attention.
**Non-trivial crux:** The rotation is *derived*, not fitted: the modulus equation forces norm-preservation and the phase equation forces an arithmetic progression φ(m)=mθ. But the decay/frequency schedule is only one convenient choice among many smooth monotone ones, so its properties (like long-range decay) must be argued/verified, not assumed unique.
```

## fact-check changes

```plaintext
- (a) Softened check (3) from asserting 'should show a clear extrapolation gain' to framing it as something to be tested (whether it yields an extrapolation gain), since the actual RoPE paper's headline claim was drop-in relative encoding compatible with linear attention, not a proven extrapolation win at the time.
- (a) Reworded check (2) to a hypothetical ('I could let θ_i be trainable') to avoid implying a completed ablation outcome.
```