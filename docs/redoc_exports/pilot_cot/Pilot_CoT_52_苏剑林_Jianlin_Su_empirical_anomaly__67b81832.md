# Pilot CoT 52 — 苏剑林 Jianlin Su · empirical_anomaly

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例1：为把"相对位置"塞进注意力内积 → 解一个函数恒等式，推出旋转位置编码 RoPE

- route: empirical_anomaly  ·  case score: 10/10

- lengths (qwen2.5-32b tokens): user(setup) **257** · assistant(cot) **1216** · whole **1530**

- fact-check changes: 4

## user (setup)

```plaintext
It is early 2021. Transformer position encoding is a crowded field with two camps, each with a concrete, measurable defect that people mostly explain away. Trainable absolute encodings (BERT/GPT-style) are simple and fast, but when you pretrain at max length 512, you can only serve 512 — feed it a longer sequence and it simply has no encoding to use. Sinusoidal absolute encoding was designed so that trig identities let a linear map 'implicitly' express relative offsets, yet in practice almost nobody uses it directly for downstream and its extrapolation is unimpressive. The relative-position family (Shaw / Transformer-XL / XLNet) genuinely works, but every one of them operates by surgically editing the attention score matrix q_i k_j^T — which means they cannot be used in linear-attention variants where you never form that matrix at all. You have run enough of these yourself to trust the measured pattern, not the folklore. The question: can you invent a genuinely new way to inject position that gets the best of both camps — the speed and matrix-free simplicity of an absolute scheme, with attention that truly depends only on relative distance? Propose the idea and how you'd derive and check it.
```

## assistant (cot)

```plaintext
Let me start from what actually shows up when I run these, not from what the papers claim. Trainable absolute encoding: pretrain at 512, and at length 600 the model has no vector for positions 512..599 — it doesn't 'degrade gracefully', it has nothing. That's not a philosophical limitation, it's a measured wall. Sinusoidal is supposed to encode relative offset via sin/cos identities, and I keep reading that it 'can' express relative position — yet in my own runs it barely extrapolates and almost nobody ships it directly. The stated story and the measurements don't line up, and 'reason unknown' is not an answer I accept.

The relative-position family (Shaw, Transformer-XL, XLNet) does work — I've verified that. But I notice a shared, awkward fact: every single one of them reaches into the expanded q_i k_j^T and patches the position term inside the score matrix. Fine when you compute that matrix. But the moment I try linear attention, where the whole point is to never materialize the n×n matrix, all of these become inapplicable. So the empirical situation is a contradiction I can state in one line: the schemes that give me true relative dependence all require touching the attention matrix; the schemes that don't touch it (absolute) don't give me clean relative dependence or extrapolation. Everyone treats this as 'just how it is'. I don't.

So instead of piling on tricks — a learnable bias here, a distance embedding there, a mask tweak — let me refuse to guess and instead write down exactly what I want as an equation. I want a transformation applied to q at its own absolute position m, and to k at its own absolute position n — purely absolute, so it's cheap and matrix-free — such that their inner product depends only on m−n. That is the north star: ⟨f(q,m), f(k,n)⟩ = g(q,k,m−n), with f(·,0) = identity. If I can solve this function equation, I automatically get 'relative behavior from an absolute operation', which is precisely the contradiction dissolved.

Now solve it, don't fit it. In 2D there's a clean identity: an inner product is the real part of a complex product, ⟨q,k⟩ = Re[q k*]. Write f in polar form, modulus times phase. Feed the target equation and split it into two: a modulus equation and a phase equation. Set m=n and use the initial condition — the modulus must be independent of position, so it's just ‖q‖; the transform cannot rescale, which is reassuring because it means it won't disturb the model's stability. For the phase, set n = m−1: the phase increment between adjacent positions is a constant. That forces the phase to be an arithmetic progression, φ(m) = mθ. There is no freedom left — the closed-form solution is simply: multiply by e^{imθ}, i.e. rotate the 2D vector by an angle proportional to its position. It fell out; I didn't search for it. Extend to d dimensions by pairing coordinates two-by-two and rotating each pair by mθ_i — a block-diagonal orthogonal rotation. Orthogonality re-confirms modulus preservation, and R_m^T R_n = R_{n−m} guarantees attention sees only the offset.

Why this cuts against the room: everyone is either adding parameters or editing the score matrix; I'm doing neither. It's a parameter-free geometric rotation applied before the dot product, so it drops straight into linear attention — the one property none of the relative schemes have.

How I keep myself honest, with no result assumed. First, reuse the sinusoidal θ_i = 10000^{-2i/d} and try to show (via Abel summation / summation-by-parts) that the attention magnitude tends to decay with |m−n| — but I must be honest that any such decay would not be unique to this choice; many smooth monotone frequency schedules should give something similar, so I won't oversell θ. Second, decision rule on making θ_i learnable: try it, and if the θ_i don't meaningfully move during training, keep them fixed — no ornament without measured payoff. Third, the real test: swap absolute encoding for this in a BERT-style model, pretrain alternating longer max-lengths, and check whether training accuracy holds as length grows (the extrapolation claim) and whether long-document matching improves over the 512-capped baseline. If it doesn't clearly help, the elegance of the derivation buys it nothing. Then write it all up — motivation → derivation → code → experiment — publicly, because writing it for others is what forces every 'I think this is right' into 'I can prove it / I measured it'.

**Core idea:** Formalize 'relative position via an absolute operation' as the identity ⟨f(q,m),f(k,n)⟩=g(q,k,m−n); solving this function equation on complex/2D pairs forces the unique closed form f(q,m)=q·e^{imθ} — i.e. rotate each coordinate-pair by an angle proportional to its position — which makes the dot product depend only on m−n while being matrix-free, norm-preserving, and usable in linear attention.
**Non-trivial crux:** The rotation is *derived*, not fitted — the modulus and phase equations leave no freedom, so f(q,m)=q e^{imθ} is forced; but the long-range decay is a property of the frequency schedule, not of the rotation itself (many smooth monotone θ_i should give it), so that part must be claimed as an empirical/half-quantitative choice, not as something the derivation proves.
```

## fact-check changes

```plaintext
- (e) Replaced Chinese phrase '原因不详' with English 'reason unknown'.
- (a) Softened the decay claim in the verification step from asserting the derivation proves decay ('prove ... decays') to what would be tested/attempted ('try to show ... tends to decay'), matching that this is an empirical/half-quantitative property rather than a proven-for-RoPE result.
- (a) Reworded the learnable-θ decision from 'fix them' phrasing kept, but framed as a genuine try-and-decide rule (no numeric winner asserted).
- (a) In the crux, changed 'many smooth monotone θ_i give it' to conditional 'should give it' to avoid asserting an unrun empirical result as established.
```