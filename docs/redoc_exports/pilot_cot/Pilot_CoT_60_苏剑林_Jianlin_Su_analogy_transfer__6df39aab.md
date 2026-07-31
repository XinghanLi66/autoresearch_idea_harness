# Pilot CoT 60 — 苏剑林 Jianlin Su · analogy_transfer

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例3：想让注意力"训短测长" → 从"熵不变性"推出 log n 缩放因子

- route: analogy_transfer  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **198** · assistant(cot) **1123** · whole **1378**

- fact-check changes: 4

## user (setup)

```plaintext
It's late 2021. Transformers dominate NLP, and everyone writes attention as softmax(QKᵀ/√d)V. The √d is universally justified by a variance argument: at initialization, dot products of d-dimensional vectors have variance proportional to d, so dividing by √d keeps logits from saturating softmax. That story explains *training stability* fine. But there's a separate, nagging problem with no clean solution yet: models trained at one sequence length degrade badly when evaluated on longer sequences ('train short, test long'). People patch this with relative position schemes, learned biases, or ad-hoc temperature tweaks found by grid search. Suppose you refuse to grid-search a temperature and refuse to lean on the variance intuition. Can you find a principled reason the scaling factor should depend on sequence length n, and derive what that dependence must be? Propose a genuinely novel idea for how to reason about the attention scale so that it generalizes across lengths.
```

## assistant (cot)

```plaintext
Let me refuse the usual entry point. The √d story is a *variance* story — it's about initialization, not about generalization across length. Grid-searching a temperature for each test length is exactly the kind of luck-hunting I dislike: you find a value that works, it fails at another length, you search again, and you've learned nothing transferable. If I want 'train short, test long' to work, I should first ask: what quantity secretly drifts when n grows, and can I write down a scaling λ(n) that keeps it fixed?

So the move is to import a lens from another field. Attention weights aᵢⱼ = softmax(λ qᵢ·kⱼ) are, for a fixed query i, a genuine probability distribution over the j positions — nonnegative, summing to one. That's an invitation. In information theory a distribution has an entropy H = −Σ aᵢⱼ log aᵢⱼ, and entropy has a very physical meaning here: H≈0 means the query focuses sharply on one token; H≈log n means it's smeared uniformly over all n tokens. That's the transfer. What carries over cleanly: entropy is exactly 'how spread out is the attention', and 'defocusing when the sequence gets longer' is precisely entropy inflating with n.

Now the discipline — what breaks in the transfer? Entropy is a global scalar summary; it can't capture *which* tokens get attended. Two very different attention patterns can share an entropy. So I am not claiming to preserve the full distribution, only this one summary statistic. And I can't make H *exactly* n-independent — the content of q,k matters too. This is a semi-quantitative estimate, and I should say so out loud rather than dress it as a theorem. The honest target is a weaker, cleaner one: **choose λ(n) so that the n-dependent part of the entropy is cancelled.** When I add new tokens, I don't want them to 'steal' attention mass from the tokens already there; I want existing tokens to keep focusing as before.

So let me actually extract the n-dependence. Write H via the standard logsumexp identity for softmax entropy. Each sum over j is a sum of n terms; approximate it as n times an average over positions, Σⱼ e^{λqᵢ·kⱼ} ≈ n · E_j[e^{λqᵢ·kⱼ}]. The log of that produces a bare **log n** term — there is the entropy inflation, sitting out front, exactly the thing I want to kill. Everything else is an expectation over the angle between q and k. Under simplifying assumptions on the geometry of q,k — treat their directions as roughly uniform and their magnitudes as controlled so that qᵢ·kⱼ scales with d — I can approximate that angular expectation into something roughly linear in λd. The whole entropy collapses to roughly H ≈ log n − c·λd + O(1) for some constant c I don't much trust — this is a heuristic estimate, not a rigorous claim.

Setting the n-dependent balance to zero, log n = c·λd, gives the closed form λ = κ log n / d, folding my distrust of c into a free constant κ. That's the leap: **the scale should grow like log n**, not stay fixed. And I can remove even κ by calibration — demand that at the mainstream pretraining length (say 512) this reduces to the ordinary 1/√d attention. That pins κ and yields a hyperparameter-free scale = log₅₁₂(n)/√d: at 512 it's exactly √d attention, longer sequences get a larger factor, shorter get smaller.

How do I keep myself honest given the approximations? I don't get to claim victory from the derivation. The decision rule: run controlled masked-LM at a short training length and evaluate across lengths. Two conditions must both hold — (i) *no regression at the training length* (if my scale hurt where the baseline was tuned, the whole idea is suspect), and (ii) *clear gains when extrapolating longer*. If instead the added log n machinery doesn't beat plain √d out-of-distribution, I discard it on parsimony. And a caveat I must preserve: a naked log n (natural log, no length calibration) blows up for large n and can destabilize training — anyone testing only at tiny lengths would never see the failure. The calibration isn't cosmetic; it's what keeps the factor sane at scale.

**Core idea:** Treat each attention row as a probability distribution and demand its *entropy* be (semi-quantitatively) invariant to sequence length; the log n entropy-inflation term then forces a length-dependent scale λ ∝ log n / d, calibrated to reduce to 1/√d at the pretraining length — giving a hyperparameter-free softmax(log₅₁₂(n)/√d · QKᵀ)V.
**Non-trivial crux:** Only the *entropy* (one scalar summary of the distribution) is imported and only its n-dependence is cancelled — it's an admitted semi-quantitative estimate resting on geometric assumptions, and the length-calibration is essential, since an uncalibrated natural-log scale explodes and destabilizes training at large n.
```

## fact-check changes

```plaintext
- Replaced Chinese '撞运气' with English 'luck-hunting'.
- Softened the fabricated-precision derivation step: removed the specific 'LayerNorm and near-orthogonal projections, fix ‖q‖=‖k‖=√d, Laplace-approximate' chain presented as if it cleanly yields the result, replacing with a labeled heuristic estimate under simplifying geometric assumptions.
- Marked the H ≈ log n − c·λd result explicitly as a heuristic estimate rather than a derived fact.
- Toned the 'kills gradients' over-absolutization to 'can destabilize training' for accuracy while preserving the caveat.
```