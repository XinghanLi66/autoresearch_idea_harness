# Pilot CoT 63 — 苏剑林 Jianlin Su · analogy_transfer

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例4：Muon 火了不止复现 → 把它重新推导成"谱范数下的最速下降"

- route: analogy_transfer  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **199** · assistant(cot) **1241** · whole **1497**

- fact-check changes: 4

## user (setup)

```plaintext
It is late 2024. A new optimizer called Muon is going viral on Twitter. It only acts on matrix-shaped parameters W (leaving vectors/scalars to Adam), and empirically it beats AdamW on language-model pretraining. Its update keeps a momentum buffer M, then applies something called msign(M) — a 'matrix sign' that, given the SVD M=UΣVᵀ, throws away the singular values and keeps only the orthogonal factors UVᵀ, approximated cheaply by a few Newton–Schulz polynomial iterations with some hand-tuned magic coefficients. The original release gives working code but essentially no mathematical derivation — just a terse comment about why those coefficients were chosen. Most of the community is busy reproducing it and reporting numbers. You suspect there is a clean principle hiding under the magic numbers. Propose a genuinely novel way to understand WHY this works from first principles, and how that understanding could be turned into something reusable and scalable.
```

## assistant (cot)

```plaintext
Everyone is racing to reproduce Muon and report a number. That's the anti-pattern I refuse to fall into: reproduce, tune, report, and then have zero theoretical handle the moment it breaks at a scale you care about. Let me instead reverse-engineer why it works. I'm Jianlin Su; my habit is to force a fuzzy 'it just works' into a solvable mathematical target and solve it into a clean closed form.

Start from the anomaly that is really an analogy. The 1×1 case: msign of a scalar x is x(x²)^{-1/2} = sign(x). So Muon is literally the matrix generalization of a sign-momentum update — the same family as the Lion/Tiger-style sign optimizers I've played with before. That's my transfer anchor. I know the sign-descent story well: the plain SGD/Adam family treats a weight matrix as a bag of loose scalars, i.e. it measures 'distance moved' in the Frobenius norm. What transfers cleanly is the *variational* framing of an optimizer step: linearize the loss and pick the step that maximizes descent under a norm constraint. Write ΔW = argmin [ ‖ΔW‖²/(2η) + Tr(GᵀΔW) ]. Split ΔW into a stepsize times a unit direction Φ. Then the whole question is: **unit in which norm?**

Here is exactly what breaks in the naive transfer, and it's the crux. If the constraint is the Frobenius norm, you recover ordinary steepest descent — the gradient direction. That's the 'bag of scalars' geometry. But a weight matrix isn't a bag of scalars; its action on activations is governed by its *spectrum*. So the invariant I want to try respecting is the spectral (2-)norm, not Frobenius. Redo the maximization with ‖Φ‖₂ = 1. Maximizing Tr(GᵀΦ) subject to bounded spectral norm is exactly the dual pairing: the answer is bounded by the nuclear norm Σᵢσᵢ(G), and equality holds precisely when Φ = msign(G) = UVᵀ (taking the orthogonal factors from a compact SVD; well-defined up to the usual degeneracy when singular values coincide). So msign is not a magic operation — it is the steepest-descent direction under the spectral-norm geometry. In one line: SGD = steepest descent under Frobenius; Muon = steepest descent under spectral. Same principle, sharper metric, because the spectral norm is the geometry that plausibly matches how a matrix transforms signals — I should treat this as the intuition and let the math and experiments earn it, not assert it as decided.

Now keep myself honest about the magic numbers. Newton–Schulz is only a cheap polynomial approximation of msign — it's trying to push every singular value toward 1, i.e. approximate t^{-1/2} near t=1. I shouldn't trust three coefficients handed down from a tweet. I'll write code that samples the singular-value distribution of random matrices and *fits* the iteration coefficients so the singular values actually collapse to 1 over the true operating range — the Taylor form near t=1 gives a starting point, but the honest test is: does the fitted polynomial map the empirical singular spectrum to ≈1 in the given number of iterations? Decision rule: if a fitted set does the job at least as well and more robustly across the sampled spectra, prefer it; if the published numbers already sit at that quality, keep them on parsimony grounds.

The spectral framing also earns its keep if it predicts failures instead of just explaining wins. If Muon is steepest descent under a spectral constraint, then the quantity I must watch as I scale is the spectral norm of W itself. Two consequences to test. First, weight decay may not be cosmetic: adding λW bounds ‖W‖₂, and without it I'd expect drift/blowup at large scale — something to verify by watching whether the spectral norm creeps up when WD is off. Second, to reuse Adam's learning rates I should align the *update RMS*: normalize any new update to the RMS scale Adam produces, and since msign of an n×m matrix of rank r has an analytic RMS of √(r/(nm)), the alignment factor simplifies to a clean form scaling like 1/√(max(n,m)) — the aim being to avoid a fresh learning-rate re-search. And if at very large scale I see attention logits exploding, the spectral-norm invariant tells me where to intervene: bound the spectral contribution of the Q/K weights, rather than blindly grid-searching a temperature.

What I would NOT overclaim: I have not shown Muon beats AdamW at trillion scale here — that requires the actual runs; I'm giving the geometry that could make the behavior predictable and the invariants I'd monitor. And cross-stage mixing (Adam-pretrain → Muon-finetune) is unexplained; I'll flag it as open, not paper over it.

**Core idea:** Muon's msign is not a heuristic — it is the exact steepest-descent direction under a *spectral-norm* constraint (UVᵀ achieves the nuclear-norm dual bound), making Muon the spectral-geometry analogue of Frobenius-geometry SGD; treating 'which norm' as the design knob turns the magic operation, the Newton–Schulz coefficients, the RMS/learning-rate alignment, and large-scale stability fixes into consequences of one principle.
**Non-trivial crux:** The whole thing hinges on choosing the *spectral* norm as the step-size metric because a weight matrix acts through its spectrum, not as a bag of scalars — and the same 'bound the spectral norm' invariant is what I'd expect to force weight decay and dictate spectral-based clipping as scale grows; drop that invariant and you lose all theoretical grip when it breaks.
```

## fact-check changes

```plaintext
- (a) Converted asserted consequences ('weight decay isn't cosmetic', clipping fix, 'no learning-rate re-search') into hypotheses to test with decision/monitoring language; kept the honest coefficient-fitting decision rule.
- (a/c) Reframed 'the geometry that actually matches' as intuition to be earned by math/experiments rather than a decided fact.
- (d) Added caveat that UVᵀ is up to SVD degeneracy when singular values coincide, and specified compact SVD.
- (d) Corrected the msign RMS formula to √(r/(nm)) with n×m rank-r matrix, and framed the alignment factor as scaling like 1/√(max(n,m)) rather than an exact simplification.
```