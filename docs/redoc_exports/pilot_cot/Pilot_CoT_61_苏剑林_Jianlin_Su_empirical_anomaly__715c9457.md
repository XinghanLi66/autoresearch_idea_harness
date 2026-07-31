# Pilot CoT 61 — 苏剑林 Jianlin Su · empirical_anomaly

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例4：Muon 火了不止复现 → 把它重新推导成"谱范数下的最速下降"

- route: empirical_anomaly  ·  case score: 9/10

- lengths (qwen2.5-32b tokens): user(setup) **230** · assistant(cot) **1205** · whole **1492**

- fact-check changes: 4

## user (setup)

```plaintext
It's late 2024. A new optimizer called Muon is going viral on Twitter. It only touches matrix-shaped parameters W (leaving vectors/scalars to Adam), and its core update is: take the momentum M, then apply a matrix 'sign' via a cheap Newton–Schulz iteration with three coefficients (roughly 3.44, -4.78, 2.03), and step along that. People report it beats AdamW on real training runs. The original release gives working code and those coefficients, but essentially no derivation — just a terse comment. Most of the community is busy reproducing it, tuning hyperparameters, and reporting numbers. You have the code and you can run it. But the reproducible fact that nags you is oddly specific: Muon does something dramatically different to matrices than to vectors, and it clearly wins — yet the standard optimizer story (Adam, SGD) treats every weight as an independent scalar and says nothing about why 'being a matrix' should matter to the optimizer at all. Propose a genuinely novel way to understand and improve this method.
```

## assistant (cot)

```plaintext
Let me start from the one thing I can measure and can't explain away: Muon only acts on matrix parameters, and on those it reportedly beats AdamW. Everyone is reproducing it; nobody is bothered by the contradiction that bothers me. The whole Adam/SGD toolkit is elementwise — it implicitly says a weight matrix is just a bag of loose scalars, a Frobenius-norm view. If that view were the whole story, treating a matrix specially should not systematically help. But it does seem to help, reproducibly. So the standard story is missing something the data already knows. I refuse to file that under 'empirically Muon is better, moving on.'

Second stubborn fact: the update is msign(M), a *matrix* sign, not an elementwise one. Let me not accept the recipe as magic. Write M = UΣVᵀ. What does msign do? It throws away the singular values and keeps only the orthogonal factors — all nonzero singular values pushed to 1. Now the scalar analogy jumps out: for a 1×1 matrix, msign(x) = x(x²)^{-1/2} = sign(x). So Muon is nothing exotic — it's the sign-momentum idea (the Lion/Tiger family I've played with) *generalized from scalars to matrices via the SVD*. The Newton–Schulz iteration is just a cheap polynomial that pushes singular values toward 1 without paying for a full SVD. Good — the mystery is now 'why is normalizing the spectrum the right move?'

Here's where I stop reproducing and start deriving. I don't want a heuristic; I want the objective this update *solves*. Every reasonable optimizer step is: linearize the loss, then pick the direction that decreases it fastest under some constraint on step size. SGD picks steepest descent under a Frobenius-norm constraint — that's the scalar-bag geometry. So the real question is: *under which norm is Muon the steepest-descent direction?* Write ΔW = (step size) × direction Φ, and maximize the alignment Tr(GᵀΦ) subject to a constraint on Φ. If the constraint is the spectral norm ‖Φ‖₂ = 1, then by duality the maximum of Tr(GᵀΦ) is the nuclear norm ‖G‖_* = Σᵢσᵢ(G), and — this is the punchline — the maximizer is exactly Φ = msign(G). So it closes in one line: SGD is steepest descent in Frobenius geometry, Muon is steepest descent in *spectral* geometry. Changing the norm is changing the geometry. That's the whole content, and it's the answer to the contradiction I started from: 'being a matrix' matters because a matrix has a spectrum, and the spectral norm is a strictly more matrix-aware notion of step size than the scalar-bag Frobenius one.

Now, keeping myself honest, three things. (1) The Newton–Schulz coefficients: I won't trust three numbers I didn't derive. They should approximate t^{-1/2} on the singular-value range — a Taylor expansion near t=1 gives a defensible starting polynomial — and then I'd fit the coefficients by sampling the actual singular-value distribution of random matrices and optimizing the fit. Decision rule: prefer whatever coefficients drive the spectrum toward 1 fastest and most stably over the realistic range, not whatever went viral. (2) Reusability: to let people swap Muon in without re-searching learning rates, I'd try to align its update RMS to Adam's typical per-element scale. msign has an analytic RMS (√(r/nm)), so the scaling should collapse to a clean factor like 0.2/√max(n,m). This is the kind of closed-form tidiness I trust — derived, then checked against runs, not blindly tuned. (3) The invariant that must not drift at scale: if Muon is spectral steepest descent, the spectral norm of the weights is the quantity to watch. If it can grow unboundedly, big runs may destabilize. Weight decay is then not decoration — it bounds ‖W‖₂. And if I see a specific failure at scale (say attention logits blowing up, i.e. the spectrum of QK products running away), the fix should follow from 'spectral norm must stay bounded' — clip the offending factors by roughly √(threshold/observed) per head — rather than from blind grid search. Whether that clip actually stabilizes a given run is something I'd have to measure.

I'd write this all up as motivation → derivation → code → experiment, publicly, because publishing forces me to make every 'I think' into 'I proved' or 'I measured.'

**Core idea:** Muon's msign update is exactly steepest descent under a *spectral-norm* constraint (SGD is steepest descent under the Frobenius norm) — changing the norm is changing the geometry; this closed-form view derives the Newton–Schulz polynomial, an Adam-reusable RMS-aligned scaling, and a spectral-norm-must-stay-bounded principle that suggests weight decay and a candidate production-scale logit-clip fix.

**Non-trivial crux:** The leap is reading msign not as a magic matrix-sign trick but as the maximizer of Tr(GᵀΦ) under ‖Φ‖₂=1 (whose dual is the nuclear norm); the caveat is that this is a *linearized single-step* optimality — it justifies the direction and the boundedness invariant, but not cross-phase mixing (e.g. Adam-pretrain then Muon-finetune), which the theory does not cover and must be treated as unresolved.
```

## fact-check changes

```plaintext
- (a) Removed asserted confirmation that fitted coefficients / RMS scaling / logit-clip empirically work; reframed as things to test with a decision rule and to be measured against runs.
- (a) Softened 'it clearly wins' / 'beats AdamW' to reported results in setup and cot, since these are claims to verify not established at that moment.
- (d) Softened 'could not systematically help' and 'will destabilize' to 'should not' / 'may', preserving the reasoning without over-absolutizing.
- (e) Replaced 'magic coefficients'/'magic Newton–Schulz coefficients' with plain descriptions; no non-English terms were present.
```