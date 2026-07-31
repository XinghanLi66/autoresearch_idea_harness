# Pilot CoT 67 — 苏剑林 Jianlin Su · empirical_anomaly

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例6：扩散模型太"重" → 用"拆楼+建楼"只靠高斯代数推出完整 DDPM

- route: empirical_anomaly  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **214** · assistant(cot) **1152** · whole **1423**

- fact-check changes: 3

## user (setup)

```plaintext
It's mid-2022. Diffusion models (DDPM) have become the hot new generative paradigm, but every derivation you can find leans on heavy machinery: variational lower bounds, score matching, energy functions, Langevin dynamics. You keep re-reading these papers and blog posts, and something bothers you. When you actually TRAIN a DDPM and watch it work, the training objective collapses to something absurdly simple — a network that just predicts the injected Gaussian noise, an L2 loss on epsilon. The sampler is a short iterative formula. The empirical recipe is almost trivial. Yet the standard 'story' explaining WHY it works requires graduate-level variational inference or stochastic calculus. That mismatch — a dead-simple, robust, reproducible training loop wrapped in intimidating theory — is exactly the kind of contradiction that makes you suspicious. Can you find a way to derive the exact DDPM training objective and sampler from first principles, using nothing heavier than basic probability, and produce an intuition anyone can picture? Propose a genuinely new way to understand it.
```

## assistant (cot)

```plaintext
Let me start from what I can actually measure, not from what the papers tell me the model is. I take a working DDPM, and I look at the training loop that people report converging robustly across datasets. The loss is ‖ε − ε_θ‖². That's it. A network predicts the noise you added. The sampler is a one-line update repeated ~1000 times. This trains stably, it's reproducible, everyone gets it to work. Now I open the standard derivations and they hand me a variational lower bound over a whole Markov chain, or a score-matching argument with Langevin dynamics. There's a contradiction here that most people just explain away: 'the ELBO simplifies to this after a lot of algebra.' But if the endpoint is THIS simple and THIS robust, the heavy machinery is almost certainly not doing the essential work — it's scaffolding around a much simpler object. I only trust what I can re-derive myself, and I refuse to believe DDPM is intrinsically harder than VAE or GAN when its actual training code is simpler than both.

So my instinct: the difficulty is in the framing, not in the math. Let me first build a picture anyone can hold in their head, then let the algebra fall onto the picture. Think of x₀ as a finished building and x_T as a pile of loose bricks — pure Gaussian noise. The forward process is demolition: knock the building down a little at a time over T steps until only bricks remain. The reverse process is construction: learn to put one layer of bricks back, one step at a time. If I can learn a single 'restore-one-step' map, then starting from a random pile of bricks and applying it repeatedly, I build a NEW building — a new sample. No variational anything in that sentence. Just: destroy in small increments, learn to undo one increment.

Now force the math onto the picture using only facts a first-year probability student knows. Forward single step: x_t = α_t x_{t−1} + β_t ε_t, with α² + β² = 1 so variance stays normalized — 'keep most of the building, chip off a little.' The only nontrivial fact I need is that a sum of independent Gaussians is Gaussian. That instantly collapses the whole chain: I can jump from x₀ to any noise level in one shot, x_t = ᾱ_t x₀ + β̄_t ε̄, with ᾱ_t the product of the α's. This is huge and it's elementary — it means during training I never simulate the chain; I sample a random step t and a single Gaussian and I'm at the right noise level directly.

For the reverse step, I don't invoke a posterior over the whole chain. I ask the plainest question: given x_t, what's the cleanest thing to predict that lets me step back? Since x_t is literally ᾱ_t x₀ + β̄_t ε̄, knowing the noise ε is equivalent to knowing x₀. So train ε_θ to predict the injected noise. When I do the variance-reduction bookkeeping carefully, a coefficient like β̄_t/β_t appears in front of each term. But the well-known DDPM finding is that dropping this per-step weight — using the bare ‖ε − ε_θ‖² — is what people report works well in practice, so I'll follow that simpler weighting rather than carry the theoretically-derived constant; the decision here is parsimony plus matching the reported recipe, not a claim that I've re-run the ablation myself. Feed the step index t in via a time embedding, the same positional-encoding trick we already understand. The sampler is then just: subtract the predicted noise, rescale by 1/α_t, add a little fresh noise σ_t z, iterate from the brick pile.

How do I keep myself honest? Two checks, and no fabricated verdicts. First, internal consistency: my derived objective and sampler must reduce term-for-term to the published DDPM equations — if a coefficient disagrees, my picture is wrong, not the working code. Second, multi-view cross-validation: a single clean derivation could be a lucky coincidence, so I'll re-derive the SAME model from independent starting points — as an autoregressive VAE, as Bayesian denoising, and via an SDE/continuous-limit reading. If all roads land on the same ε-prediction objective, then I truly understand it, and I also see the exact boundaries between DDPM and its variants. Decision rule: keep the lightest derivation that reproduces the exact known formulas; only reach for heavier machinery where the light tool provably cannot reach.

**Core idea:** Reframe diffusion as 'demolish a building into bricks, then learn to rebuild one layer at a time,' and derive DDPM's exact noise-prediction objective and sampler using only elementary Gaussian algebra (sum of independent Gaussians is Gaussian) — no variational inference, no score matching.
**Non-trivial crux:** The one-shot forward jump x_t = ᾱ_t x₀ + β̄_t ε̄ makes 'predict the injected noise' equivalent to 'know x₀,' collapsing the whole chain; the theoretically-derived β̄_t/β_t weight can be dropped to match DDPM's reported simplified objective on parsimony grounds — and the simplicity is trustworthy only because the single clean derivation is corroborated by re-deriving the same model from multiple independent viewpoints.
```

## fact-check changes

```plaintext
- (a) Removed the fabricated empirical verdict that the β̄_t/β_t reweighting 'doesn't clearly help' and 'trains best anyway'; recast it as following DDPM's reported simplified objective on parsimony grounds, explicitly noting the ablation wasn't re-run here.
- (a) Adjusted the crux one-liner correspondingly to remove the asserted 'doesn't clearly help' outcome.
- (c)/factual: changed 'later as an SDE discretization' to an SDE/continuous-limit reading to avoid implying a fixed chronological ordering while keeping the multi-view cross-validation intent.
```