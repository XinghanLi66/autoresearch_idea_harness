# Pilot CoT 68 — 苏剑林 Jianlin Su · first_principles

- researcher: **苏剑林 Jianlin Su**  ·  style: 数学第一性推导驱动·想法写成推导+代码+实验博客·化简成优雅闭式解(RoPE)

- case: 苏剑林 (Jianlin Su) 案例6：扩散模型太"重" → 用"拆楼+建楼"只靠高斯代数推出完整 DDPM

- route: first_principles  ·  case score: 8/10

- lengths (qwen2.5-32b tokens): user(setup) **168** · assistant(cot) **1215** · whole **1440**

- fact-check changes: 3

## user (setup)

```plaintext
It is mid-2022. Diffusion models (DDPM) are producing striking image samples and clearly work, but every derivation on offer is heavy: variational lower bounds, energy functions, score matching, Langevin dynamics, SDEs. To even understand why the training objective is a simple noise-prediction MSE, a reader must wade through machinery that seems disproportionate to the final loss. You suspect DDPM is actually simpler than VAE or GAN — that the standard derivations are historical accidents, not the essence. Your task: starting from what MUST be true about the forward and reverse processes, derive the exact DDPM training objective and sampler using only elementary probability, grounded in an intuitive picture, without invoking variational inference or score matching. Produce a genuinely novel derivation route to the same closed-form result.
```

## assistant (cot)

```plaintext
Let me start where I always start: not with "which tool should I use" but with "what do I actually want, written as something I can solve." Everyone reaches for the variational lower bound because Ho et al. did, and because the object is a Markov chain and VAE-style machinery fits Markov chains. But that's imitation, not necessity. I only trust what I can re-derive myself. So let me ask: what is the minimal set of things that MUST hold, and can the objective be forced out of those alone?

Here is the picture I want first, because a picture makes the algebra obvious. A clean sample x0 is a finished building. Pure Gaussian noise xT is a pile of bricks. The forward process is demolition — knock the building down brick by brick over T steps. The reverse process is reconstruction — learn, at each step, to put one layer of bricks back. If I can learn a single "put-back-one-step" map, then starting from a fresh pile of bricks and applying it repeatedly must assemble a new, valid building. That is generation. This is nothing mystical; it's just a Markov chain, but now I can see it.

Now let the math fall onto the picture. Demolition, one step: x_t = alpha_t x_{t-1} + beta_t eps_t, with eps_t standard Gaussian. I want an invariant here — variance should not blow up or vanish as I stack many steps, otherwise the noise level drifts uncontrollably. The cleanest way to hold total scale fixed is alpha_t^2 + beta_t^2 = 1. That single normalization is the invariant that keeps the whole chain well-behaved; I choose alpha_t near 1 so each step removes only a little of the building.

Because a sum of independent Gaussians is again Gaussian, I don't need to crawl step by step. Composing the recursion collapses to x_t = bar_alpha_t x0 + bar_beta_t eps, with bar_alpha_t the product of the alphas and bar_beta_t = sqrt(1 - bar_alpha_t^2) forced by the same normalization. This is pure Gaussian bookkeeping — no lower bound anywhere. It also gives me a practical gift for free: I can jump to any noise level in one shot when training.

Now the reverse step, and here is the leap that avoids the heavy machinery entirely. I do NOT need to write down a variational bound over the whole chain. I just ask: given the one-step demolition x_t = alpha_t x_{t-1} + beta_t eps_t, what is the most natural thing to learn to invert it? The step injected exactly eps_t. So let a network predict that injected noise. If I could name the noise, I could subtract it and recover x_{t-1} up to the fixed coefficients. So the objective is simply: make the network's predicted noise match the actual injected noise, in mean-square. Feeding the network the one-shot form x_t = bar_alpha_t x0 + bar_beta_t eps, the target is that same eps. A careful accounting of the per-step coefficients puts a factor bar_beta_t/beta_t in front, but that's a constant reweighting per t.

Here I apply my parsimony rule: keep only what the derivation forces, and let experiment decide the rest. The clean derivation hands me the reweighted loss; the coefficient is just a per-t constant. The decision rule I'd use: if dropping it (going to plain ||eps - eps_theta||^2) doesn't clearly hurt sample quality, prefer the simpler unweighted form — it's the same target, cheaper to reason about. And in fact this matches what Ho et al. reported: the unweighted 'simple' objective is what actually trains well in practice, so my parsimony instinct is consistent with the empirical record rather than a claim I'm asserting fresh. I won't claim which wins in my own run before running it; I'd try both and let the images and the loss curves choose.

Sampling then just runs the picture backward: x_{t-1} = (1/alpha_t)(x_t - beta_t eps_theta(x_t, t)) + sigma_t z, iterated from xT ~ N(0, I). The time index t I feed in through a sinusoidal-style embedding, exactly the position-encoding idea reused from the Transformer.

Notice what never appeared: no free energy, no score matching, no Langevin, not even a variational bound. That's the point I'm betting on against the consensus — the machinery was never necessary; elementary Gaussian identities plus one invariant reach the identical closed form. And because I understand it this cleanly, I can re-derive the SAME DDPM from a VAE view, a Bayesian-denoising view, an SDE view — multiple independent derivations meeting at one answer is what "truly understanding it" means, and it's what reveals the boundaries between DDPM, DDIM, and the SDE picture.

**Core idea:** Derive DDPM using only elementary Gaussian algebra plus the variance-preserving invariant alpha^2 + beta^2 = 1, grounded in a demolish-then-rebuild picture: the forward step injects a known Gaussian noise, so the reverse objective is forced to be "predict that injected noise" (MSE), and the sampler is its inversion — no variational bound or score matching required.

**Non-trivial crux:** The whole heavy apparatus collapses because you invert the chain ONE step at a time (predicting the per-step injected noise) rather than optimizing the full-chain likelihood; the exact derivation yields a per-t coefficient bar_beta_t/beta_t, and dropping it to plain ||eps - eps_theta||^2 is a parsimony choice (consistent with the 'simple' objective Ho et al. found trains best), not a mathematical identity.
```

## fact-check changes

```plaintext
- (a) Reframed the loss-weighting discussion: instead of implying I'd discover the winner via my own run, noted that the unweighted 'simple' objective is the empirically preferred form reported by Ho et al., keeping the decision-rule framing without asserting an unrun result.
- Clarified the sinusoidal time embedding is reused from the Transformer (period-appropriate) to avoid a vague provenance claim.
- Adjusted the Non-trivial crux to attribute the empirical support for the unweighted objective to Ho et al. rather than leaving it as an unvalidated parsimony bet.
```