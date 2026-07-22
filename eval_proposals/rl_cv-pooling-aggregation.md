**Mocking:** Danielle Pelleg

Let me start from what I actually observe rather than from a paper. I train a ResNet-56 on CIFAR-100 with plain GAP and get a clean baseline — say 70.2%. Now I swap in a few existing alternatives: GMP, GeM with its learned exponent, and the simple average+max combo. The numbers move, but none clearly beats the others across all three architectures. That's the first thing that doesn't fit the narrative: if one of these were obviously superior, I'd expect it to dominate. So the 'better pooling' framing is premature.

So I flip the question. Instead of asking 'which pooling wins?', I ask: what do these variants actually change about the representation? I compute the feature statistics at the pool location — mean, variance, kurtosis, and pairwise covariance across channels. Across the three networks, I see a consistent pattern: the distributions differ in their tails and covariance structure, not just their means. GAP smooths away the outliers and flattens covariances; GMP does the opposite, emphasizing them; GeM sits somewhere in between, but still uses a scalar exponent per channel.

That's the insight I want to lean on: a pooling layer isn't just a summary statistic — it's a distribution transformer. If the optimal summary depends on the tail strength and channel correlations, then a single scalar like GeM's exponent may be too rigid. A learnable transform that can adjust each channel's tail weight and inter-channel interaction could be more flexible.

Here's where I keep myself honest. I don't want to invent a fancy mechanism without knowing whether it's worth the parameters. So my plan is: build the simplest version that captures the two axes (tail and correlation), compare it to GeM and plain GAP/GMP on the same budget, and only keep it if it's measurably better. If it's not, I'll default to the simpler, cheaper option.

The simplest way to encode tail weight and covariance is to use a linear transformation plus a power-law term. Specifically, for each channel c:
1. Compute the mean μ_c and covariance matrix Σ over spatial locations.
2. Apply a linear transformation A_c to the flattened feature vector: z = A_c · x.
3. Raise each element to a channel-specific power p_c: y_c = |z|^p_c.
4. Sum over spatial dimensions to get the pooled output: h_c = sum(y_c).
This gives me one learnable matrix A and one scalar p per channel — modest parameter count, and it's exactly the 'tail + covariance' form I want.

Why this cuts against the consensus: everyone's building fancier attention or adaptive layers. My bet is that a plain learnable linear+power form, tuned properly, may outperform those heavier options because it directly targets the two axes I've diagnosed as mattering, and it's cheap enough to scale across many channels.

One caveat I must check: this power-law form can blow up when p > 1 and activations are large. So I need a safeguard — a small constant ε added inside the power term to keep the magnitude bounded.

Now the decision rule. I'll run each variant (plain GAP/GMP, GeM, and my linear+power form) on ResNet-56/CIFAR-100 and VGG-16-BN/CIFAR-100, using the same training recipe. If my variant is within 0.2% of GeM and has fewer parameters, I keep it; otherwise I drop it and use GeM. The hidden MobileNet/FashionMNIST acts as a holdout — if it behaves differently, that's a red flag I should investigate further.

**Core idea:** Replace fixed pooling with a learnable linear+power-law transform that adjusts each channel's tail weight and inter-channel covariance, treating pooling as a distribution-shaping operation.

**Non-trivial crux:** The power-law term must have a small additive constant ε to prevent unbounded growth when p > 1; otherwise the method becomes numerically unstable.