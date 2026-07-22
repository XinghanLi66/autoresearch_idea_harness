**Mocking:** [AI Researcher's Deep Reasoning]

Let me start from a property rather than a trick. I want a pooling strategy that is invariant to spatial translation of features but still preserves some notion of spatial distribution. Global average pooling is translation-invariant by construction — shifting the feature map doesn't change the mean. That's a good property to keep. What it discards, though, is the entire shape of the distribution. Two feature maps with identical mean but different spatial concentration will be treated identically. That's the gap.

What's the most direct, minimal way to add spatial information while staying translation-invariant? The first-order statistic (mean) is translation-invariant. The second-order statistic (variance) is not, because it measures spread relative to the origin. But the second central moment — variance computed about the mean — is invariant under translation. So I should include the spatial variance of each channel, computed about the channel's own mean, not about the origin. That's the second-order term I want.

Now the invariant question. If I take [mean, variance] of each channel, does that form a meaningful, translation-invariant feature? Yes: mean captures the location of the distribution, variance captures its spread, and both are computed relative to the mean, so neither changes under spatial translation. This is a clean, minimal invariant feature vector of length 2C that respects the invariance property I want. It's also cheap — I'm reusing the mean and then computing a second pass over the same feature map.

How do I fit this into the interface? The input is [B, C, H, W]. For each channel, I'll compute the spatial mean and then the spatial variance about that mean. Concatenate [mean, variance] across channels to get [B, 2C], then project back to [B, C] with a 1×1 convolution. The projection is needed to maintain output dimensionality exactly, and it should be learnable so the network can decide whether to keep, suppress, or combine the two statistics.

Why not go to higher-order moments? The second-order term already captures a critical spatial property — concentration — with minimal cost. Adding third-order (skewness) or higher would cost more and likely provide diminishing returns. I'll keep it at two statistics for parsimony.

Why not spatial attention maps or learnable weights? Those require data and training, and I can't access training data within the pooling layer. I'm restricted to data-free, architecture-adaptive design. The mean-variance invariant is data-free by construction. I can build it from the feature map itself.

What about the 1×1 case? When H=W=1, the spatial mean and variance reduce to the single value and zero variance, so the variance term becomes trivial. The invariant is still valid, and the 1×1 case is a natural special case of the general formulation — no extra handling needed.

Now the crux. The mean and variance must be computed per channel, and the variance must be computed about the per-channel mean, not the origin. That's the invariant part. If I compute variance about the origin, the translation invariance is lost. I must keep the order: compute mean, then compute variance about that mean, then concatenate. Any shortcut that computes variance relative to the origin breaks the invariant and is wrong.

Core idea: Use the spatial mean and the second central moment (variance about the mean) of each channel as a translation-invariant feature, then project back to the original channel dimension with a 1×1 convolution.

Non-trivial crux: The variance must be computed about the per-channel mean (second central moment) to preserve translation invariance; computing variance about the origin breaks the invariant and is incorrect.