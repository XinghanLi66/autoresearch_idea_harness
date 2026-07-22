**Mocking:** Alexey

Let me start from the mechanism I want to target, not from an existing defense. Membership inference exploits the fact that train points are fit more tightly than non-train points — the model's decision boundary is closer to train examples. So the most direct way to reduce membership signal is to make train and non-train examples *look statistically more similar* under the model. That's a statement about the *distribution* of model outputs, not just the mean or any single point.

Now, what's the simplest distributional regularizer I can put on the logits? A Gaussian prior. If I place a Gaussian prior on the logit vector for each example, and then optimize KL divergence between the empirical logits and this prior, that directly pulls the logits toward a common Gaussian shape — pushing both train and non-train logits toward the same distribution. That's exactly what I need: a single term that makes train and non-train distributions converge.

Why does this cut against the consensus? Most defenses add a separate regularization term (e.g. entropy penalty) and treat privacy and utility as competing objectives. My prior-based formulation *combines* them into one objective. The intuition is that the prior itself is doing the work of both reducing overfitting and matching distributions — the entropy penalty is no longer needed as a separate term.

I have to be careful here, because naively putting a Gaussian prior on the logits could push the whole distribution toward zero, which would destroy discriminative power. So I keep the standard cross-entropy loss and add the KL term as a small coefficient — the prior regularizes, but doesn't dominate. And I'll use a *per-class* prior rather than a global one, so the prior isn't pulling all classes toward the same mean.

One more thing to watch: the prior should be learned rather than fixed. If I fix the prior, it might not match the true logits distribution well, especially early in training. A learned prior adapts, which means it can start from a broad prior that lets the model learn, then tighten as the logits settle — that's a natural annealing behavior that I don't have to hand-tune.

How do I check I'm actually moving in the right direction without waiting for full runs? Before running the full attack, I'd look at the logits histogram across train vs non-train splits at different epochs — if the prior is working, the two histograms should get closer over time. Decision rule: if the histograms diverge despite the prior, the prior isn't helping; if they converge and attack AUC drops while test accuracy stays stable, I've likely found something real.

Core idea: Use a per-class learned Gaussian prior on the logits, optimizing KL divergence to the prior, which pushes train and non-train logits toward a common distribution and collapses the membership signal — combined with cross-entropy, this replaces separate privacy terms.

Non-trivial crux: The prior must be per-class and learned (not fixed), and added as a small coefficient to cross-entropy — otherwise it can collapse discriminative information, and a fixed prior won't adapt to the evolving logits distribution.