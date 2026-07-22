**Mocking:** A world-class AI researcher

Let me reason from the structure of the task, not from a single experiment.

First, what is the bottleneck I'm looking at? The target is a supervised imputation task: I'm given a sequence with 25% missing, and I must predict those missing values given the observed values. The naive assumption is: it's a regression problem on the masked values. The naive implementation is a single linear head regressing to the mask. The reference models use this, and it works, but I suspect it's far from optimal for a multivariate, temporal task.

So I want to be precise about the *mechanism* I'm questioning. The linear head is a fixed, low-dimensional mapping from the input to the mask. That's a bottleneck. Why? Because the mask is a *sparse* representation of the imputation task. It throws away position information and forces all the signal into a tiny output layer. Worse, the mask is a binary, 0/1 matrix. A linear layer must learn a dense mapping between two very different representations, which is information loss by construction.

This is a representation bottleneck I can *see* in the architecture. A good analogy: the mask is a compressed, lossy code of the imputation task, and I'm asking a linear layer to decode it. That's the wrong layer for a decoding problem.

So the leap: turn the *target* into a representation I can decode, not a fixed output I must regress. The representation I want is the input itself. The mask is a *sparse* code for 'this position is missing.' So the natural decoding architecture is *attention* over the input sequence. Attention is exactly the mechanism for this: it can reconstruct any position by mixing the whole sequence. My intuition is that a single attention layer will be orders of magnitude more expressive than a single linear layer for this task.

Now, why does this cut against the consensus? The community's default for masked prediction is a linear head. That's the standard. My claim is that the mask is an *unsuitable* target representation, and the fix is to change the target, not to add more parameters. That's the non-trivial crux.

Why I keep myself honest: I'm not claiming a single attention head wins on its own. I must compare it to the linear head on the same data, same architecture size, to isolate the effect of the target representation from the effect of model complexity. Decision rule: if the attention-decoded target matches or improves the linear head without increasing parameters, I keep it; if it doesn't clearly help on the masked metric, I revert. I'm also cautious that this is a *sparse* target, and attention may be more sensitive to missingness than a linear regression.

One other lever to consider: since attention is a global mechanism, the imputation should be coherent across the sequence. That coherence is what I'd expect to win on the masked metric (which is a whole-region, not per-point, evaluation). I'll treat the sequence as a sequence, not as a collection of points.

**Core idea:** Replace the linear head regressing to the mask with a single self-attention layer over the input sequence: treat the mask as a *target representation* to be decoded, not as the output to regress.

**Non-trivial crux:** The mask is a lossy, sparse code of the task; the gain comes from changing the *target representation* (attention-decoding the input) rather than from adding parameters, and this must be validated on the masked metric against a comparable linear-head baseline.