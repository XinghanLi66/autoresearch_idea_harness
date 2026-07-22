**Core idea:** Introduce a cross-variable attention mechanism that lets the model dynamically attend to relevant exogenous features, with a learnable target-specific gate to suppress irrelevant feature influence.

**Non-trivial crux:** The gate must be trained to act on the *residual* of a baseline prediction (not the raw target) so it modulates error rather than overwriting a weak signal; without a strong baseline prediction to gate, the mechanism becomes an unconstrained regressor and loses its selective value.

---

**Reasoning chain:**

Start from the problem structure: in a many-to-one (MS) setup, the target's future depends on its own history AND a set of exogenous variables. The naive approach (stacking all features) gives the model no reason to distinguish between informative and noise variables — every input is treated equally, and irrelevant features can drown out the signal. This is the central *representation bottleneck*: the model is forced to learn which features matter by default rather than being helped to focus.

What's the mechanism that *could* help here? Attention, because it is the canonical way to let a model focus on a subset of its input. But plain self-attention among all variables would still treat the target like any other feature. What I want is a structure that *biases* attention toward the target's own history while allowing auxiliary variables to contribute only where they're relevant.

So I look at the analogy: a transformer with cross-attention between a target "query" and a set of "keys/values" provided by the exogenous features. This is exactly the architecture of the encoder-decoder attention in sequence-to-sequence transformers — use the decoder's target as query and the encoder's features as key/value. That's a strong hint: map the target's own history into a query, the exogenous features into key/values, and let the attention matrix select which features are relevant at each step.

But there's a hidden assumption to check: that the baseline prediction and the attention-based correction are equally important across the forecast horizon. In practice, the baseline is more reliable near the present and the attention correction is more valuable farther out. So the gate shouldn't be a flat scalar — it should be a *per-position* gate that lets the attention mechanism dominate where the baseline is weakest. The gate should be trained on the *residual* (prediction error), so it learns to correct the error rather than overriding a weak baseline.

Now the crux: if I gate the *raw target* (the signal itself), the gate has nothing to correct — a weak baseline prediction simply gets turned off, and the gate becomes an unconstrained regressor. The whole mechanism collapses back to an un-gated model. That's exactly the kind of silent failure I want to avoid. The fix is to gate the *residual* — then the gate *modulates the error*, and only the error, which is precisely the quantity it's meant to correct.

Why does this cut against the field? Most exogenous work either stacks everything or adds a fixed, non-learned mask — the "all inputs equally" default. A learnable, per-position gate trained on the residual is more expressive but also more fragile to mis-apply; you have to be careful about what you gate and how it's trained.

Implementation sketch (not code): in the decoder, produce a target-only query from the target's history, cross-attend to the full feature sequence, and feed the result through a gated residual where the gate is trained on the baseline's prediction error — the baseline prediction is the main term, the gated attention correction is the add-on.