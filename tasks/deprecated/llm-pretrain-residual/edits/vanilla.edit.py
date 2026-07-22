"""Vanilla residual stream baseline — rigorous codebase edit ops.

No-op: the template already implements the default residual stream design
(resid_lambdas * x + x0_lambdas * x0 + value embedding gating), so this
baseline leaves the editable regions unchanged.

This is the nanochat default: per-layer residual scaling, initial embedding
blending, and ResFormer-style value embeddings with sigmoid gating.
"""

OPS = []
