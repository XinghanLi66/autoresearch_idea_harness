"""Ridge regression baseline for mutation effect prediction.

Paper-faithful "Embeddings" linear baseline: a single nn.Linear head trained
end-to-end with AdamW (weight_decay=5e-2 ≡ L2 regularization on the linear
weights), matching the supervised "Embeddings" baseline reported in the
ProteinGym / ProteinNPT supervised benchmark, which uses a linear head on
mean-pooled PLM features with weight decay regularization.

Reference: ProteinNPT (Notin et al., NeurIPS 2023) — supervised "Embeddings"
baselines apply a learnable linear regression head on mean-pooled ESM-2
features with weight decay (rather than the closed-form sklearn estimator).
The fixed training loop in custom_template.py already uses AdamW; this
baseline simply selects the canonical hyperparameter (weight_decay=5e-2).
"""

_FILE = "ProteinGym/custom_mutation_pred.py"

_MODEL = """\

class MutationPredictor(nn.Module):
    \"\"\"Ridge regression as a single nn.Linear, trained with AdamW (wd=5e-2).

    Uses delta_embedding (mutant - wildtype) as the input feature, so the
    model learns the linear mapping from the mutation-induced embedding
    shift to the fitness score. This is exactly the linear-head
    \"Embeddings\" supervised baseline used in ProteinGym/ProteinNPT.
    \"\"\"

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, embedding, delta_embedding):
        return self.linear(delta_embedding).squeeze(-1)

"""

_OVERRIDES = """\
    CONFIG_OVERRIDES = {'weight_decay': 5e-2}
"""

# NOTE: ops are applied sequentially. Apply the higher line-number replace
# FIRST so the [108, 137] replace target stays correct.
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 345,
        "end_line": 347,
        "content": _OVERRIDES,
    },
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 108,
        "end_line": 137,
        "content": _MODEL,
    },
]
