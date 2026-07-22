"""Score spec for ai4bio-protein-function-prediction."""
from mlsbench.scoring.dsl import *

# ---------- Beta-lactamase (regression) ----------
# Refs: median of cnn=0.689 / transformer=~0.21 / dgl_gcn=0.290 → ref ~0.45 (mid-baseline).
term("spearman_Beta",
    col("spearman_Beta").higher().id()
    .bounded_power(bound=1.0))

term("pearson_Beta",
    col("pearson_Beta").higher().id()
    .bounded_power(bound=1.0))

term("mse_Beta",
    col("mse_Beta").lower().id()
    .bounded_power(bound=0.0))

# ---------- Fluorescence (regression) ----------
# Refs: median of cnn=0.683 / transformer=~0.55 / dgl_gcn=0.671 → ref ~0.65.
term("spearman_Fluorescence",
    col("spearman_Fluorescence").higher().id()
    .bounded_power(bound=1.0))

term("pearson_Fluorescence",
    col("pearson_Fluorescence").higher().id()
    .bounded_power(bound=1.0))

term("mse_Fluorescence",
    col("mse_Fluorescence").lower().id()
    .bounded_power(bound=0.0))

# ---------- Solubility (binary classification) ----------
term("pr_auc_Solubility",
    col("pr_auc_Solubility").higher().id()
    .bounded_power(bound=1.0))

term("f1_Solubility",
    col("f1_Solubility").higher().id()
    .bounded_power(bound=1.0))

term("accuracy_Solubility",
    col("accuracy_Solubility").higher().id()
    .bounded_power(bound=1.0))

term("roc_auc_Solubility",
    col("roc_auc_Solubility").higher().id()
    .bounded_power(bound=1.0))

setting("Beta", weighted_mean(("spearman_Beta", 1.0), ("pearson_Beta", 1.0), ("mse_Beta", 1.0)))
setting("Fluorescence", weighted_mean(("spearman_Fluorescence", 1.0), ("pearson_Fluorescence", 1.0), ("mse_Fluorescence", 1.0)))
setting("Solubility", weighted_mean(
    ("pr_auc_Solubility", 1.0),
    ("f1_Solubility", 1.0),
    ("accuracy_Solubility", 1.0),
    ("roc_auc_Solubility", 1.0),
))

task(gmean("Beta", "Fluorescence", "Solubility"))
