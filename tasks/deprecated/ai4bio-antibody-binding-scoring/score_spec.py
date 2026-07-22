"""Score spec for ai4bio-antibody-binding-scoring.

Zero-shot Spearman targets anchored to AbBiBench-reported ESM-2 numbers
(https://github.com/MSBMI-SAFE/AbBiBench README leaderboard):
  ESM-2 mean PLL (3B, paper): 3gbn_h1=0.23, 4fqi_h1=-0.02, 4d5_her2 n/a
Pre-computed ESM-2 3B mean PLL on our CSVs (replicating the AbBiBench
upstream off-by-one indexing in
`vendor/external_packages/AbBiBench/models/ESM-2/get_model_log_likelihood.py:85-96`,
where `probs[0, i, token_idx]` is read at `i` corresponding to the
tokenized-without-BOS position while the model output includes the BOS
token — i.e. each token's probability is read at its predecessor's
position, rather than the standard textbook mean PLL):
  3gbn_h1=+0.23, 4fqi_h1=-0.02, 4d5_her2=-0.20
ProteinMPNN / ESM-IF1 structural ceiling (AbBiBench leaderboard, not
available to the agent): 0.59 / 0.61 on 3gbn_h1, 0.61 / 0.65 on 4fqi_h1,
0.32 / n.a. on 4d5_her2.

Normalization uses dynamic leaderboard anchors: the worst baseline is the
0-point floor and the best baseline is the 50-point anchor. The structural
ceiling above is context for plausibility, not a hand-coded reference.
"""
from mlsbench.scoring.dsl import *

term("spearman_influenza",
    col("spearman_influenza").higher().id()
    .bounded_power(bound=1.0))

term("spearman_sars",
    col("spearman_sars").higher().id()
    .bounded_power(bound=1.0))

term("spearman_her2",
    col("spearman_her2").higher().id()
    .bounded_power(bound=1.0))

setting("influenza", weighted_mean(("spearman_influenza", 1.0)))
setting("sars", weighted_mean(("spearman_sars", 1.0)))
setting("her2", weighted_mean(("spearman_her2", 1.0)))

task(gmean("influenza", "sars", "her2"))
