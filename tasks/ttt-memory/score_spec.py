"""Score spec for ttt-memory (Titans / Nested Learning, nanoGPT 345M).

NIAH metrics come from the standalone niah-eval-345m test_cmd which applies
NTK-aware RoPE base scaling so that contexts past the vanilla extrapolation
cliff (~1.5x training length) stay roughly in-distribution. We use:

  niah_pass_1536 (binary, ~1.5x train_ctx): with NTK scaling, deep memories
                                            hold rank=1 here; cliff for
                                            vanilla. Discriminates
                                            depth>=2 vs. linear memory.
  niah_pass_2k   (binary, 2x train_ctx):    stretch metric — mostly tells
                                            us whether memory
                                            consolidation generalizes at
                                            all past the cliff. Floors at
                                            0.0 for most baselines.
  niah_rank_2k   (continuous, lower better, 2x train_ctx): mean rank of
                                            the first token in the six-letter
                                            answer sequence. Scored after
                                            log1p compression so raw rank
                                            outliers do not dominate.

niah_pass_1280 is recorded but excluded from scoring because it saturates
near 1.0 across baselines (no discriminative signal).

Normalization uses dynamic leaderboard anchors: the worst baseline is the
0-point floor and the best baseline is the 50-point anchor. The bounded_power
curve gives credit for improving past that and saturates near each metric's
hard upper bound.
"""
from mlsbench.scoring.dsl import *

# ── Pretraining quality (in-distribution LM) ────────────────────────────────
term("val_loss",
    col("val_loss_gpt-345m").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl",
    col("wikitext2_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

term("lambada_ppl",
    col("lambada_ppl_gpt-345m").lower().id()
    .bounded_power(bound=1.0))

# ── NIAH passkey retrieval (hidden, NTK-scaled) ─────────────────────────────
term("niah_pass_1536",
    col("niah_pass_1536_niah-eval-345m").higher().id()
    .bounded_power(bound=1.0))

term("niah_pass_2k",
    col("niah_pass_2k_niah-eval-345m").higher().id()
    .bounded_power(bound=1.0))

term("niah_rank_2k",
    col("niah_rank_2k_niah-eval-345m").lower().log1p()
    .bounded_power(bound=1.0))

# ── Downstream zero-shot ────────────────────────────────────────────────────
term("arc_easy",
    col("arc_easy_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("hellaswag",
    col("hellaswag_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("piqa",
    col("piqa_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

term("winogrande",
    col("winogrande_lm-eval-345m").higher().id()
    .bounded_power(bound=100.0))

# ── Setting: combine within each test_cmd group ─────────────────────────────
setting("gpt-345m", weighted_mean(
    ("val_loss", 2.0),
    ("wikitext2_ppl", 0.75),
    ("lambada_ppl", 0.75),
))

setting("lm-eval-345m", weighted_mean(
    ("arc_easy", 1.0),
    ("hellaswag", 1.0),
    ("piqa", 1.0),
    ("winogrande", 1.0),
))

setting("niah-eval-345m", weighted_mean(
    ("niah_pass_1536", 1.0),
    ("niah_pass_2k", 1.0),
    ("niah_rank_2k", 1.0),
))

task(gmean("gpt-345m", "lm-eval-345m", "niah-eval-345m"))
