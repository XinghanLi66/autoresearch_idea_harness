"""Score spec for llm-ttt-adaptation."""
from mlsbench.scoring.dsl import *

term("val_loss",
    col("val_loss_ttt-eval-345m").lower().id()
    .bounded_power(bound=0.0))

term("wikitext2_ppl",
    col("wikitext2_ppl_ttt-eval-345m").lower().id()
    .bounded_power(bound=1.0))

term("lambada_ppl",
    col("lambada_ppl_ttt-eval-345m").lower().id()
    .bounded_power(bound=1.0))

setting("ttt-eval-345m", weighted_mean(
    ("val_loss", 2.0),
    ("wikitext2_ppl", 1.0),
    ("lambada_ppl", 1.0),
))

task(gmean("ttt-eval-345m"))
