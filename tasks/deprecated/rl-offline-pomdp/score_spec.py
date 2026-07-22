"""Score spec for rl-offline-pomdp (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("normalized_score_mon_hum_neu",
    col("normalized_score_mon_hum_neu").higher().id()
    .sigmoid())

term("normalized_score_val_hum_neu",
    col("normalized_score_val_hum_neu").higher().id()
    .sigmoid())

term("normalized_score_ran_hum_neu",
    col("normalized_score_ran_hum_neu").higher().id()
    .sigmoid())

setting("mon-hum-neu", weighted_mean(("normalized_score_mon_hum_neu", 1.0)))
setting("val-hum-neu", weighted_mean(("normalized_score_val_hum_neu", 1.0)))
setting("ran-hum-neu", weighted_mean(("normalized_score_ran_hum_neu", 1.0)))

task(gmean("mon-hum-neu", "val-hum-neu", "ran-hum-neu"))
