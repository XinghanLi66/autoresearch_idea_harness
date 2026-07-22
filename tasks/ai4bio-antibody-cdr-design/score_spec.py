"""Score spec for ai4bio-antibody-cdr-design."""
from mlsbench.scoring.dsl import *

term("aar_epitope_group",
    col("aar_epitope_group").higher().id()
    .bounded_power(bound=1.0))

term("rmsd_epitope_group",
    col("rmsd_epitope_group").lower().id()
    .bounded_power(bound=0.0))

term("tm_score_epitope_group",
    col("tm_score_epitope_group").higher().id()
    .bounded_power(bound=1.0))

term("aar_antigen_fold",
    col("aar_antigen_fold").higher().id()
    .bounded_power(bound=1.0))

term("rmsd_antigen_fold",
    col("rmsd_antigen_fold").lower().id()
    .bounded_power(bound=0.0))

term("tm_score_antigen_fold",
    col("tm_score_antigen_fold").higher().id()
    .bounded_power(bound=1.0))

term("aar_temporal",
    col("aar_temporal").higher().id()
    .bounded_power(bound=1.0))

term("rmsd_temporal",
    col("rmsd_temporal").lower().id()
    .bounded_power(bound=0.0))

term("tm_score_temporal",
    col("tm_score_temporal").higher().id()
    .bounded_power(bound=1.0))

setting("epitope_group", weighted_mean(("aar_epitope_group", 1.0), ("rmsd_epitope_group", 1.0), ("tm_score_epitope_group", 1.0)))
setting("antigen_fold", weighted_mean(("aar_antigen_fold", 1.0), ("rmsd_antigen_fold", 1.0), ("tm_score_antigen_fold", 1.0)))
setting("temporal", weighted_mean(("aar_temporal", 1.0), ("rmsd_temporal", 1.0), ("tm_score_temporal", 1.0)))

task(gmean("epitope_group", "antigen_fold", "temporal"))
