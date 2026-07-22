"""Score spec for ai4sci-sbdd-drug-design."""
from mlsbench.scoring.dsl import *

# qed: drug-likeness, higher is better, bounded at 1
term("qed_denovo",
    col("qed_denovo").higher().id()
    .bounded_power(bound=1.0))

# sa: normalized synthesizability score (higher = more synthesizable = better)
term("sa_denovo",
    col("sa_denovo").higher().id()
    .bounded_power(bound=1.0))

term("validity_denovo",
    col("validity_denovo").higher().id()
    .bounded_power(bound=1.0))

term("qed_linker",
    col("qed_linker").higher().id()
    .bounded_power(bound=1.0))

term("sa_linker",
    col("sa_linker").higher().id()
    .bounded_power(bound=1.0))

term("validity_linker",
    col("validity_linker").higher().id()
    .bounded_power(bound=1.0))

term("qed_frag",
    col("qed_frag").higher().id()
    .bounded_power(bound=1.0))

term("sa_frag",
    col("sa_frag").higher().id()
    .bounded_power(bound=1.0))

term("validity_frag",
    col("validity_frag").higher().id()
    .bounded_power(bound=1.0))

setting("denovo", weighted_mean(("qed_denovo", 1.0), ("sa_denovo", 1.0), ("validity_denovo", 1.0)))
setting("linker", weighted_mean(("qed_linker", 1.0), ("sa_linker", 1.0), ("validity_linker", 1.0)))
setting("frag", weighted_mean(("qed_frag", 1.0), ("sa_frag", 1.0), ("validity_frag", 1.0)))

task(gmean("denovo", "linker", "frag"))
