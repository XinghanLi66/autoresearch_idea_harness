"""Score spec for pde-foundation-icl."""
from mlsbench.scoring.dsl import *

term("last_step_rel_err_NS2D",
    col("last_step_rel_err_NS2D").lower().id()
    .bounded_power(bound=0.0))

term("all_avg_rel_err_NS2D",
    col("all_avg_rel_err_NS2D").lower().id()
    .bounded_power(bound=0.0))

term("last_step_rel_err_COMPRESSIBLE2D",
    col("last_step_rel_err_COMPRESSIBLE2D").lower().id()
    .bounded_power(bound=0.0))

term("all_avg_rel_err_COMPRESSIBLE2D",
    col("all_avg_rel_err_COMPRESSIBLE2D").lower().id()
    .bounded_power(bound=0.0))

term("last_step_rel_err_EULER2D",
    col("last_step_rel_err_EULER2D").lower().id()
    .bounded_power(bound=0.0))

term("all_avg_rel_err_EULER2D",
    col("all_avg_rel_err_EULER2D").lower().id()
    .bounded_power(bound=0.0))

setting("NS2D", weighted_mean(("last_step_rel_err_NS2D", 1.0), ("all_avg_rel_err_NS2D", 1.0)))
setting("COMPRESSIBLE2D", weighted_mean(("last_step_rel_err_COMPRESSIBLE2D", 1.0), ("all_avg_rel_err_COMPRESSIBLE2D", 1.0)))
setting("EULER2D", weighted_mean(("last_step_rel_err_EULER2D", 1.0), ("all_avg_rel_err_EULER2D", 1.0)))

task(gmean("NS2D", "COMPRESSIBLE2D", "EULER2D"))
