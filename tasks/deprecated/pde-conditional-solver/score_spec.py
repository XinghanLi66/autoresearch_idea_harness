"""Score spec for pde-conditional-solver (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("rel_err_Plasticity",
    col("rel_err_Plasticity").lower().id()
    .bounded_power(bound=0.0))

term("rel_err_SWE",
    col("rel_err_SWE").lower().id()
    .bounded_power(bound=0.0))

term("rel_err_CFD",
    col("rel_err_CFD").lower().id()
    .bounded_power(bound=0.0))

setting("Plasticity", weighted_mean(("rel_err_Plasticity", 1.0)))
setting("SWE", weighted_mean(("rel_err_SWE", 1.0)))
setting("CFD", weighted_mean(("rel_err_CFD", 1.0)))

task(gmean("Plasticity", "SWE", "CFD"))
