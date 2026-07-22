"""Score spec for pde-steady-solver (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("rel_err_Airfoil",
    col("rel_err_Airfoil").lower().id()
    .bounded_power(bound=0.0))

term("rel_err_Darcy",
    col("rel_err_Darcy").lower().id()
    .bounded_power(bound=0.0))

term("rel_err_Pipe",
    col("rel_err_Pipe").lower().id()
    .bounded_power(bound=0.0))

setting("Airfoil", weighted_mean(("rel_err_Airfoil", 1.0)))
setting("Darcy", weighted_mean(("rel_err_Darcy", 1.0)))
setting("Pipe", weighted_mean(("rel_err_Pipe", 1.0)))

task(gmean("Airfoil", "Darcy", "Pipe"))
