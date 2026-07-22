"""Score spec for ai4sci-mol3d-generation (auto-generated, review before use)."""
from mlsbench.scoring.dsl import *

term("atom_stable_QM9",
    col("atom_stable_QM9").higher().id()
    .bounded_power(bound=1.0))

term("mol_stable_QM9",
    col("mol_stable_QM9").higher().id()
    .bounded_power(bound=1.0))

term("valid_QM9",
    col("valid_QM9").higher().id()
    .bounded_power(bound=1.0))

term("unique_QM9",
    col("unique_QM9").higher().id()
    .bounded_power(bound=1.0))

term("comp_valid_MP_20",
    col("comp_valid_MP-20").higher().id()
    .bounded_power(bound=1.0))

term("struct_valid_MP_20",
    col("struct_valid_MP-20").higher().id()
    .bounded_power(bound=1.0))

term("valid_MP_20",
    col("valid_MP-20").higher().id()
    .bounded_power(bound=1.0))

term("atom_stable_GEOM_DRUG",
    col("atom_stable_GEOM-DRUG").higher().id()
    .bounded_power(bound=1.0))

term("mol_stable_GEOM_DRUG",
    col("mol_stable_GEOM-DRUG").higher().id()
    .bounded_power(bound=1.0))

term("valid_GEOM_DRUG",
    col("valid_GEOM-DRUG").higher().id()
    .bounded_power(bound=1.0))

term("unique_GEOM_DRUG",
    col("unique_GEOM-DRUG").higher().id()
    .bounded_power(bound=1.0))

setting("QM9", weighted_mean(("atom_stable_QM9", 1.0), ("mol_stable_QM9", 1.0), ("valid_QM9", 1.0), ("unique_QM9", 1.0)))
setting("MP-20", weighted_mean(("comp_valid_MP_20", 1.0), ("struct_valid_MP_20", 1.0), ("valid_MP_20", 1.0)))
setting("GEOM-DRUG", weighted_mean(("atom_stable_GEOM_DRUG", 1.0), ("mol_stable_GEOM_DRUG", 1.0), ("valid_GEOM_DRUG", 1.0), ("unique_GEOM_DRUG", 1.0)))

task(gmean("QM9", "MP-20", "GEOM-DRUG"))
