"""Score spec for dex-retarget."""
from mlsbench.scoring.dsl import *

# mpjpe_mm: mean per-joint position error in mm, lower is better, bounded at 0
# smoothness: trajectory jerk measure, lower = smoother motion = better, bounded at 0
# fps: frames per second throughput, higher is better (unbounded, use sigmoid)

term("allegro_mpjpe_mm",
    col("allegro_mpjpe_mm").lower().id()
    .bounded_power(bound=0.0))

term("allegro_smoothness",
    col("allegro_smoothness").lower().id()
    .bounded_power(bound=0.0))

term("allegro_fps",
    col("allegro_fps").higher().id()
    .sigmoid())

term("svh_mpjpe_mm",
    col("svh_mpjpe_mm").lower().id()
    .bounded_power(bound=0.0))

term("svh_smoothness",
    col("svh_smoothness").lower().id()
    .bounded_power(bound=0.0))

term("svh_fps",
    col("svh_fps").higher().id()
    .sigmoid())

term("leap_mpjpe_mm",
    col("leap_mpjpe_mm").lower().id()
    .bounded_power(bound=0.0))

term("leap_smoothness",
    col("leap_smoothness").lower().id()
    .bounded_power(bound=0.0))

term("leap_fps",
    col("leap_fps").higher().id()
    .sigmoid())

setting("allegro", weighted_mean(("allegro_mpjpe_mm", 1.0), ("allegro_smoothness", 1.0), ("allegro_fps", 1.0)))
setting("svh", weighted_mean(("svh_mpjpe_mm", 1.0), ("svh_smoothness", 1.0), ("svh_fps", 1.0)))
setting("leap", weighted_mean(("leap_mpjpe_mm", 1.0), ("leap_smoothness", 1.0), ("leap_fps", 1.0)))

task(gmean("allegro", "svh", "leap"))
