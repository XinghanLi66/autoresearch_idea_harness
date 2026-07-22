"""Score spec for ar-video-kv-temporal-policy."""
from mlsbench.scoring.dsl import *


term("temporal_quality_main_ucf_short",
    col("temporal_quality_main_ucf-short").higher().id()
    .bounded_power(bound=100.0))

term("peak_kv_memory_mb_ucf_short",
    col("peak_kv_memory_mb_ucf-short").lower().id()
    .bounded_power(bound=0.0))

term("temporal_quality_main_dmlab_long",
    col("temporal_quality_main_dmlab-long").higher().id()
    .bounded_power(bound=100.0))

term("peak_kv_memory_mb_dmlab_long",
    col("peak_kv_memory_mb_dmlab-long").lower().id()
    .bounded_power(bound=0.0))

term("temporal_quality_main_ucf_scene_cut",
    col("temporal_quality_main_ucf-scene-cut").higher().id()
    .bounded_power(bound=100.0))

term("peak_kv_memory_mb_ucf_scene_cut",
    col("peak_kv_memory_mb_ucf-scene-cut").lower().id()
    .bounded_power(bound=0.0))

setting("ucf-short", weighted_mean(
    ("temporal_quality_main_ucf_short", 2.0),
    ("peak_kv_memory_mb_ucf_short", 0.5),
))
setting("dmlab-long", weighted_mean(
    ("temporal_quality_main_dmlab_long", 2.0),
    ("peak_kv_memory_mb_dmlab_long", 0.5),
))
setting("ucf-scene-cut", weighted_mean(
    ("temporal_quality_main_ucf_scene_cut", 2.0),
    ("peak_kv_memory_mb_ucf_scene_cut", 0.5),
))

task(gmean("ucf-short", "dmlab-long", "ucf-scene-cut"))
