"""causalcache_vdm baseline: CausalCache-VDM inspired cache sharing + queue/dequeue.

Maps CausalCache-VDM's core mechanisms:
  - shared KV cache pool across generation context
  - queue/dequeue management (queue_depth limits retained chunks)
  - long-term resident promotion for frequently reused / high-importance chunks
  - selective boundary transition (respect chunk decisions, no blanket flush)

Fidelity: concept-inspired / mechanism-family mapping.
  Implements the shared-pool + queue-eviction family found in causal video
  diffusion KV-sharing literature.  Full denoising-timestep shared cache is
  outside the replay harness scope; retention/promotion logic is preserved.

Canonical reference: https://arxiv.org/abs/2411.16375
  Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation
  and Cache Sharing (Gao et al., 2024).
  Mechanism: causal temporal attention enables pre-computing and reusing KV caches
  of conditional frames; cache sharing across all denoising timesteps eliminates
  redundant storage; KV-cache stored in a queue (FIFO eviction at P_max).
"""

_FILE = "FAR/custom_video_eval.py"

_POLICY = """\
class VideoTemporalKVPolicy:
    \"\"\"CausalCache-VDM concept-inspired: LT pool + priority-based queue eviction.
    Maps the queue/pool family from Ca2-VDM (arXiv:2411.16375) into the replay harness.
    Note: eviction uses priority ranking (not strict FIFO); cross-denoising-step
    cache sharing is abstracted away as the harness operates at chunk granularity.
    \"\"\"

    def build_temporal_cache_plan(self, chunk_meta_list, rollout_state, budget_state):
        decisions = []
        for chunk in chunk_meta_list:
            r = chunk.recentness
            kf = chunk.keyframe_score
            is_long_term = chunk.long_term_resident
            overlap_score = chunk.recent_hit_count / max(chunk.access_count, 1)
            # Frequently accessed chunks are long-term candidates even before
            # the resident flag is set; anchor check precedes drift eviction.
            is_lt_candidate = is_long_term or kf >= 0.75 or chunk.access_count >= 3
            if is_lt_candidate:
                action = "demote_to_long_term"
            elif chunk.feature_drift >= 0.50:
                action = "drop"
            elif r >= 0.20 or overlap_score >= 0.70:
                action = "keep"
            else:
                action = "drop"
            priority = (
                1.6 * r
                + 1.5 * overlap_score
                + 1.2 * kf
                + 0.8 * chunk.temporal_similarity
                + 0.4 * (1.0 - chunk.motion_strength)
            )
            decisions.append(ChunkDecision(chunk_id=chunk.chunk_id, action=action, priority=priority))
        return TemporalCachePlan(
            chunk_decisions=decisions,
            retention_family="queue",
            anchor_preservation_rule="keyframe_anchor",
            boundary_transition_rule="selective",
            queue_depth=6,
            long_term_budget_fraction=0.35,
            chunkwise_reuse=False,
            compression_mode="none",
        )
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 685,
        "end_line": 716,
        "content": _POLICY,
    },
]
