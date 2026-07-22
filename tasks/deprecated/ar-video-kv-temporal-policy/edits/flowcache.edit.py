"""flowcache baseline: FlowCache-inspired chunkwise adaptive caching.

Maps FlowCache's core mechanisms:
  - chunkwise adaptive caching (per-chunk flow-score-based keep/drop decision)
  - flow_score = temporal_similarity * (1 - feature_drift) * (1 - 0.3*motion)
    as a proxy for the optical-flow-derived cache utility signal
  - boundary detection via boundary_score signal: pre-boundary chunks are
    explicitly dropped so anchor rule rescues keyframes to demote_to_long_term
  - fixed memory bound with priority-ranked eviction
  - motion-aware compression for demoted chunks (token_drop proxy)

Fidelity: SOTA paper-backed mapping.

Canonical reference: https://arxiv.org/abs/2602.10825
  Flow Caching for Autoregressive Video Generation (2026)
  chunk-specific KV compression with flow-based adaptive cache decisions.
"""

_FILE = "FAR/custom_video_eval.py"

_POLICY = """\
class VideoTemporalKVPolicy:
    \"\"\"FlowCache-inspired: chunkwise adaptive caching with boundary-score detection.\"\"\"

    def build_temporal_cache_plan(self, chunk_meta_list, rollout_state, budget_state):
        # Detect scene transition: find the most recent chunk whose boundary_score
        # indicates a boundary event (>= 0.45).  Chunks older than that boundary
        # are pre-boundary and should be evicted (anchor rule rescues keyframes).
        boundary_age = None
        for chunk in chunk_meta_list:
            if chunk.boundary_score >= 0.45:
                da = chunk.dynamic_age
                if boundary_age is None or da < boundary_age:
                    boundary_age = da

        decisions = []
        for chunk in chunk_meta_list:
            r = chunk.recentness
            kf = chunk.keyframe_score
            sim = chunk.temporal_similarity
            drift = chunk.feature_drift
            motion = chunk.motion_strength
            flow_score = sim * (1.0 - drift) * (1.0 - 0.3 * motion)

            is_pre_boundary = (
                boundary_age is not None and chunk.dynamic_age > boundary_age
            )

            if is_pre_boundary:
                # Drop pre-boundary chunks; keyframe_anchor rule rescues kf>=0.75
                # to demote_to_long_term so critical anchors are not fully evicted.
                action = "drop"
            elif kf >= 0.75:
                action = "keep"
            elif flow_score >= 0.70 or r >= 0.35:
                action = "keep"
            elif flow_score >= 0.40:
                action = "demote_to_long_term"
            else:
                action = "drop"

            priority = (
                2.0 * flow_score
                + 1.5 * kf
                + 1.2 * r
                + 0.8 * sim
            )
            decisions.append(ChunkDecision(chunk_id=chunk.chunk_id, action=action, priority=priority))

        return TemporalCachePlan(
            chunk_decisions=decisions,
            retention_family="chunkwise",
            anchor_preservation_rule="keyframe_anchor",
            boundary_transition_rule="selective",
            queue_depth=0,
            long_term_budget_fraction=0.30,
            chunkwise_reuse=True,
            compression_mode="token_drop",
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
