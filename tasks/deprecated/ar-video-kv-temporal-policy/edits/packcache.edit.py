"""packcache baseline: PackCache-inspired anchor preservation + temporal decay.

Maps PackCache's core mechanisms:
  - temporal anchor preservation (keyframe_score-based), checked before drift eviction
  - cross-frame temporal decay (age_decay influences priority scoring)
  - boundary demotion (demote_all at oracle boundary events via boundary_transition_rule)

Fidelity: partial/faithful mapping.  Full spatial position re-embedding is
outside the replay harness scope; the retention/decay mechanism is preserved.

Canonical reference: https://arxiv.org/abs/2601.04359
  PackCache: A Training-Free Acceleration Method for Unified Autoregressive Video
  Generation via Compact KV-Cache (Li et al., 2026)
"""

_FILE = "FAR/custom_video_eval.py"

_POLICY = """\
class VideoTemporalKVPolicy:
    \"\"\"PackCache-inspired: keyframe anchor preservation + temporal decay in priority scoring.
    Anchor check (keyframe_score / access_count) precedes drift eviction.
    age_decay influences priority ranking only, not the keep/drop/demote decision.
    boundary_transition_rule=demote_all handles cross-scene chunk demotion at oracle boundaries.
    \"\"\"

    def build_temporal_cache_plan(self, chunk_meta_list, rollout_state, budget_state):
        decisions = []
        for chunk in chunk_meta_list:
            r = chunk.recentness
            kf = chunk.keyframe_score
            age_decay = max(0.0, 1.0 - 0.12 * chunk.dynamic_age)
            hit_ratio = chunk.recent_hit_count / max(chunk.access_count, 1)
            # Anchor check precedes drift eviction: keyframe/high-access chunks are preserved
            if kf >= 0.72 or chunk.access_count >= 3:
                action = "demote_to_long_term"
            elif chunk.feature_drift >= 0.45:
                action = "drop"
            elif r >= 0.25 and chunk.temporal_similarity >= 0.85:
                action = "keep"
            else:
                action = "drop"
            priority = (
                2.1 * chunk.temporal_similarity
                + 1.7 * kf
                + 1.1 * age_decay
                + 0.9 * hit_ratio
                + 0.4 * (1.0 - chunk.motion_strength)
            )
            decisions.append(ChunkDecision(chunk_id=chunk.chunk_id, action=action, priority=priority))
        return TemporalCachePlan(
            chunk_decisions=decisions,
            retention_family="anchor",
            anchor_preservation_rule="keyframe_anchor",
            boundary_transition_rule="demote_all",
            queue_depth=0,
            long_term_budget_fraction=0.25,
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
