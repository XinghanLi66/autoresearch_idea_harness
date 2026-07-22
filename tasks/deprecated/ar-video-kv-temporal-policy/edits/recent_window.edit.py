"""recent_window baseline: naive recency anchor for video temporal KV retention."""

_FILE = "FAR/custom_video_eval.py"

_POLICY = """\
class VideoTemporalKVPolicy:
    \"\"\"Naive recency baseline: keep the most recent chunks by recentness threshold.
    Budget capacity is enforced by the evaluator's _apply_plan; this policy expresses
    intent (keep recent / drop old) and the evaluator applies capacity constraints.
    \"\"\"

    def build_temporal_cache_plan(self, chunk_meta_list, rollout_state, budget_state):
        decisions = []
        for chunk in chunk_meta_list:
            r = chunk.recentness
            action = "keep" if r >= 0.34 else "drop"
            priority = 3.0 * r + 0.5 * chunk.temporal_similarity
            decisions.append(ChunkDecision(chunk_id=chunk.chunk_id, action=action, priority=priority))
        return TemporalCachePlan(
            chunk_decisions=decisions,
            retention_family="recency",
            anchor_preservation_rule="none",
            boundary_transition_rule="passthrough",
            queue_depth=0,
            long_term_budget_fraction=0.0,
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
