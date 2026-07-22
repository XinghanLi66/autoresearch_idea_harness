"""Pre-edit operations for titans-lmm package.

Applied before any task starts:
1. Inject TRAIN_METRICS / VAL_METRICS / TEST_METRICS prints into train_utils.py
2. Fix inter_dim bug in titans.py (uses hidden_dim instead of seq_len)
3. Fix weather dataset return shape in dataset.py (2D → 3D)

Operations are ordered bottom-to-top within each file to avoid line-number shifts.
Line numbers reference the original file at commit ce6a1ed.
"""

# ── train_utils.py: Metric injection ─────────────────────────────────

# Op 1: Inject TEST_METRICS after line 68
_TEST_METRICS = (
    '    print(f"TEST_METRICS test_mse={test_loss/test_num:.6f}", flush=True)\n'
)

# Op 2: Inject VAL_METRICS after line 53 (epoch level)
_VAL_METRICS = (
    '        print(f"VAL_METRICS epoch={epoch} val_mse={total_loss/sample_num:.6f}", flush=True)\n'
)

# Op 3: Inject TRAIN_METRICS after line 35 (epoch level, after batch loop)
_TRAIN_METRICS = (
    '        print(f"TRAIN_METRICS epoch={epoch} train_loss={train_loss:.6f}", flush=True)\n'
)

# ── titans.py: Fix inter_dim bug ─────────────────────────────────────
# Line 19: inter_dim should use seq_len, not hidden_dim.
# Original coincidentally works when context_window == hidden_dim == 16,
# but breaks for context_window=32.

# ── dataset.py: Fix weather non-segmented return shape ───────────────
# Line 42: self.data[index, :, -1] returns (500,) — 2D when batched.
# MSELoss cannot broadcast 2D labels against 3D model output.
# Fix: use -1: slice to preserve the last dimension → (500, 1).

# ── Operations (bottom-to-top within each file) ──────────────────────

OPS = [
    # --- train_utils.py ---
    {
        "op": "insert",
        "file": "titans-lmm/train_utils.py",
        "after_line": 68,
        "content": _TEST_METRICS,
    },
    {
        "op": "insert",
        "file": "titans-lmm/train_utils.py",
        "after_line": 53,
        "content": _VAL_METRICS,
    },
    {
        "op": "insert",
        "file": "titans-lmm/train_utils.py",
        "after_line": 35,
        "content": _TRAIN_METRICS,
    },
    # --- titans.py: fix inter_dim (line 19) ---
    {
        "op": "replace",
        "file": "titans-lmm/titans.py",
        "start_line": 19,
        "end_line": 19,
        "content": "        self.inter_dim = (pm_len + 2 * seq_len)\n",
    },
    # --- dataset.py: fix weather non-segmented return (line 42) ---
    {
        "op": "replace",
        "file": "titans-lmm/dataset.py",
        "start_line": 42,
        "end_line": 42,
        "content": "        return self.data[index, :, :-1], self.data[index, :, -1:]\n",
    },
]
