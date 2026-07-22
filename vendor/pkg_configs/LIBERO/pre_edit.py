"""Pre-edit operations for LIBERO package.

Applied to the LIBERO workspace before the agent starts:
1. Register the Custom algorithm class in the algo registry
2. Create Hydra config for the custom lifelong algorithm
3. Inject TRAIN_METRICS output in base.py for per-epoch training feedback
4. Inject EVAL_METRICS output in main.py for per-task evaluation feedback
5. Inject TEST_METRICS output in main.py for final leaderboard metric

Operations on the same file are ordered bottom-to-top to avoid line-number
shifts. Line numbers reference commit 8f1084e3.
"""

# ── Op 1: Register Custom algo in __init__.py (after line 7) ────────
_CUSTOM_IMPORT = "from libero.lifelong.algos.custom import Custom\n"

# ── Op 2: Create Hydra config for custom algo ───────────────────────
_CUSTOM_YAML = "algo: Custom\n"

# ── Op 3: Inject TRAIN_METRICS in base.py (after line 183) ──────────
# Line 183 in base.py is the closing paren of the epoch loss print.
# We insert a TRAIN_METRICS line right after it, at the same indentation.
_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS task={self.current_task} '
    'epoch={epoch} loss={training_loss:.6f}", flush=True)\n'
)

# ── Op 4: Inject EVAL_METRICS in main.py (after line 258) ───────────
# Line 258 prints per-task success rates. We add a parseable summary.
_EVAL_METRICS = (
    '                print(f"EVAL_METRICS after_task={i} '
    'avg_success={np.mean(S):.4f}", flush=True)\n'
)

# ── Op 5: Inject TEST_METRICS in main.py (after line 263) ───────────
# Line 263 prints "[info] finished learning". We add the final metric.
_TEST_METRICS = """\
    if cfg.eval.eval:
        _S_mat = result_summary["S_conf_mat"]
        _avg_final = float(np.mean(_S_mat[n_tasks - 1][:n_manip_tasks]))
        print(f"TEST_METRICS avg_final_success={_avg_final:.4f}", flush=True)
        for _ti in range(n_manip_tasks):
            print(f"TASK_METRICS task={_ti} success_rate={float(_S_mat[n_tasks - 1][_ti]):.4f}", flush=True)
"""

# ── Op 6: Point dataset folder to pre-downloaded data ────────────────
# Line 57 in main.py: cfg.folder = cfg.folder or get_libero_path("datasets")
# Replace with our pre-downloaded data path.
_DATA_PATH_FIX = '    cfg.folder = cfg.folder or "/data/libero"\n'

# ── Operations (bottom-to-top within each file) ─────────────────────

OPS = [
    # main.py — bottom-to-top
    {
        "op": "insert",
        "file": "LIBERO/libero/lifelong/main.py",
        "line": 263,
        "content": _TEST_METRICS,
    },
    {
        "op": "insert",
        "file": "LIBERO/libero/lifelong/main.py",
        "line": 258,
        "content": _EVAL_METRICS,
    },
    {
        "op": "replace",
        "file": "LIBERO/libero/lifelong/main.py",
        "start_line": 57,
        "end_line": 57,
        "content": _DATA_PATH_FIX,
    },
    # base.py
    {
        "op": "insert",
        "file": "LIBERO/libero/lifelong/algos/base.py",
        "line": 183,
        "content": _TRAIN_METRICS,
    },
    # __init__.py
    {
        "op": "insert",
        "file": "LIBERO/libero/lifelong/algos/__init__.py",
        "line": 7,
        "content": _CUSTOM_IMPORT,
    },
    # Create custom algo Hydra config
    {
        "op": "create",
        "file": "LIBERO/libero/configs/lifelong/custom.yaml",
        "content": _CUSTOM_YAML,
    },
]
