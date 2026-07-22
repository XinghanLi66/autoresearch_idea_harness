"""Pre-edit operations for the katakomba package.

Injects TRAIN_METRICS and eval score print statements in baseline algorithm
files so that metrics appear on stdout.

Also patches render.py to remove cv2 dependency (GLIBC incompatibility with
Apptainer --nv injected libGLX.so.0 on hosts with newer GLIBC than container).
"""

# TRAIN_METRICS snippet for BC (has `loss` variable, no loss_info dict)
_TRAIN_METRICS_BC = (
    '        if step % 1000 == 0:\n'
    '            print(f"TRAIN_METRICS step={step} loss={loss.detach().item():.4f}", flush=True)'
)

# TRAIN_METRICS snippet for CQL/IQL (have loss_info dict)
_TRAIN_METRICS_LOSS_INFO = (
    '        if step % 1000 == 0:\n'
    '            _items = {k: v.item() if hasattr(v, "item") else v for k, v in loss_info.items()}\n'
    '            metrics_str = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in _items.items())\n'
    '            print(f"TRAIN_METRICS step={step} {metrics_str}", flush=True)'
)

# Eval normalized score print (all algorithms)
_EVAL_SCORE_PRINT = (
    '            print(f"Normalized score: {np.mean(normalized_scores):.4f}", flush=True)\n'
    '            print(f"D4RL score: {np.mean(normalized_scores):.6f}", flush=True)'
)

# ── Pre-edit operations ──────────────────────────────────────────────

OPS = [
    # render.py — replace cv2.resize with PIL Image.resize (line 75, bottom-to-top)
    {"op": "replace", "file": "katakomba/katakomba/utils/render.py",
     "start_line": 75, "end_line": 75,
     "content": '                char = np.array(Image.fromarray(char).resize(rescale_font_size, Image.Resampling.LANCZOS))\n'},
    # render.py — remove import cv2 (line 6)
    {"op": "delete", "file": "katakomba/katakomba/utils/render.py",
     "line": 6},
    # bc_chaotic_lstm.py — TRAIN_METRICS (line 367)
    {"op": "insert", "file": "katakomba/algorithms/small_scale/bc_chaotic_lstm.py",
     "after_line": 367, "content": _TRAIN_METRICS_BC},
    # bc_chaotic_lstm.py — eval score (line 386)
    {"op": "insert", "file": "katakomba/algorithms/small_scale/bc_chaotic_lstm.py",
     "after_line": 386, "content": _EVAL_SCORE_PRINT},
    # cql_chaotic_lstm.py — TRAIN_METRICS (line 442)
    {"op": "insert", "file": "katakomba/algorithms/small_scale/cql_chaotic_lstm.py",
     "after_line": 442, "content": _TRAIN_METRICS_LOSS_INFO},
    # cql_chaotic_lstm.py — eval score (line 457)
    {"op": "insert", "file": "katakomba/algorithms/small_scale/cql_chaotic_lstm.py",
     "after_line": 457, "content": _EVAL_SCORE_PRINT},
    # iql_chaotic_lstm.py — TRAIN_METRICS (line 482)
    {"op": "insert", "file": "katakomba/algorithms/small_scale/iql_chaotic_lstm.py",
     "after_line": 482, "content": _TRAIN_METRICS_LOSS_INFO},
    # iql_chaotic_lstm.py — eval score (line 497)
    {"op": "insert", "file": "katakomba/algorithms/small_scale/iql_chaotic_lstm.py",
     "after_line": 497, "content": _EVAL_SCORE_PRINT},
]
