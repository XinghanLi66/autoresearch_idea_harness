"""Pre-edit operations for the Derf package.

Patches ViT/main.py to:
1. Add '--normtype custom' support (imports from custom_norm.py)
2. Add TRAIN_METRICS output after each epoch
3. Add EVAL_METRIC output at end of training
"""

# ── 1. Add 'custom' to normtype choices (line 203-206) ──────────────────
# Original:
#     parser.add_argument('--normtype', type=str, default='derf',
#                         choices=['layernorm', 'derf', 'dyt', ..., 'logquad_clip'])
_NORMTYPE_CHOICES = """\
    parser.add_argument('--normtype', type=str, default='derf',
                        choices=['layernorm', 'derf', 'dyt', 'satursin', 'isru', 'expsign', 'smoothsign',
                                 'relsign', 'cubsign', 'exproot', 'saturlog', 'arctan', 'linear_clip',
                                 'power23_clip', 'logsign_clip', 'smoothsign_clip', 'logquad_clip', 'custom'])
"""

# ── 2. Add custom normtype handler after line 351 (logquad_clip block) ──
_CUSTOM_NORM_HANDLER = """\
    elif args.normtype == 'custom':
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from custom_norm import convert_ln_to_custom
        model = convert_ln_to_custom(model)
"""

# ── 3. TRAIN_METRICS after max accuracy print (after line 489) ──────────
_TRAIN_METRICS = """\
            print(f"TRAIN_METRICS step={epoch} train_loss={train_stats['loss']:.4f} accuracy={test_stats['acc1']:.4f}", flush=True)
"""

# ── 4. EVAL_METRIC after training time print (after line 535) ───────────
_EVAL_METRIC = """\
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    throughput = len(dataset_train) * args.epochs / total_time if total_time > 0 else 0
    print(f"EVAL_METRIC accuracy={max_accuracy:.4f} throughput={throughput:.2f} memory={peak_memory:.1f}", flush=True)
"""

# ── Operations ──────────────────────────────────────────────────────────

OPS = [
    # 1. Replace normtype choices to add 'custom'
    {
        "op": "replace",
        "file": "Derf/ViT/main.py",
        "start_line": 203,
        "end_line": 206,
        "content": _NORMTYPE_CHOICES,
    },
    # 2. Insert custom normtype handler after the last elif (logquad_clip)
    {
        "op": "insert",
        "file": "Derf/ViT/main.py",
        "after_line": 351,
        "content": _CUSTOM_NORM_HANDLER,
    },
    # 3. Insert TRAIN_METRICS after max accuracy print
    {
        "op": "insert",
        "file": "Derf/ViT/main.py",
        "after_line": 489,
        "content": _TRAIN_METRICS,
    },
    # 4. Insert EVAL_METRIC after training time print
    {
        "op": "insert",
        "file": "Derf/ViT/main.py",
        "after_line": 535,
        "content": _EVAL_METRIC,
    },
]
