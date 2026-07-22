"""State identity baseline — raw state as goal (no learned representation).

The GoalRepresentation is an identity mapping: encode_goal returns raw goals,
compute_rep_loss returns 0. rep_dim is set to obs_dim at runtime via auto-detect.
"""

_FILE = "dual-goal-representations/custom_train.py"

# Change the default rep_dim from 256 to 0 (triggers auto-detect to obs_dim).
_NEW_REP_DIM = """\
    parser.add_argument('--rep_dim', type=int, default=0,
                        help='Goal representation dimension.')
"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 341,
        "end_line": 342,
        "content": _NEW_REP_DIM,
    },
]
