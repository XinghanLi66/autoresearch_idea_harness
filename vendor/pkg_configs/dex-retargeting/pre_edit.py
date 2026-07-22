"""Pre-edit operations for dex-retargeting package.

Register the 'custom' retargeting type so the evaluation harness can load
CustomOptimizer via the standard RetargetingConfig pipeline.

Changes:
1. Add 'custom' to RetargetingType enum in constants.py
2. Add 'custom' to _TYPE list and build() branch in retargeting_config.py
"""

# ── Op 1: Add 'custom' to RetargetingType enum (constants.py line 39) ────────

_CUSTOM_ENUM = """\
    dexpilot = enum.auto()  # For teleoperation, with finger closing prior
    custom = enum.auto()  # Custom optimizer for MLS-Bench evaluation
"""

# ── Op 2: Add 'custom' to _TYPE list (retargeting_config.py line 65) ─────────

_CUSTOM_TYPE_LIST = """\
    _TYPE = ["vector", "position", "dexpilot", "custom"]
"""

# ── Op 3: Add 'custom' type validation branch (retargeting_config.py line 71) ─

_CUSTOM_VALIDATION = """\
        if self.type not in self._TYPE:
            raise ValueError(f"Retargeting type must be one of {self._TYPE}")

        if self.type == "custom":
            if self.target_link_names is None:
                raise ValueError("Custom retargeting requires: target_link_names")
            self.target_link_human_indices = self.target_link_human_indices.squeeze()
"""

# ── Op 4: Add 'custom' build branch (retargeting_config.py after dexpilot) ───

_CUSTOM_BUILD = """\
        elif self.type == "dexpilot":
            optimizer = DexPilotOptimizer(
                robot,
                joint_names,
                finger_tip_link_names=self.finger_tip_link_names,
                wrist_link_name=self.wrist_link_name,
                target_link_human_indices=self.target_link_human_indices,
                scaling=self.scaling_factor,
                project_dist=self.project_dist,
                escape_dist=self.escape_dist,
            )
        elif self.type == "custom":
            from dex_retargeting.custom_optimizer import CustomOptimizer
            optimizer = CustomOptimizer(
                robot,
                joint_names,
                target_link_names=self.target_link_names,
                target_link_human_indices=self.target_link_human_indices,
                norm_delta=self.normal_delta,
                huber_delta=self.huber_delta,
            )
"""

# ── Operations (bottom-to-top within each file) ─────────────────────────────

OPS = [
    # constants.py: add custom enum member
    {
        "op": "replace",
        "file": "dex-retargeting/src/dex_retargeting/constants.py",
        "start_line": 39,
        "end_line": 39,
        "content": _CUSTOM_ENUM,
    },
    # retargeting_config.py: ops ordered bottom-to-top so line shifts
    # from earlier ops don't affect later ones within the same file.
    # Op A: add custom build branch (line 218-228, dexpilot elif + body)
    {
        "op": "replace",
        "file": "dex-retargeting/src/dex_retargeting/retargeting_config.py",
        "start_line": 218,
        "end_line": 228,
        "content": _CUSTOM_BUILD,
    },
    # Op B: add custom validation in __post_init__ (line 71-72)
    {
        "op": "replace",
        "file": "dex-retargeting/src/dex_retargeting/retargeting_config.py",
        "start_line": 71,
        "end_line": 72,
        "content": _CUSTOM_VALIDATION,
    },
    # Op C: expand _TYPE list (line 65)
    {
        "op": "replace",
        "file": "dex-retargeting/src/dex_retargeting/retargeting_config.py",
        "start_line": 65,
        "end_line": 65,
        "content": _CUSTOM_TYPE_LIST,
    },
]
