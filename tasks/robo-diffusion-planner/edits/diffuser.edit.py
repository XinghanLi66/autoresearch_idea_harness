"""Diffuser baseline — trajectory diffusion with classifier guidance.

Reference: Planning with Diffusion for Flexible Behavior Synthesis (Janner et al., 2022)
Paper: https://arxiv.org/abs/2205.09991

Key features:
  - Trajectory diffusion (obs + act concatenated)
  - JannerUNet1d architecture
  - CumRewClassifier for value-based guidance
  - Discrete diffusion SDE
  - Select best trajectory by log probability

The template is already based on diffuser_d4rl_mujoco.py, so only config path needs to be changed.
"""

_FILE = "CleanDiffuser/pipelines/custom_planner.py"

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 21,
        "end_line": 21,
        "content": '@hydra.main(config_path="../configs/diffuser/mujoco", config_name="mujoco", version_base=None)\n',
    },
]
