"""Mid-edit operations for the robo-diffusion-planner task.

Applied to the CleanDiffuser workspace after pre_edit, before the agent starts.
Creates:
  - CleanDiffuser/pipelines/custom_planner.py (the agent's editable algorithm file)
  - CleanDiffuser/configs/custom/mujoco/mujoco.yaml (custom config)
  - CleanDiffuser/configs/custom/mujoco/task/hopper-medium-v2.yaml
  - CleanDiffuser/configs/custom/mujoco/task/walker2d-medium-v2.yaml
  - CleanDiffuser/configs/custom/mujoco/task/halfcheetah-medium-v2.yaml
"""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).parent / "custom_template.py"
_CUSTOM_PY = _TEMPLATE_PATH.read_text()

_BASE_CONFIG = """defaults:
  - _self_
  - task: hopper-medium-v2

pipeline_name: custom_planner
mode: train
seed: 42
device: cuda:0

# Environment
terminal_penalty: -100
discount: 0.997

# Diffuser
solver: ddpm
model_dim: 32
diffusion_steps: 20
sampling_steps: 20
predict_noise: False
action_loss_weight: 10.
ema_rate: 0.9999

# Training (aligned with CleanDiffuser configs/diffuser/mujoco/mujoco.yaml,
# which is what the package ships and what Diffuser (Janner et al., 2022)
# uses on D4RL-MuJoCo medium-v2)
diffusion_gradient_steps: 1000000
classifier_gradient_steps: 1000000
batch_size: 64
log_interval: 1000
save_interval: 100000

# Inference (aligned with package: 50 parallel envs x 3 episodes = 150 rollouts)
ckpt: latest
num_envs: 50
num_episodes: 3
num_candidates: 64
temperature: 0.5
use_ema: True

# AdaptDiffuser
ft_ckpt: latest

# hydra
hydra:
  job:
    chdir: false
"""

_HOPPER_CONFIG = """env_name: "hopper-medium-v2"
dim_mult: [1, 2, 2, 2]
w_cg: 0.3
horizon: 32
"""

_WALKER2D_CONFIG = """env_name: "walker2d-medium-v2"
dim_mult: [1, 2, 2, 2]
w_cg: 0.007
horizon: 32
"""

_HALFCHEETAH_CONFIG = """env_name: "halfcheetah-medium-v2"
dim_mult: [1, 4, 2]
w_cg: 0.0001
horizon: 4
"""

# ── Mid-edit operations ──────────────────────────────────────────────

OPS = [
    {
        "op": "create",
        "file": "CleanDiffuser/pipelines/custom_planner.py",
        "content": _CUSTOM_PY,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/mujoco.yaml",
        "content": _BASE_CONFIG,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/task/hopper-medium-v2.yaml",
        "content": _HOPPER_CONFIG,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/task/walker2d-medium-v2.yaml",
        "content": _WALKER2D_CONFIG,
    },
    {
        "op": "create",
        "file": "CleanDiffuser/configs/custom/mujoco/task/halfcheetah-medium-v2.yaml",
        "content": _HALFCHEETAH_CONFIG,
    },
]
