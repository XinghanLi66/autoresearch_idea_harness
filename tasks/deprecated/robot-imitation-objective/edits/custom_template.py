"""Imitation learning with customizable generative training objective.

This script trains a policy for robomimic manipulation tasks using a
ChiUNet1d backbone network. The training objective (how to train the
network and how to sample actions at inference) is defined by the
ImitationObjective class, which is the EDITABLE section.

Fixed infrastructure:
  - Dataset loading (robomimic low-dim demonstrations)
  - ChiUNet1d backbone network (same architecture for all objectives)
  - Evaluation loop (rollout in environment, compute success rate)
  - Training loop structure, logging, checkpointing

Editable section:
  - ImitationObjective class: defines how to compute the training loss
    and how to sample actions at inference time.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")

# ---- CleanDiffuser imports (dataset, env, network) ----
from cleandiffuser.dataset.robomimic_dataset import RobomimicDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.nn_diffusion import ChiUNet1d
from cleandiffuser.nn_condition import IdentityCondition

import gym
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


# ============================================================================
# EDITABLE SECTION: ImitationObjective
# ============================================================================
# Implement your generative training objective here. The SAME ChiUNet1d
# backbone is used for all objectives -- only the training formulation
# (loss computation) and inference procedure (action sampling) change.
#
# The backbone network signature:
#   network(x, t, condition) -> output
#     x:         (B, horizon, act_dim)   -- noisy/intermediate actions
#     t:         (B,)                    -- timestep (integer or float)
#     condition:  (B, obs_steps * obs_dim) -- flattened observation
#     output:    (B, horizon, act_dim)   -- predicted noise/velocity/clean actions
#
# You have access to:
#   - self.network:     the ChiUNet1d backbone (nn.Module, on device)
#   - self.network_ema: EMA copy of the network (for inference)
#   - self.act_dim:     action dimension
#   - self.horizon:     action prediction horizon
#   - self.device:      torch device
#   - self.config:      dict with keys from CONFIG below
#
# CONFIG dict is passed at construction; you may add custom keys.
#
# If your objective introduces additional trainable parameters (e.g. an
# encoder for CVAE), return them from extra_params() so they are included
# in the optimizer. They will be optimized jointly with the backbone.

CONFIG = {}


class ImitationObjective:
    """Generative training objective for imitation learning.

    This class wraps the training loss and sampling procedure.
    The backbone network is shared -- only the objective differs.
    """

    def __init__(self, network: nn.Module, network_ema: nn.Module,
                 act_dim: int, horizon: int, device: str, config: dict):
        self.network = network
        self.network_ema = network_ema
        self.act_dim = act_dim
        self.horizon = horizon
        self.device = device
        self.config = config

    def extra_params(self):
        """Return any additional trainable parameters (list of nn.Parameter).

        These will be added to the main optimizer alongside the backbone.
        Default: empty list (no extra parameters).
        """
        return []

    def compute_loss(self, network: nn.Module, actions: torch.Tensor,
                     condition: torch.Tensor) -> torch.Tensor:
        """Compute training loss.

        Args:
            network:   the backbone nn (same as self.network, but passed
                       explicitly so you can call network(...) directly)
            actions:   (B, horizon, act_dim) -- clean demonstration actions
            condition: (B, obs_steps * obs_dim) -- flattened observations

        Returns:
            loss: scalar tensor (mean loss for the batch)
        """
        raise NotImplementedError("Implement your training objective here")

    def sample(self, network_ema: nn.Module, condition: torch.Tensor,
               prior_shape: tuple) -> torch.Tensor:
        """Generate actions at inference time.

        Args:
            network_ema: EMA backbone network (use for inference)
            condition:   (B, obs_steps * obs_dim)
            prior_shape: (B, horizon, act_dim) -- shape of output

        Returns:
            actions: (B, horizon, act_dim) -- generated action sequence
        """
        raise NotImplementedError("Implement your sampling procedure here")


# ============================================================================
# END EDITABLE SECTION
# ============================================================================


# ---- Environment helpers (FIXED) ----

ENV_CONFIGS = {
    "lift": {
        "dataset_path": "/data/robomimic/datasets/lift/ph/low_dim_abs.hdf5",
        "obs_dim": 19, "action_dim": 10, "max_episode_steps": 400,
    },
    "can": {
        "dataset_path": "/data/robomimic/datasets/can/ph/low_dim_abs.hdf5",
        "obs_dim": 23, "action_dim": 10, "max_episode_steps": 400,
    },
    "square": {
        "dataset_path": "/data/robomimic/datasets/square/ph/low_dim_abs.hdf5",
        "obs_dim": 23, "action_dim": 10, "max_episode_steps": 500,
    },
}

OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OBS_STEPS = 2
ACTION_STEPS = 8
HORIZON = 16
BATCH_SIZE = 256
GRADIENT_STEPS = 300000
EVAL_FREQ = 100000
LOG_FREQ = 1000
LR = 1e-4
EMA_RATE = 0.995


def make_env(dataset_path, obs_keys, seed, idx, max_episode_steps):
    """Create a single robomimic evaluation environment."""
    def thunk():
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False,
            use_image_obs=False)
        env = RobomimicLowdimWrapper(
            env=env, obs_keys=obs_keys, init_state=None,
            render_hw=(256, 256), render_camera_name="agentview")
        env = MultiStepWrapper(
            env, n_obs_steps=OBS_STEPS, n_action_steps=ACTION_STEPS,
            max_episode_steps=max_episode_steps)
        env.seed(seed + idx)
        return env
    return thunk


def evaluate(envs, dataset, objective, num_envs, num_episodes,
             obs_steps, action_steps, horizon, act_dim, device,
             max_episode_steps):
    """Run rollouts and return success rate."""
    episode_rewards = []
    episode_success = []

    n_rounds = max(1, num_episodes // num_envs)
    for i in range(n_rounds):
        ep_reward = np.zeros(num_envs)
        obs = envs.reset()
        t = 0

        while t < max_episode_steps:
            obs_seq = obs.astype(np.float32)
            nobs = dataset.normalizer["obs"]["state"].normalize(obs_seq)
            nobs = torch.tensor(nobs, device=device, dtype=torch.float32)
            condition = nobs.flatten(start_dim=1)

            with torch.no_grad():
                prior_shape = (num_envs, horizon, act_dim)
                naction = objective.sample(
                    objective.network_ema, condition, prior_shape)

            naction = naction.detach().cpu().numpy()
            action_pred = dataset.normalizer["action"].unnormalize(naction)

            start = obs_steps - 1
            end = start + action_steps
            action = action_pred[:, start:end, :]
            action = dataset.undo_transform_action(action)

            obs, reward, done, info = envs.step(action)
            ep_reward += reward
            t += action_steps

        success = [1.0 if r > 0 else 0.0 for r in ep_reward]
        episode_rewards.extend(ep_reward.tolist())
        episode_success.extend(success)

    return float(np.mean(episode_success))


def ema_update(model, model_ema, rate):
    """Exponential moving average update."""
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), model_ema.parameters()):
            p_ema.data.mul_(rate).add_(p.data, alpha=1.0 - rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        choices=["lift", "can", "square"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gradient_steps", type=int, default=GRADIENT_STEPS)
    parser.add_argument("--eval_freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_cfg = ENV_CONFIGS[args.env]
    obs_dim = env_cfg["obs_dim"]
    act_dim = env_cfg["action_dim"]
    max_ep_steps = env_cfg["max_episode_steps"]
    dataset_path = env_cfg["dataset_path"]

    device = args.device if torch.cuda.is_available() else "cpu"

    # ---- Dataset ----
    dataset = RobomimicDataset(
        dataset_path, horizon=HORIZON, obs_keys=OBS_KEYS,
        pad_before=OBS_STEPS - 1, pad_after=ACTION_STEPS - 1,
        abs_action=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)

    # ---- Backbone Network (FIXED) ----
    nn_diffusion = ChiUNet1d(
        act_dim, obs_dim, OBS_STEPS,
        model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
        obs_as_global_cond=True, timestep_emb_type="positional").to(device)
    nn_diffusion_ema = deepcopy(nn_diffusion).requires_grad_(False)
    nn_diffusion_ema.eval()

    # ---- Objective ----
    objective = ImitationObjective(
        network=nn_diffusion,
        network_ema=nn_diffusion_ema,
        act_dim=act_dim,
        horizon=HORIZON,
        device=device,
        config=CONFIG)

    # ---- Optimizer (includes any extra params from the objective) ----
    extra = list(objective.extra_params()) if hasattr(objective, 'extra_params') else []
    all_params = list(nn_diffusion.parameters()) + extra
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=1e-6)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.gradient_steps)

    # ---- Environment for evaluation ----
    envs = gym.vector.SyncVectorEnv(
        [make_env(dataset_path, OBS_KEYS, args.seed, idx, max_ep_steps)
         for idx in range(args.num_envs)])

    # ---- Training Loop ----
    nn_diffusion.train()
    n_step = 0
    loss_accum = 0.0

    best_success = -1.0
    save_dir = args.save_dir or os.environ.get(
        "SAVE_PATH", f"/tmp/robot_imitation_{args.env}")
    os.makedirs(save_dir, exist_ok=True)

    for batch in loop_dataloader(dataloader):
        nobs = batch["obs"]["state"].to(device)
        naction = batch["action"].to(device)

        condition = nobs[:, :OBS_STEPS, :].flatten(start_dim=1)

        # Compute loss via objective
        loss = objective.compute_loss(nn_diffusion, naction, condition)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(nn_diffusion.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        # EMA update
        ema_update(nn_diffusion, nn_diffusion_ema, EMA_RATE)

        loss_accum += loss.item()
        n_step += 1

        if n_step % LOG_FREQ == 0:
            avg_loss = loss_accum / LOG_FREQ
            print(f"TRAIN_METRICS step={n_step} loss={avg_loss:.6f}", flush=True)
            loss_accum = 0.0

        if n_step % args.eval_freq == 0:
            print(f"Evaluating at step {n_step}...")
            nn_diffusion.eval()
            nn_diffusion_ema.eval()

            success_rate = evaluate(
                envs, dataset, objective, args.num_envs, args.eval_episodes,
                OBS_STEPS, ACTION_STEPS, HORIZON, act_dim, device, max_ep_steps)

            print(f"TEST_METRICS step={n_step} success_rate={success_rate:.4f}",
                  flush=True)

            if success_rate >= best_success:
                best_success = success_rate
                extra_state = {}
                if hasattr(objective, 'extra_params'):
                    extra_state = {f"extra_{i}": p.data.clone()
                                   for i, p in enumerate(objective.extra_params())}
                torch.save({
                    "network": nn_diffusion.state_dict(),
                    "network_ema": nn_diffusion_ema.state_dict(),
                    **extra_state,
                }, os.path.join(save_dir, "best.pt"))

            nn_diffusion.train()

        if n_step >= args.gradient_steps:
            break

    # ---- Final Evaluation ----
    print("Final evaluation...")
    nn_diffusion.eval()
    nn_diffusion_ema.eval()

    # Load best checkpoint if exists
    best_path = os.path.join(save_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        nn_diffusion.load_state_dict(ckpt["network"])
        nn_diffusion_ema.load_state_dict(ckpt["network_ema"])
        if hasattr(objective, 'extra_params'):
            for i, p in enumerate(objective.extra_params()):
                key = f"extra_{i}"
                if key in ckpt:
                    p.data.copy_(ckpt[key])

    final_success = evaluate(
        envs, dataset, objective, args.num_envs, args.eval_episodes,
        OBS_STEPS, ACTION_STEPS, HORIZON, act_dim, device, max_ep_steps)

    print(f"TEST_METRICS step={n_step} success_rate={final_success:.4f}", flush=True)

    envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
