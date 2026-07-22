"""
Diffusion Policy with Custom Observation Conditioning -- Robomimic

Self-contained training + evaluation script for the robot-policy-conditioning task.
Trains a DDPM diffusion policy on robomimic low-dim demonstrations and evaluates
success rate via rollouts. The agent modifies the ConditioningModule class to
design how observations condition the denoising process.

Usage:
    python custom_conditioning.py --env lift --seed 42
"""

import os
import sys
import math
import time
import warnings
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import gym

warnings.filterwarnings("ignore")

# ── CleanDiffuser imports ──────────────────────────────────────────────────────
# These are available from the CleanDiffuser package installed in the container.
from cleandiffuser.dataset.robomimic_dataset import RobomimicDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from cleandiffuser.env.wrapper import VideoRecordingWrapper, MultiStepWrapper
from cleandiffuser.env.utils import VideoRecorder
from cleandiffuser.utils import at_least_ndim

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


# ══════════════════════════════════════════════════════════════════════════════
# FIXED: Timestep embedding
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# ══════════════════════════════════════════════════════════════════════════════
# EDITABLE: Conditioning Module
# ══════════════════════════════════════════════════════════════════════════════

# EDITABLE REGION START
class ConditioningModule(nn.Module):
    """Observation conditioning module for diffusion-based robot policy.

    Controls how observation information conditions the denoising network.
    The default implementation is a simple additive conditioning baseline
    (adds a projected observation embedding to features at each layer).

    Args:
        obs_dim: dimension of the flattened observation vector
        act_dim: dimension of the action vector
        hidden_dim: feature dimension of the base denoising network (default 256)

    Methods to implement:
        encode_condition(obs) -> cond:
            Process raw observation (B, obs_dim) into a conditioning
            representation. Can be a tensor, tuple, or any structure.

        condition_features(h, t_emb, cond, layer_idx) -> h:
            Apply conditioning to intermediate features h (B, hidden_dim)
            using timestep embedding t_emb (B, hidden_dim), conditioning
            representation cond, and layer index layer_idx (int).
            Must return tensor of shape (B, hidden_dim).
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Default: simple MLP to project obs to hidden_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode_condition(self, obs):
        """Encode observation into conditioning representation.

        Args:
            obs: (B, obs_dim) raw observation vector

        Returns:
            cond: conditioning representation (any structure)
        """
        return self.obs_encoder(obs)

    def condition_features(self, h, t_emb, cond, layer_idx):
        """Apply conditioning to intermediate denoising features.

        Args:
            h: (B, hidden_dim) intermediate features after residual block
            t_emb: (B, hidden_dim) timestep embedding
            cond: output of encode_condition
            layer_idx: int, which layer (0-indexed) is being conditioned

        Returns:
            h: (B, hidden_dim) conditioned features
        """
        # Default: additive conditioning
        return h + cond
# EDITABLE REGION END


# ══════════════════════════════════════════════════════════════════════════════
# FIXED: Base Denoising Network (residual MLP with conditioning hooks)
# ══════════════════════════════════════════════════════════════════════════════

class ResidualMLPBlock(nn.Module):
    """A single residual MLP block: Linear -> GELU -> Linear + skip."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x) + x


class DenoisingNetwork(nn.Module):
    """Fixed base denoising MLP with conditioning hooks.

    Architecture:
        1. Project noisy action to hidden_dim
        2. Add timestep embedding
        3. For each of 4 residual blocks:
           - Apply residual MLP block
           - Apply conditioning via ConditioningModule.condition_features()
        4. Project back to action_dim
    """

    def __init__(self, act_dim, hidden_dim, conditioning_module, n_blocks=4):
        super().__init__()
        self.act_proj = nn.Linear(act_dim, hidden_dim)
        self.time_emb = SinusoidalTimestepEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden_dim, act_dim)
        self.conditioning = conditioning_module

    def forward(self, x_noisy, timestep, obs):
        """
        Args:
            x_noisy: (B, act_dim) noisy action
            timestep: (B,) diffusion timestep (integer or continuous)
            obs: (B, obs_dim) observation

        Returns:
            noise_pred: (B, act_dim) predicted noise
        """
        # Timestep embedding
        t_emb = self.time_emb(timestep)
        t_emb = self.time_mlp(t_emb)

        # Encode observation condition
        cond = self.conditioning.encode_condition(obs)

        # Project noisy action and add timestep embedding
        h = self.act_proj(x_noisy) + t_emb

        # Pass through residual blocks with conditioning
        for i, block in enumerate(self.blocks):
            h = block(h)
            h = self.conditioning.condition_features(h, t_emb, cond, layer_idx=i)

        return self.out_proj(h)


# ══════════════════════════════════════════════════════════════════════════════
# FIXED: DDPM Training + DDIM Sampling
# ══════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """Cosine noise schedule for DDPM training and DDIM sampling."""

    def __init__(self, num_train_timesteps=100, device="cpu"):
        self.num_train_timesteps = num_train_timesteps
        self.device = device

        # Cosine schedule
        steps = num_train_timesteps + 1
        s = 0.008
        t = torch.linspace(0, num_train_timesteps, steps) / num_train_timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        self.betas = betas.to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, 0).to(device)

    def add_noise(self, x0, noise, timesteps):
        """Add noise to clean data: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps."""
        alpha_bar = self.alphas_cumprod[timesteps]
        alpha_bar = alpha_bar.view(-1, *([1] * (x0.dim() - 1)))
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    def ddim_step(self, x_t, noise_pred, t_idx, t_next_idx):
        """Single DDIM denoising step."""
        alpha_bar_t = self.alphas_cumprod[t_idx]
        alpha_bar_next = self.alphas_cumprod[t_next_idx] if t_next_idx >= 0 else torch.tensor(1.0, device=self.device)

        # Predict x0
        x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        x0_pred = x0_pred.clamp(-1, 1)

        # DDIM deterministic step
        x_next = alpha_bar_next.sqrt() * x0_pred + (1 - alpha_bar_next).sqrt() * noise_pred
        return x_next

    def ddim_sample(self, model, obs, shape, num_inference_steps=10):
        """DDIM sampling loop."""
        device = self.device
        B = shape[0]

        # Uniform timestep schedule for DDIM
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        timesteps = list(reversed(timesteps))

        x_t = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                noise_pred = model(x_t, t_batch, obs)

            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x_t = self.ddim_step(x_t, noise_pred, t, t_next)

        return x_t.clamp(-1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# FIXED: Environment + Evaluation
# ══════════════════════════════════════════════════════════════════════════════

ENV_CONFIGS = {
    "lift": {
        "obs_dim": 19, "action_dim": 10,
        "max_episode_steps": 400,
        "obs_keys": ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
    },
    "can": {
        "obs_dim": 23, "action_dim": 10,
        "max_episode_steps": 400,
        "obs_keys": ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
    },
    "square": {
        "obs_dim": 23, "action_dim": 10,
        "max_episode_steps": 500,
        "obs_keys": ("object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
    },
}


def make_env(dataset_path, env_cfg, seed, idx):
    """Create a single robomimic environment."""
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": list(env_cfg["obs_keys"])})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
    env = RobomimicLowdimWrapper(
        env=env, obs_keys=list(env_cfg["obs_keys"]),
        render_hw=(256, 256), render_camera_name="agentview")
    video_recorder = VideoRecorder.create_h264(
        fps=10, codec="h264", input_pix_fmt="rgb24", crf=22,
        thread_type="FRAME", thread_count=1)
    env = VideoRecordingWrapper(env, video_recorder, file_path=None, steps_per_render=2)
    env = MultiStepWrapper(env, n_obs_steps=1, n_action_steps=1,
                           max_episode_steps=env_cfg["max_episode_steps"])
    env.seed(seed + idx)
    return env


def evaluate(model, scheduler, dataset, env_cfg, dataset_path, seed,
             num_episodes=50, num_envs=10, num_inference_steps=10, device="cuda"):
    """Evaluate policy by rolling out in the environment."""
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]

    envs = gym.vector.SyncVectorEnv(
        [lambda idx=i: make_env(dataset_path, env_cfg, seed, idx) for i in range(num_envs)]
    )

    model.eval()
    episode_successes = []

    for batch_idx in range(num_episodes // num_envs):
        obs = envs.reset()  # (num_envs, 1, obs_dim)
        t = 0

        while t < env_cfg["max_episode_steps"]:
            obs_flat = obs.reshape(num_envs, -1).astype(np.float32)
            # Normalize observation
            nobs = dataset.normalizer["obs"]["state"].normalize(obs_flat)
            nobs = torch.tensor(nobs, device=device, dtype=torch.float32)

            with torch.no_grad():
                naction = scheduler.ddim_sample(
                    model, nobs, (num_envs, action_dim),
                    num_inference_steps=num_inference_steps)

            action_np = naction.cpu().numpy().clip(-1, 1)
            action_pred = dataset.normalizer["action"].unnormalize(action_np)
            action_pred = dataset.undo_transform_action(action_pred)
            action = action_pred.reshape(num_envs, 1, -1)

            obs, reward, done, info = envs.step(action)
            t += 1

        # Success = any positive reward during episode
        for env_idx in range(num_envs):
            rewards = envs.envs[env_idx].get_rewards()
            success = 1.0 if any(r > 0 for r in rewards) else 0.0
            episode_successes.append(success)

    envs.close()
    model.train()
    return np.mean(episode_successes)


# ══════════════════════════════════════════════════════════════════════════════
# FIXED: Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="can", choices=["lift", "can", "square"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--ema-rate", type=float, default=0.995)
    parser.add_argument("--dataset-root", type=str, default="/data/robomimic/datasets")
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Env: {args.env}, Seed: {args.seed}")

    env_cfg = ENV_CONFIGS[args.env]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset_path = os.path.join(args.dataset_root, args.env, "ph", "low_dim_abs.hdf5")
    print(f"Loading dataset: {dataset_path}")
    dataset = RobomimicDataset(
        dataset_path, horizon=1, obs_keys=env_cfg["obs_keys"],
        pad_before=0, pad_after=0, abs_action=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)
    print(f"Dataset: {dataset}")

    # ── Model ──────────────────────────────────────────────────────────────────
    conditioning = ConditioningModule(obs_dim, action_dim, args.hidden_dim)
    model = DenoisingNetwork(action_dim, args.hidden_dim, conditioning).to(device)
    model_ema = deepcopy(model).requires_grad_(False).eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    n_cond_params = sum(p.numel() for p in conditioning.parameters())
    print(f"Total params: {n_params:,} | Conditioning params: {n_cond_params:,}")

    # ── Training ───────────────────────────────────────────────────────────────
    best_success = 0.0
    step = 0
    start_time = time.time()
    loss_accum = []

    for batch in loop_dataloader(dataloader):
        if step >= args.gradient_steps:
            break

        obs = batch["obs"]["state"].to(device).squeeze(1)  # (B, obs_dim)
        action = batch["action"].to(device).squeeze(1)      # (B, action_dim)

        # DDPM forward: add noise
        noise = torch.randn_like(action)
        t = torch.randint(0, args.diffusion_steps, (action.shape[0],), device=device)
        noisy_action = scheduler.add_noise(action, noise, t)

        # Predict noise
        noise_pred = model(noisy_action, t, obs)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p, p_ema in zip(model.parameters(), model_ema.parameters()):
                p_ema.data.mul_(args.ema_rate).add_(p.data, alpha=1.0 - args.ema_rate)

        loss_accum.append(loss.item())
        step += 1

        # Log
        if step % 5000 == 0:
            avg_loss = np.mean(loss_accum[-5000:])
            elapsed = time.time() - start_time
            print(f"TRAIN_METRICS step={step} loss={avg_loss:.6f} "
                  f"elapsed={elapsed:.0f}s", flush=True)

        # Evaluate
        if step % args.eval_freq == 0 or step == args.gradient_steps:
            print(f"\n--- Evaluation at step {step} ---")
            success_rate = evaluate(
                model_ema, scheduler, dataset, env_cfg, dataset_path,
                seed=args.seed, num_episodes=args.eval_episodes,
                num_envs=args.num_envs, num_inference_steps=args.ddim_steps,
                device=device)

            if success_rate > best_success:
                best_success = success_rate

            print(f"TRAIN_METRICS step={step} success_rate={success_rate:.4f} "
                  f"best_success={best_success:.4f}", flush=True)

    # ── Final Evaluation ───────────────────────────────────────────────────────
    print("\n--- Final Evaluation ---")
    final_success = evaluate(
        model_ema, scheduler, dataset, env_cfg, dataset_path,
        seed=args.seed, num_episodes=args.eval_episodes,
        num_envs=args.num_envs, num_inference_steps=args.ddim_steps,
        device=device)

    best_success = max(best_success, final_success)
    print(f"\nTEST_METRICS success_rate={best_success:.4f}", flush=True)
    print(f"\nTraining complete. Best success rate: {best_success:.4f}")


if __name__ == "__main__":
    main()
