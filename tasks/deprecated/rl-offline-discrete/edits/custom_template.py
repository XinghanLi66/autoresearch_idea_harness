# Custom offline RL algorithm for MLS-Bench — Atari discrete control
#
# EDITABLE section: QNetwork class + OfflineAlgorithm class.
# FIXED sections: everything else (config, encoder, buffer, eval, training loop).
import argparse
import os
import random

import ale_py
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# FIXED: Configuration
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--fraction", type=float, default=0.01,
                        help="Fraction of transitions per epoch (matching d3rlpy)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_timesteps", type=int, default=1_500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--optim_eps", type=float, default=1e-8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--checkpoints_path", type=str, default=None)
    return parser.parse_args()


# =====================================================================
# FIXED: Utilities
# =====================================================================
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


# =====================================================================
# FIXED: Nature DQN Encoder
# =====================================================================
class NatureDQNEncoder(nn.Module):
    """Nature DQN CNN encoder (Mnih et al. 2015).

    Input: (B, in_channels, 84, 84) float32 [0,1]
    Output: (B, feature_dim) feature vector
    """
    def __init__(self, in_channels: int = 4, feature_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x).reshape(x.size(0), -1))


# =====================================================================
# FIXED: Atari Replay Buffer
# =====================================================================
class AtariReplayBuffer:
    """Replay buffer for d4rl-atari offline data with frame stacking.

    Stores raw single-frame (N, 84, 84) uint8 observations.
    sample() returns frame-stacked (B, 4, 84, 84) float32 [0,1] observations.
    Frame stacking respects episode boundaries.
    Rewards are clipped to [-1, 1] (matches d3rlpy ClipRewardScaler).
    """
    def __init__(self, dataset: dict, frame_stack: int = 4):
        obs = dataset["observations"]
        if obs.ndim == 4:
            obs = obs.squeeze(1)  # (N, 1, 84, 84) -> (N, 84, 84)
        self.observations = obs  # (N, 84, 84) uint8
        self.actions = dataset["actions"].astype(np.int64)
        self.rewards = np.clip(dataset["rewards"].astype(np.float32), -1.0, 1.0)
        self.terminals = dataset["terminals"].astype(np.float32)
        self.frame_stack = frame_stack
        self.size = len(self.actions)

        # Precompute episode start flags for frame stacking
        self._episode_starts = np.zeros(self.size, dtype=bool)
        self._episode_starts[0] = True
        for i in range(1, self.size):
            if self.terminals[i - 1] > 0.5:
                self._episode_starts[i] = True

        # Precompute episode id for each step (fast boundary checking)
        self._episode_ids = np.zeros(self.size, dtype=np.int32)
        ep_id = 0
        for i in range(self.size):
            if self._episode_starts[i] and i > 0:
                ep_id += 1
            self._episode_ids[i] = ep_id

    def _get_stacked_obs(self, idx: int) -> np.ndarray:
        """Get frame-stacked observation at index idx. Shape: (4, 84, 84)."""
        frames = []
        ep_id = self._episode_ids[idx]
        for offset in range(self.frame_stack - 1, -1, -1):
            frame_idx = idx - offset
            if frame_idx < 0 or self._episode_ids[frame_idx] != ep_id:
                frames.append(np.zeros((84, 84), dtype=np.uint8))
            else:
                frames.append(self.observations[frame_idx])
        return np.stack(frames, axis=0)

    def sample(self, batch_size: int) -> tuple:
        """Sample a batch. Returns (obs, actions, rewards, next_obs, dones).

        obs, next_obs: (B, 4, 84, 84) float32 [0,1] on DEVICE
        actions: (B,) int64 on DEVICE
        rewards: (B,) float32 on DEVICE (clipped to [-1, 1])
        dones: (B,) float32 on DEVICE
        """
        indices = np.random.randint(0, self.size - 1, size=batch_size)

        obs_batch = np.stack([self._get_stacked_obs(i) for i in indices])
        next_obs_batch = np.stack([self._get_stacked_obs(i + 1) for i in indices])

        obs = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE) / 255.0
        next_obs = torch.tensor(next_obs_batch, dtype=torch.float32, device=DEVICE) / 255.0
        actions = torch.tensor(self.actions[indices], dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(self.terminals[indices], dtype=torch.float32, device=DEVICE)

        return obs, actions, rewards, next_obs, dones


# =====================================================================
# FIXED: Evaluation
# =====================================================================
def make_atari_eval_env(game: str):
    """Create a preprocessed Atari evaluation environment with frame stacking."""
    game_name = game.capitalize()
    env = gymnasium.make(f"ALE/{game_name}-v5", render_mode=None, frameskip=1)
    env = gymnasium.wrappers.AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=84,
        terminal_on_life_loss=False, grayscale_obs=True, scale_obs=False,
    )
    return _FrameStack(env, stack_size=4)


class _FrameStack:
    """Simple frame-stacking wrapper for evaluation."""
    def __init__(self, env, stack_size=4):
        self.env = env
        self.stack_size = stack_size
        self.frames = np.zeros((stack_size, 84, 84), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[:] = 0
        self.frames[-1] = obs
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = obs
        return self.frames.copy(), reward, terminated, truncated, info

    def close(self):
        self.env.close()


@torch.no_grad()
def evaluate(env, algorithm, num_episodes: int, device: str) -> float:
    """Evaluate algorithm on Atari environment. Returns mean episode reward."""
    algorithm.model.eval()
    episode_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        algorithm.begin_episode()
        while not done:
            obs_t = torch.tensor(
                np.array(obs), dtype=torch.float32, device=device
            ).unsqueeze(0) / 255.0
            action = algorithm.select_action(obs_t)
            obs, reward, terminated, truncated, _ = env.step(action)
            algorithm.observe(reward)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    algorithm.model.train()
    return float(np.mean(episode_rewards))


# =====================================================================
# EDITABLE: QNetwork and OfflineAlgorithm
# =====================================================================
class QNetwork(nn.Module):
    """Q-network for discrete actions. Uses NatureDQNEncoder."""
    def __init__(self, observation_shape, action_dim, feature_dim=512):
        super().__init__()
        self.encoder = NatureDQNEncoder(observation_shape[0], feature_dim)
        self.head = nn.Linear(feature_dim, action_dim)

    def forward(self, obs):
        return self.head(self.encoder(obs))


class OfflineAlgorithm:
    """Offline RL algorithm. Default: Behavioral Cloning (BC)."""
    def __init__(self, observation_shape, action_dim, config, device, buffer):
        self.device = device
        self.config = config
        self.buffer = buffer
        self.model = QNetwork(observation_shape, action_dim).to(device)

    def parameters(self):
        return self.model.parameters()

    def train_step(self, obs, actions, rewards, next_obs, dones):
        logits = self.model(obs)
        loss = F.cross_entropy(logits, actions)
        return loss, {"loss": loss.item()}

    def after_gradient_step(self, optimizer):
        pass

    def select_action(self, obs):
        return torch.argmax(self.model(obs), dim=-1).item()

    def begin_episode(self):
        pass

    def observe(self, reward):
        pass


# =====================================================================
# FIXED: Training loop
# =====================================================================
def train():
    config = parse_args()
    print(f"Game: {config.game}, Fraction: {config.fraction}, Seed: {config.seed}")
    print(f"Device: {DEVICE}")

    set_seed(config.seed)

    if config.checkpoints_path is not None:
        os.makedirs(config.checkpoints_path, exist_ok=True)

    # Load dataset: sample fraction from all 50 epochs
    # (matching d3rlpy get_atari_transitions)
    # Memory-efficient: pre-allocate arrays and fill in-place
    import gzip, gc
    game_name = config.game.capitalize()
    data_root = os.path.expanduser(f"~/.d4rl/datasets/{game_name}/1")
    n_per_epoch = int(1_000_000 * config.fraction)
    n_epochs = 50
    total_n = n_per_epoch * n_epochs
    # Pre-allocate final arrays
    all_obs = np.empty((total_n, 84, 84), dtype=np.uint8)
    all_act = np.empty(total_n, dtype=np.int64)
    all_rew = np.empty(total_n, dtype=np.float32)
    all_term = np.empty(total_n, dtype=np.float32)
    offset = 0
    for epoch in range(1, n_epochs + 1):
        epoch_dir = os.path.join(data_root, str(epoch))
        if not os.path.isdir(epoch_dir):
            break
        # Load small arrays first (actions/rewards/terminals ~4MB each)
        with gzip.open(os.path.join(epoch_dir, "action.gz"), "rb") as f:
            act = np.load(f)
        n = min(n_per_epoch, len(act))
        all_act[offset:offset+n] = act[:n].astype(np.int64)
        del act
        with gzip.open(os.path.join(epoch_dir, "reward.gz"), "rb") as f:
            rew = np.load(f)
        all_rew[offset:offset+n] = rew[:n].astype(np.float32)
        del rew
        with gzip.open(os.path.join(epoch_dir, "terminal.gz"), "rb") as f:
            term = np.load(f)
        all_term[offset:offset+n] = term[:n].astype(np.float32)
        del term
        # Load observations last (largest: ~7GB), slice immediately
        with gzip.open(os.path.join(epoch_dir, "observation.gz"), "rb") as f:
            obs = np.load(f)
        all_obs[offset:offset+n] = obs[:n]
        del obs
        gc.collect()
        offset += n
        print(f"  Loaded epoch {epoch}: {n} transitions", flush=True)
    # Trim if fewer transitions than expected
    all_obs = all_obs[:offset]
    all_act = all_act[:offset]
    all_rew = all_rew[:offset]
    all_term = all_term[:offset]
    # Get action_dim from ALE environment spec
    import gymnasium
    _tmp_env = gymnasium.make(f"ALE/{game_name}-v5")
    action_dim = _tmp_env.action_space.n
    _tmp_env.close()
    dataset = {
        "observations": all_obs,
        "actions": all_act,
        "rewards": all_rew,
        "terminals": all_term,
    }
    buffer = AtariReplayBuffer(dataset, frame_stack=4)
    del dataset, all_obs, all_act, all_rew, all_term
    gc.collect()
    obs_shape = (4, 84, 84)
    print(f"Dataset size: {buffer.size}, Action dim: {action_dim}")

    # Create eval env (gymnasium)
    eval_env = make_atari_eval_env(config.game)

    algorithm = OfflineAlgorithm(obs_shape, action_dim, config, DEVICE, buffer)

    # ── FIXED: Parameter count guard ──
    MAX_PARAMS = 5_000_000
    total_params = sum(p.numel() for p in algorithm.parameters())
    assert total_params <= MAX_PARAMS, (
        f"Model has {total_params:,} parameters, exceeding the limit of {MAX_PARAMS:,}. "
        f"Do not inflate network capacity to gain unfair advantage."
    )
    # ── FIXED: Encoder integrity check ──
    if hasattr(algorithm.model, "encoder"):
        enc = algorithm.model.encoder
        assert isinstance(enc, NatureDQNEncoder), (
            f"QNetwork.encoder must be NatureDQNEncoder, got {type(enc).__name__}. "
            f"Do not replace the fixed encoder."
        )
        # Verify conv layers match standard Nature DQN (only fc dim may vary)
        ref = NatureDQNEncoder(obs_shape[0], 512)
        ref_conv_params = sum(p.numel() for p in ref.conv.parameters())
        enc_conv_params = sum(p.numel() for p in enc.conv.parameters())
        assert enc_conv_params == ref_conv_params, (
            f"Encoder conv has {enc_conv_params:,} params but expected {ref_conv_params:,}. "
            f"Do not modify the convolutional layers."
        )
        del ref

    if hasattr(algorithm, "create_optimizer"):
        optimizer = algorithm.create_optimizer()
    else:
        optimizer = torch.optim.Adam(
            algorithm.parameters(), lr=config.learning_rate, eps=config.optim_eps
        )
    print(f"Parameters: {total_params:,}")

    for step in range(1, config.max_timesteps + 1):
        obs, actions, rewards, next_obs, dones = buffer.sample(config.batch_size)
        loss, info = algorithm.train_step(obs, actions, rewards, next_obs, dones)
        optimizer.zero_grad()
        loss.backward()
        clip_norm = getattr(algorithm, "clip_grad_norm", None)
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(algorithm.parameters(), clip_norm)
        optimizer.step()
        algorithm.after_gradient_step(optimizer)

        if step % 1000 == 0:
            metrics_str = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in info.items()
            )
            print(f"TRAIN_METRICS step={step} {metrics_str}", flush=True)

        if step % config.eval_freq == 0:
            score = evaluate(eval_env, algorithm, config.eval_episodes, DEVICE)
            print(f"Eval return ({config.game}): {score:.2f}")
            print(f"D4RL score: {score:.6f}", flush=True)

    eval_env.close()


if __name__ == "__main__":
    train()
