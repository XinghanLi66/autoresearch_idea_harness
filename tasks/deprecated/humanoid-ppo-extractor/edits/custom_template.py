import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
import humanoid_bench
# ── Custom imports (editable) ────────────────────────────────────────────


# ======================================================================
# EDITABLE — Custom feature extractor
# ======================================================================
class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for PPO on humanoid control.

    Input: observation vector from the environment (observation_space.shape[0] dims).
    Output: feature vector of size features_dim.
    """
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(n_input, 64),
            nn.Tanh(),
            nn.Linear(64, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations):
        return self.net(observations)

# ======================================================================
# FIXED — Training infrastructure (do not edit below)
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO with custom feature extractor for humanoid control"
    )
    parser.add_argument("--env_name", type=str, default="h1-stand-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    return parser.parse_args()


def make_env(env_name, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        env.action_space.seed(seed + rank)
        return env
    return _init


class TrainMetricsCallback(BaseCallback):
    """Callback that prints training metrics in a parseable format."""

    def __init__(self, log_interval=50000, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        if (self.num_timesteps % self.log_interval < self.model.n_envs
                and self.episode_rewards):
            recent = self.episode_rewards[-100:]
            mean_r = np.mean(recent)
            mean_l = np.mean(self.episode_lengths[-100:])
            print(
                f"TRAIN_METRICS timestep={self.num_timesteps} "
                f"mean_reward={mean_r:.2f} mean_length={mean_l:.1f}",
                flush=True,
            )
        return True


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = SubprocVecEnv(
        [make_env(args.env_name, i, args.seed) for i in range(args.num_envs)]
    )

    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs={},
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        learning_rate=args.learning_rate,
        batch_size=256,
        n_steps=2048,
        seed=args.seed,
    )

    callback = TrainMetricsCallback(log_interval=50000)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # Final evaluation
    eval_env = DummyVecEnv(
        [make_env(args.env_name, 0, args.seed + 1000)]
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    print(
        f"TEST_METRICS mean_reward={mean_reward:.2f} "
        f"std_reward={std_reward:.2f}",
        flush=True,
    )

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
