"""Offline RL policy training on OGBench.

This script trains an offline RL agent on OGBench environments.
The critic (Value network with TD loss) and the training loop are FIXED.
Only the PolicyTrainer class (policy network creation, actor loss, action sampling)
is editable -- this defines the policy training approach.
"""

import copy
import os
import time
from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from collections import defaultdict
from tqdm import tqdm, trange

# ============================================================================
# Network primitives (FIXED)
# ============================================================================


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


import distrax


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class GaussianActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)
        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        return distribution


class VectorFieldActor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)
        return self.mlp(inputs)


class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.value_net(inputs).squeeze(-1)


# ============================================================================
# Flax utilities (FIXED)
# ============================================================================

import functools

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    modules: dict

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, (list, tuple)):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None
        return cls(step=1, apply_fn=model_def.apply, model_def=model_def, params=params,
                   tx=tx, opt_state=opt_state, **kwargs)

    def __call__(self, *args, params=None, method=None, **kwargs):
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None
        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state, **kwargs)

    def apply_loss_fn(self, loss_fn):
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        return self.apply_gradients(grads=grads), info


# ============================================================================
# Dataset (FIXED)
# ============================================================================

from flax.core.frozen_dict import FrozenDict


def get_size(data):
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        return batch

    def get_subset(self, idxs):
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result


# ============================================================================
# Evaluation utilities (FIXED)
# ============================================================================


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def evaluate(agent, env, policy_fn, num_eval_episodes=50, eval_temperature=0):
    actor_fn = supply_rng(policy_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    stats = defaultdict(list)
    for i in trange(num_eval_episodes, desc='Evaluating'):
        observation, info = env.reset()
        done = False
        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
        for k, v in flatten(info).items():
            stats[k].append(v)
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


# ============================================================================
# Environment and dataset loading (FIXED)
# ============================================================================

import gymnasium
import ogbench


class EpisodeMonitor(gymnasium.Wrapper):
    def __init__(self, env, filter_regexes=None):
        import re
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        import re
        observation, reward, terminated, truncated, info = self.env.step(action)
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}
        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time
        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


def make_env_and_datasets(env_name, action_clip_eps=1e-5):
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
    eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
    env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    env.reset()
    eval_env.reset()
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )
    return env, eval_env, train_dataset, val_dataset


# ============================================================================
# PolicyTrainer — EDITABLE SECTION
# ============================================================================
# Design your policy training approach below. You may use any combination of:
# - Flow matching + distillation (like FQL)
# - Advantage-weighted regression (like IQL)
# - DDPG + BC regularization (like ReBRAC/TD3+BC)
# - Or any novel approach
#
# The critic (Value network with ensemble=2) is FIXED and provided via
# `network.select('critic')` and `network.select('target_critic')`.
#
# Available network primitives: GaussianActor, VectorFieldActor, Value,
# MLP, ModuleDict, TrainState, distrax, optax.


class PolicyTrainer:
    """Defines the policy training approach for offline RL.

    Implement the three methods below to define:
    1. What policy network(s) to create
    2. How to compute the actor loss
    3. How to sample actions at inference time
    """

    @staticmethod
    def create_policy_networks(obs_dim, action_dim, hidden_dims, layer_norm):
        """Create policy network definitions and their example inputs.

        Args:
            obs_dim: Observation dimension (int).
            action_dim: Action dimension (int).
            hidden_dims: Tuple of hidden layer sizes, e.g. (512, 512, 512, 512).
            layer_norm: Whether to use layer norm in actor networks (bool).

        Returns:
            A dict mapping network name -> (module_def, example_inputs_tuple).
            These will be added to the ModuleDict alongside the fixed critic networks.
            Example:
                {
                    'actor': (GaussianActor(...), (ex_observations,)),
                }
        """
        raise NotImplementedError("Implement create_policy_networks")

    @staticmethod
    def compute_actor_loss(network, batch, config, rng):
        """Compute the actor loss given a batch of data.

        Args:
            network: TrainState containing all networks (critic, target_critic, + your policy nets).
                     Use network.select('name') to access specific networks.
                     Use params=grad_params when you want gradients to flow through a network.
            batch: Dict with keys 'observations', 'actions', 'rewards', 'masks', 'next_observations'.
            config: Dict with hyperparameters (discount, tau, alpha, lr, etc.).
            rng: JAX random key.

        Returns:
            (actor_loss, info_dict) where info_dict contains metrics for logging.

        Note: grad_params is handled by the caller -- the network's params are the
              differentiable ones inside jax.grad. Use `network.select('actor')(obs)`
              (without explicit params) since the caller wraps this in grad context.
        """
        raise NotImplementedError("Implement compute_actor_loss")

    @staticmethod
    def sample_actions(network, observations, config, seed, temperature=1.0):
        """Sample actions for evaluation.

        Args:
            network: TrainState containing all networks.
            observations: Observation array.
            config: Dict with hyperparameters.
            seed: JAX random key.
            temperature: Sampling temperature (0 = deterministic).

        Returns:
            Action array, clipped to [-1, 1].
        """
        raise NotImplementedError("Implement sample_actions")

    @staticmethod
    def get_target_update_modules():
        """Return list of module names that need target network updates.

        The critic target update is always done. Return additional module names
        if your policy approach uses target networks (e.g., ['actor'] for TD3-style).

        Returns:
            List of module name strings, e.g. [] or ['actor'].
        """
        return []

    @staticmethod
    def needs_next_actions():
        """Whether the dataset should provide next_actions in batches.

        Return True if your approach needs batch['next_actions'] (e.g., for
        critic regularization like ReBRAC).

        Returns:
            bool
        """
        return False


# ============================================================================
# Agent (FIXED)
# ============================================================================


class OfflineRLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Fixed critic loss: standard TD learning with target network."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = PolicyTrainer.sample_actions(
            self.network, batch['next_observations'], self.config, sample_rng, temperature=0.0
        )
        next_actions = jnp.clip(next_actions, -1, 1)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config.get('q_agg', 'mean') == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = PolicyTrainer.compute_actor_loss(
            self.network.replace(params=grad_params), batch, self.config, actor_rng
        )
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        for mod in PolicyTrainer.get_target_update_modules():
            self.target_update(new_network, mod)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        return PolicyTrainer.sample_actions(self.network, observations, self.config, seed, temperature)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        obs_dim = ex_observations.shape[-1]
        ex_times = ex_actions[..., :1]

        # Fixed critic networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
        )

        # Add policy networks from PolicyTrainer.
        policy_nets = PolicyTrainer.create_policy_networks(
            obs_dim, action_dim, config['actor_hidden_dims'], config.get('actor_layer_norm', False)
        )
        network_info.update(policy_nets)

        # Add target networks for policy modules that need them.
        for mod in PolicyTrainer.get_target_update_modules():
            if mod in policy_nets:
                network_info[f'target_{mod}'] = (copy.deepcopy(policy_nets[mod][0]), policy_nets[mod][1])

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        for mod in PolicyTrainer.get_target_update_modules():
            params[f'modules_target_{mod}'] = params[f'modules_{mod}']

        config['obs_dim'] = obs_dim
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ============================================================================
# Training loop (FIXED)
# ============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offline_steps', type=int, default=1000000)
    parser.add_argument('--eval_interval', type=int, default=100000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--q_agg', type=str, default='mean')
    parser.add_argument('--normalize_q_loss', action='store_true', default=False)
    args = parser.parse_args()

    # Set up environment.
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(args.env_name)
    if PolicyTrainer.needs_next_actions():
        train_dataset.return_next_actions = True

    # Set seeds.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Config dict.
    config = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        q_agg=args.q_agg,
        normalize_q_loss=args.normalize_q_loss,
        value_hidden_dims=(512, 512, 512, 512),
        actor_hidden_dims=(512, 512, 512, 512),
        layer_norm=True,
        actor_layer_norm=False,
    )

    # Create agent.
    example_batch = train_dataset.sample(1)
    agent = OfflineRLAgent.create(
        args.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Train.
    first_time = time.time()
    last_time = time.time()

    for i in tqdm(range(1, args.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        if i % args.log_interval == 0:
            elapsed = time.time() - first_time
            step_time = (time.time() - last_time) / args.log_interval
            last_time = time.time()
            critic_loss = float(update_info.get('critic/critic_loss', 0))
            q_mean = float(update_info.get('critic/q_mean', 0))
            actor_loss = float(update_info.get('actor/actor_loss', 0))
            print(f"TRAIN_METRICS step={i} critic_loss={critic_loss:.4f} q_mean={q_mean:.4f} "
                  f"actor_loss={actor_loss:.4f} step_time={step_time:.4f} elapsed={elapsed:.1f}",
                  flush=True)

        if args.eval_interval != 0 and (i == 1 or i % args.eval_interval == 0):
            eval_stats = evaluate(
                agent=agent,
                env=eval_env,
                policy_fn=agent.sample_actions,
                num_eval_episodes=args.eval_episodes,
                eval_temperature=0,
            )
            success = float(eval_stats.get('success', 0.0))
            print(f"TEST_METRICS step={i} success_rate={success:.4f}", flush=True)

    # Final evaluation.
    eval_stats = evaluate(
        agent=agent,
        env=eval_env,
        policy_fn=agent.sample_actions,
        num_eval_episodes=args.eval_episodes,
        eval_temperature=0,
    )
    success = float(eval_stats.get('success', 0.0))
    print(f"TEST_METRICS step={args.offline_steps} success_rate={success:.4f}", flush=True)
    print(f"Total training time: {time.time() - first_time:.1f}s", flush=True)


if __name__ == '__main__':
    main()
