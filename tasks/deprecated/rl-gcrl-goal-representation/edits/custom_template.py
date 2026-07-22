"""Goal-conditioned RL with custom goal representation.

This script trains a GCIVL agent with a pluggable GoalRepresentation module.
The GoalRepresentation defines how raw goal observations are encoded before
being passed to actor, value, and contrastive modules.

EDITABLE REGION: GoalRepresentation class (encode_goal, compute_rep_loss, setup).
FIXED: Everything else (GCIVL agent, training loop, evaluation, dataset loading).
"""

import copy
import os
import sys
import time
import random
import argparse
from typing import Any, Sequence
from collections import defaultdict

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import numpy as np
import optax
import tqdm

# ===== Imports from the dual-goal-representations codebase =====
sys.path.insert(0, os.path.join(os.environ.get("PKG_DIR", "/workspace/dual-goal-representations")))
from utils.networks import MLP, GCValue, GCActor, GCBilinearValue, ensemblize
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.datasets import Dataset, GCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate


# ############################################################################
# EDITABLE REGION START — GoalRepresentation
# ############################################################################

class GoalRepresentation(nn.Module):
    """Goal representation module that encodes raw goal observations.

    This module defines how goals are represented before being passed to
    the actor, value function, and contrastive learning components.

    The default implementation is an identity mapping (raw state as goal).
    Modify this class to implement novel goal representation strategies.

    Attributes:
        obs_dim: Observation/goal dimension.
        rep_dim: Output representation dimension.
        hidden_dims: Hidden layer dimensions for the representation network.
        layer_norm: Whether to apply layer normalization.
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        """Initialize the representation network.

        Override this to define custom network architectures for
        goal encoding. The default is an identity mapping.
        """
        pass

    def encode_goal(self, goals):
        """Encode raw goal observations into a learned representation.

        Args:
            goals: Raw goal observations of shape (batch, obs_dim).

        Returns:
            Goal representations of shape (batch, rep_dim).
            For the default identity, rep_dim must equal obs_dim.
        """
        return goals

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        """Compute auxiliary loss for training the representation.

        Args:
            observations: Current observations (batch, obs_dim).
            goals: Goal observations (batch, obs_dim).
            next_observations: Next observations (batch, obs_dim).
            rewards: Rewards (batch,).
            masks: Continuation masks (batch,), 0 at goal.
            actions: Actions (batch, act_dim), optional.

        Returns:
            Scalar loss value. Return 0.0 if no auxiliary loss is needed.
            Info dict with auxiliary metrics.
        """
        return 0.0, {}

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        """Forward pass.

        Args:
            goals: Raw goal observations.
            mode: 'encode' to return encoded goals, 'rep_loss' to compute
                  the representation auxiliary loss.
            Other args: Only used when mode='rep_loss'.

        Returns:
            If mode='encode': Encoded goal representations.
            If mode='rep_loss': (loss, info_dict).
        """
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations,
                rewards, masks, actions)
        return self.encode_goal(goals)

# ############################################################################
# EDITABLE REGION END
# ############################################################################


# ============================================================================
# FIXED: GCIVL Agent with Goal Representation
# ============================================================================

class GCIVLRepAgent(flax.struct.PyTreeNode):
    """GCIVL agent with pluggable goal representation."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss using encoded goals."""
        goal_reps = self.network.select('goal_rep')(
            batch['value_goals'], params=grad_params)

        (next_v1_t, next_v2_t) = self.network.select('target_value')(
            batch['next_observations'], goal_reps)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(
            batch['observations'], goal_reps)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(
            batch['observations'], goal_reps, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss using encoded goals."""
        goal_reps = self.network.select('goal_rep')(batch['actor_goals'])

        v1, v2 = self.network.select('value')(batch['observations'], goal_reps)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], goal_reps)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(
            batch['observations'], goal_reps, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update({
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            })

        return actor_loss, actor_info

    def rep_loss(self, batch, grad_params):
        """Compute the goal representation auxiliary loss."""
        rep_loss_val, rep_info = self.network.select('goal_rep')(
            batch['value_goals'],
            observations=batch['observations'],
            next_observations=batch['next_observations'],
            rewards=batch['rewards'],
            masks=batch['masks'],
            actions=batch['actions'],
            mode='rep_loss',
            params=grad_params,
        )
        return rep_loss_val, rep_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        rep_loss_val, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        loss = value_loss + actor_loss + rep_loss_val
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
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        goal_reps = self.network.select('goal_rep')(goals)
        dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        obs_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        rep_dim = config['rep_dim']

        # Goal representation module.
        goal_rep_def = GoalRepresentation(
            obs_dim=obs_dim,
            rep_dim=rep_dim,
            hidden_dims=config['rep_hidden_dims'],
            layer_norm=config['layer_norm'],
        )

        # For identity representation, rep_dim == obs_dim.
        ex_goal_reps = jnp.zeros(shape=(1, rep_dim))

        # Value and actor networks take encoded goals.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (ex_observations,)),
            value=(value_def, (ex_observations, ex_goal_reps)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goal_reps)),
            actor=(actor_def, (ex_observations, ex_goal_reps)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ============================================================================
# FIXED: Training and Evaluation Loop
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GCIVL with Custom Goal Representation')
    parser.add_argument('--env_name', type=str, default='antmaze-large-navigate-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_steps', type=int, default=1000000)
    parser.add_argument('--eval_interval', type=int, default=100000)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--eval_tasks', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--expectile', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--rep_dim', type=int, default=256,
                        help='Goal representation dimension.')
    parser.add_argument('--save_dir', type=str, default='exp/')
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration dict.
    config = dict(
        lr=args.lr,
        batch_size=args.batch_size,
        rep_hidden_dims=(512, 512, 512),
        actor_hidden_dims=(512, 512, 512),
        value_hidden_dims=(512, 512, 512),
        layer_norm=True,
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        alpha=args.alpha,
        const_std=True,
        discrete=False,
        rep_dim=args.rep_dim,
        # Dataset config.
        dataset_class='GCDataset',
        oraclerep=False,
        norm=False,
        value_p_curgoal=0.2,
        value_p_trajgoal=0.5,
        value_p_randomgoal=0.3,
        value_geom_sample=True,
        actor_p_curgoal=0.0,
        actor_p_trajgoal=1.0,
        actor_p_randomgoal=0.0,
        actor_geom_sample=False,
        gc_negative=True,
        p_aug=0.0,
        frame_stack=None,
    )

    # Set up environment and dataset.
    env, train_dataset, val_dataset = make_env_and_datasets(args.env_name, frame_stack=None)
    train_dataset = GCDataset(Dataset.create(norm=False, **train_dataset), config)

    # Auto-set rep_dim to obs_dim for identity representation if needed.
    obs_dim = train_dataset.dataset['observations'].shape[-1]
    if args.rep_dim <= 0:
        config['rep_dim'] = obs_dim
        args.rep_dim = obs_dim

    # Initialize agent.
    random.seed(args.seed)
    np.random.seed(args.seed)

    example_batch = train_dataset.sample(1)
    agent = GCIVLRepAgent.create(
        args.seed, example_batch['observations'], example_batch['actions'], config
    )

    # Training loop.
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, args.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log training metrics.
        if i % args.log_interval == 0:
            metrics_parts = [f"step={i}"]
            for k, v in update_info.items():
                metrics_parts.append(f"{k}={float(v):.6f}")
            metrics_parts.append(f"time={time.time() - first_time:.1f}")
            print("TRAIN_METRICS " + " ".join(metrics_parts), flush=True)
            last_time = time.time()

        # Evaluate.
        if i == 1 or i % args.eval_interval == 0:
            eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            overall_metrics = defaultdict(list)
            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            num_tasks = args.eval_tasks if args.eval_tasks is not None else len(task_infos)

            for task_id in tqdm.trange(1, num_tasks + 1, desc='Evaluating'):
                eval_info, _, _ = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=args.eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=3,
                    eval_temperature=0.0,
                    eval_gaussian=None,
                    eval_goal_gaussian=None,
                    diff=None,
                )
                for k, v in eval_info.items():
                    if k == 'success':
                        overall_metrics[k].append(v)

            if overall_metrics.get('success'):
                mean_success = float(np.mean(overall_metrics['success']))
                print(f"TEST_METRICS step={i} success_rate={mean_success:.4f}", flush=True)
            else:
                print(f"TEST_METRICS step={i} success_rate=0.0000", flush=True)

    print("Training complete.", flush=True)


if __name__ == '__main__':
    main()
