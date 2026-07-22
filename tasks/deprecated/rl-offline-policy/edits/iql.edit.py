"""IQL baseline -- Implicit Q-Learning (Kostrikov, Nair, Levine, 2022).

Policy approach: Train a separate V(s) network with expectile regression,
then use advantage-weighted regression (AWR) for the actor.
Actor loss = -E[exp(alpha * A) * log pi(a|s)] where A = Q(s,a) - V(s).

Note: IQL also needs a value network V(s) in addition to Q(s,a).
"""

_FILE = "fql/custom_train.py"

_IQL_POLICY_TRAINER = """\
class PolicyTrainer:
    \"\"\"IQL: Implicit Q-Learning policy trainer.

    Uses expectile regression for value function, then advantage-weighted
    regression for the actor. No policy gradient through Q-values.
    \"\"\"

    @staticmethod
    def create_policy_networks(obs_dim, action_dim, hidden_dims, layer_norm):
        ex_obs = jnp.zeros((1, obs_dim))

        actor_def = GaussianActor(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            layer_norm=layer_norm,
            state_dependent_std=False,
            const_std=True,
        )

        value_def = Value(
            hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            num_ensembles=1,
        )

        return {
            'actor': (actor_def, (ex_obs,)),
            'value': (value_def, (ex_obs,)),
        }

    @staticmethod
    def compute_actor_loss(network, batch, config, rng):
        # Expectile value loss.
        expectile = 0.9
        q1, q2 = jax.lax.stop_gradient(
            network.select('target_critic')(batch['observations'], actions=batch['actions'])
        )
        q = jnp.minimum(q1, q2)
        v = network.select('value')(batch['observations'])
        diff = q - v
        weight = jnp.where(diff >= 0, expectile, (1 - expectile))
        value_loss = (weight * (diff ** 2)).mean()

        # AWR actor loss.
        v_detached = jax.lax.stop_gradient(v)
        adv = q - v_detached
        alpha = config.get('alpha', 10.0)
        exp_a = jnp.exp(adv * alpha)
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = network.select('actor')(batch['observations'])
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(jax.lax.stop_gradient(exp_a) * log_prob).mean()

        total_loss = value_loss + actor_loss

        return total_loss, {
            'actor_loss': actor_loss,
            'value_loss': value_loss,
            'total_loss': total_loss,
            'adv_mean': adv.mean(),
            'v_mean': v.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }

    @staticmethod
    def sample_actions(network, observations, config, seed, temperature=1.0):
        dist = network.select('actor')(observations, temperature=temperature)
        actions = dist.sample(seed=seed)
        return jnp.clip(actions, -1, 1)

    @staticmethod
    def get_target_update_modules():
        return []

    @staticmethod
    def needs_next_actions():
        return False

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 395,
        "end_line": 484,
        "content": _IQL_POLICY_TRAINER,
    },
]
