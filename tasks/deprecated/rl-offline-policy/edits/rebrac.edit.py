"""ReBRAC baseline -- Revisited Behavior-Regularized Actor-Critic (Tarasov et al., 2024).

Policy approach: Deterministic policy (TD3-style) with BC regularization.
Actor loss = -Q(s, pi(s)) / |Q| + alpha * ||pi(s) - a||^2.
Also adds smoothed noise to target policy for critic updates.
"""

_FILE = "fql/custom_train.py"

_REBRAC_POLICY_TRAINER = """\
class PolicyTrainer:
    \"\"\"ReBRAC: TD3+BC with layer normalization and behavior regularization.

    Uses a deterministic Gaussian actor with exploration noise.
    Actor loss balances Q-value maximization with BC regularization.
    \"\"\"

    @staticmethod
    def create_policy_networks(obs_dim, action_dim, hidden_dims, layer_norm):
        ex_obs = jnp.zeros((1, obs_dim))

        actor_def = GaussianActor(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            layer_norm=layer_norm,
            tanh_squash=True,
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=0.01,
        )

        return {
            'actor': (actor_def, (ex_obs,)),
        }

    @staticmethod
    def compute_actor_loss(network, batch, config, rng):
        dist = network.select('actor')(batch['observations'])
        actions = dist.mode()

        # Q loss.
        qs = network.select('critic')(batch['observations'], actions=actions)
        q = jnp.min(qs, axis=0)

        # Normalize Q values by the absolute mean.
        lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
        actor_loss = -(lam * q).mean()

        # BC loss.
        alpha_actor = config.get('alpha', 10.0) * 0.001  # Scale alpha for ReBRAC-style
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)
        bc_loss = (alpha_actor * mse).mean()

        total_loss = actor_loss + bc_loss

        return total_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'total_loss': total_loss,
            'mse': mse.mean(),
        }

    @staticmethod
    def sample_actions(network, observations, config, seed, temperature=1.0):
        dist = network.select('actor')(observations, temperature=temperature)
        actions = dist.mode()
        actor_noise = 0.2
        actor_noise_clip = 0.5
        noise = jnp.clip(
            jax.random.normal(seed, actions.shape) * actor_noise * temperature,
            -actor_noise_clip,
            actor_noise_clip,
        )
        actions = jnp.clip(actions + noise, -1, 1)
        return actions

    @staticmethod
    def get_target_update_modules():
        return ['actor']

    @staticmethod
    def needs_next_actions():
        return True

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 395,
        "end_line": 484,
        "content": _REBRAC_POLICY_TRAINER,
    },
]
