"""FQL baseline -- Flow Q-Learning (Park, Li, Levine, 2025).

Policy approach: Train a flow-matching BC policy (multi-step ODE) as teacher,
distill it into a one-step policy, then add Q-value guidance.
Actor loss = bc_flow_loss + alpha * distill_loss + q_loss.
"""

_FILE = "fql/custom_train.py"

_FQL_POLICY_TRAINER = """\
class PolicyTrainer:
    \"\"\"FQL: Flow Q-Learning policy trainer.

    Uses flow matching for behavioral cloning, distills into a one-step policy,
    and adds Q-value guidance for improvement beyond the dataset.
    \"\"\"

    @staticmethod
    def create_policy_networks(obs_dim, action_dim, hidden_dims, layer_norm):
        ex_obs = jnp.zeros((1, obs_dim))
        ex_act = jnp.zeros((1, action_dim))
        ex_times = jnp.zeros((1, 1))

        actor_bc_flow_def = VectorFieldActor(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            layer_norm=layer_norm,
        )
        actor_onestep_flow_def = VectorFieldActor(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            layer_norm=layer_norm,
        )

        return {
            'actor_bc_flow': (actor_bc_flow_def, (ex_obs, ex_act, ex_times)),
            'actor_onestep_flow': (actor_onestep_flow_def, (ex_obs, ex_act)),
        }

    @staticmethod
    def compute_actor_loss(network, batch, config, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = network.select('actor_bc_flow')(batch['observations'], x_t, t)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Compute flow actions for distillation (stop gradient on teacher).
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))

        # Euler method for flow ODE (teacher, no gradient).
        flow_steps = 10
        flow_actions = noises
        for i in range(flow_steps):
            ft = jnp.full((*batch['observations'].shape[:-1], 1), i / flow_steps)
            vels = jax.lax.stop_gradient(
                network.select('actor_bc_flow')(batch['observations'], flow_actions, ft)
            )
            flow_actions = flow_actions + vels / flow_steps
        target_flow_actions = jax.lax.stop_gradient(jnp.clip(flow_actions, -1, 1))

        # Distillation loss.
        actor_actions = network.select('actor_onestep_flow')(batch['observations'], noises)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
        qs = network.select('critic')(batch['observations'], actions=actor_actions_clipped)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()
        if config.get('normalize_q_loss', False):
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        alpha = config.get('alpha', 10.0)
        actor_loss = bc_flow_loss + alpha * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
        }

    @staticmethod
    def sample_actions(network, observations, config, seed, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        action_dim = config['action_dim']
        noises = jax.random.normal(
            action_seed,
            (*observations.shape[:-1], action_dim),
        )
        actions = network.select('actor_onestep_flow')(observations, noises)
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
        "content": _FQL_POLICY_TRAINER,
    },
]
