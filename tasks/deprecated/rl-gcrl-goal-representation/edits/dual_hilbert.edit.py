"""Dual Hilbert (symmetric) representation baseline.

Implements the Hilbert/symmetric dual goal representation from the paper.
Uses a shared phi network for both states and goals. V(s,g) = -||phi(s) - phi(g)||.
Trained via IQL-style expectile regression with a separate MLP critic.
"""

_FILE = "dual-goal-representations/custom_train.py"

_HILBERT_REP = '''\
class GoalRepresentation(nn.Module):
    """Hilbert (symmetric) dual goal representation.

    Uses a shared phi network for both state and goal encoding.
    V(s,g) = -||phi(s) - phi(g)||_2, so goals that are temporally
    close to many states get similar representations.

    Goal encoding for downstream use is phi(g), averaged across ensemble.
    Trained via IQL expectile regression with a separate MLP critic.
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        # Shared representation network phi for both states and goals.
        mlp_module = ensemblize(MLP, 2)
        self.phi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        # Critic for rep learning: Q(s, g, a)
        critic_module = ensemblize(MLP, 2)
        self.rep_critic = critic_module((*self.hidden_dims, 1),
                                        activate_final=False, layer_norm=self.layer_norm)

    def _hilbert_value(self, observations, goals):
        """Compute Hilbert value: -||phi(s) - phi(g)||."""
        phi_s = self.phi(observations)
        phi_g = self.phi(goals)
        squared_dist = jnp.square(phi_s - phi_g).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
        return v

    def encode_goal(self, goals):
        """Encode goals using phi network, averaged across ensemble."""
        phi_g = self.phi(goals)  # (2, batch, rep_dim)
        return phi_g.mean(axis=0)  # (batch, rep_dim)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        """IQL-style representation learning loss for Hilbert rep."""
        expectile = 0.7

        # Compute Q from rep_critic.
        critic_input = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input = jnp.concatenate([critic_input, actions], axis=-1)
        q1_t, q2_t = self.rep_critic(critic_input)
        q1_t = q1_t.squeeze(-1)
        q2_t = q2_t.squeeze(-1)
        q_t = jnp.minimum(q1_t, q2_t)

        # Rep value = Hilbert V(s, g).
        v = self._hilbert_value(observations, goals)
        v_mean = v.mean(axis=0)

        # Expectile loss for representation value.
        adv = q_t - v_mean
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        value_loss = (weight * (adv ** 2)).mean()

        # Critic loss: TD target from rep value at next state.
        next_v = self._hilbert_value(next_observations, goals)
        next_v_mean = next_v.mean(axis=0)
        td_target = rewards + 0.99 * masks * next_v_mean

        q1, q2 = self.rep_critic(critic_input)
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)
        critic_loss = ((q1 - td_target) ** 2 + (q2 - td_target) ** 2).mean()

        total_loss = value_loss + critic_loss
        info = {
            'rep_value_loss': value_loss,
            'rep_critic_loss': critic_loss,
            'rep_v_mean': v_mean.mean(),
        }
        return total_loss, info

    def __call__(self, goals, observations=None, next_observations=None,
                 rewards=None, masks=None, actions=None, mode='encode'):
        if mode == 'rep_loss':
            return self.compute_rep_loss(
                observations, goals, next_observations,
                rewards, masks, actions)
        return self.encode_goal(goals)
'''

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 41,
        "end_line": 120,
        "content": _HILBERT_REP,
    },
]
