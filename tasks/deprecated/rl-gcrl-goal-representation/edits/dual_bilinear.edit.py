"""Dual bilinear representation baseline.

Implements the bilinear dual goal representation from Park, Mann, Levine (ICLR 2026).
The representation learns phi(s) and psi(g) networks where V(s,g) = phi(s)^T psi(g) / sqrt(d).
The goal encoding uses psi(g). Trained via IQL-style expectile regression with a
separate rep_critic for TD-based learning of the representation value function.
"""

_FILE = "dual-goal-representations/custom_train.py"

# Replace the GoalRepresentation class with a bilinear dual representation.
_BILINEAR_REP = '''\
class GoalRepresentation(nn.Module):
    """Bilinear dual goal representation.

    Learns separate state (phi) and goal (psi) representations where
    V(s,g) = phi(s)^T psi(g) / sqrt(d). The goal encoding for downstream
    use is psi(g), averaged across ensemble members.

    Also maintains a separate MLP critic for learning the representation
    via IQL expectile regression.
    """

    obs_dim: int
    rep_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512)
    layer_norm: bool = True

    def setup(self):
        # Bilinear value: phi(s)^T psi(g) / sqrt(d)
        mlp_module = ensemblize(MLP, 2)
        self.phi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.rep_dim),
                              activate_final=False, layer_norm=self.layer_norm)
        # Critic for rep learning: Q(s, g, a)
        critic_module = ensemblize(MLP, 2)
        self.rep_critic = critic_module((*self.hidden_dims, 1),
                                        activate_final=False, layer_norm=self.layer_norm)
        # Target critic parameters are managed externally via target_update.

    def _bilinear_value(self, observations, goals):
        """Compute bilinear value: phi(s)^T psi(g) / sqrt(d)."""
        phi_s = self.phi(observations)
        psi_g = self.psi(goals)
        v = (phi_s * psi_g / jnp.sqrt(self.rep_dim)).sum(axis=-1)
        return v

    def encode_goal(self, goals):
        """Encode goals using psi network, averaged across ensemble."""
        psi_g = self.psi(goals)  # (2, batch, rep_dim)
        return psi_g.mean(axis=0)  # (batch, rep_dim)

    def compute_rep_loss(self, observations, goals, next_observations,
                         rewards, masks, actions=None):
        """IQL-style representation learning loss.

        Trains the bilinear value as a representation value function and
        the MLP critic as a TD-based critic, using expectile regression.
        """
        expectile = 0.7  # rep_expectile from the paper

        # Compute Q from rep_critic (stop grad on targets).
        critic_input_obs = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs = jnp.concatenate([critic_input_obs, actions], axis=-1)
        q1_t, q2_t = self.rep_critic(critic_input_obs)
        q1_t = q1_t.squeeze(-1)
        q2_t = q2_t.squeeze(-1)
        q_t = jnp.minimum(q1_t, q2_t)

        # Rep value = bilinear V(s, g)
        v = self._bilinear_value(observations, goals)
        v_mean = v.mean(axis=0)  # average over ensemble

        # Expectile loss for representation value.
        adv = q_t - v_mean
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        value_loss = (weight * (adv ** 2)).mean()

        # Critic loss: TD target from rep value at next state.
        next_v = self._bilinear_value(next_observations, goals)
        next_v_mean = next_v.mean(axis=0)
        td_target = rewards + 0.99 * masks * next_v_mean

        critic_input_obs_cur = jnp.concatenate([observations, goals], axis=-1)
        if actions is not None:
            critic_input_obs_cur = jnp.concatenate([critic_input_obs_cur, actions], axis=-1)
        q1, q2 = self.rep_critic(critic_input_obs_cur)
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
        "content": _BILINEAR_REP,
    },
]
