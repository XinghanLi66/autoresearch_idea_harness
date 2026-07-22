# Cooperative MARL: Value Decomposition Mixing Network

## Objective
Improve cooperative multi-agent reinforcement learning by designing a better value decomposition mixing network. You can modify the `CustomMixer` class (lines 13-49) and add custom imports (lines 7-8) in `custom.py`.

## Background
In cooperative MARL, agents share a common reward but each agent has only a partial observation. Value decomposition methods learn individual agent Q-values and combine them into a joint `Q_tot` using a mixing network. The quality of this mixing network directly determines how well individual agents can coordinate.

The training uses EPyMARL with Q-learning on three super-hard SMAC maps via **smaclite** (a pure-Python reimplementation of the StarCraft Multi-Agent Challenge — no StarCraft II binary required):
- **3s5z_vs_3s6z**: 3 Stalkers + 5 Zealots vs 3 Stalkers + 6 Zealots; asymmetric combat requiring precise coordination.
- **corridor**: 6 Zealots in a narrow corridor vs 24 Zerglings; requires tight spatial coordination.
- **MMM2**: 1 Medivac + 2 Marauders + 7 Marines vs stronger enemy; heterogeneous team requiring heal micro.

The global state contains unit health, positions, and other features not visible in individual observations. The default mixer is a simple learnable weighted sum that does not condition on the global state. Each setup trains for 2M environment timesteps with epsilon-greedy exploration (epsilon annealed over the first 50K steps).

## Interface
Your `CustomMixer` must:
- Inherit from `nn.Module`
- Accept `args` in `__init__` with attributes: `n_agents`, `state_shape`, `mixing_embed_dim`
- Implement `forward(self, agent_qs, states)` where:
  - `agent_qs`: shape `(batch, T, n_agents)` — individual agent Q-values
  - `states`: shape `(batch, T, state_dim)` — global state information
  - Returns `q_tot`: shape `(batch, T, 1)` — joint action value

## Reference Implementations
- **VDN** (`vdn.py`): Simple sum, `Q_tot = sum(Q_i)`. No parameters, no state conditioning.
- **LinearMixer**: Learnable weighted sum with a bias term. State-agnostic but more flexible than VDN.
- **QMIX** (`qmix.py`): Uses hypernetworks conditioned on global state to generate mixing weights. Enforces monotonicity via absolute value on weights.

## Evaluation
Final performance is measured by **test return** and **test win rate** averaged over 32 test episodes with greedy policy, evaluated separately on all three maps:

- Primary: `test_return_mean_<map>`
- Secondary: `test_battle_won_mean_<map>`

Higher is better.
