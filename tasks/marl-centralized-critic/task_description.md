# Cooperative MARL: Centralized Critic Architecture for MAPPO

## Objective
Improve cooperative multi-agent reinforcement learning by designing a better **centralized critic architecture** for MAPPO (Multi-Agent PPO). You can modify the `CustomCritic` class (lines 13-69) and add custom imports (lines 7-8) in `custom_critic.py`.

## Background
In cooperative MARL with partial observability, each agent only sees a local observation but the team shares a common reward. Centralized-Training-with-Decentralized-Execution (CTDE) methods train a centralized value function during training (which can see the global state and all agents' information) and use it to reduce variance when computing advantages for each agent's decentralized policy gradient update. The architecture of this centralized critic — what it conditions on and how it mixes per-agent features — directly determines how tight the bias-variance tradeoff is and therefore how well MAPPO scales to hard multi-agent cooperation tasks.

The training uses EPyMARL's `ppo_learner` with the MAPPO default hyperparameters on three SMAC maps via **smaclite** (a pure-Python reimplementation of the StarCraft Multi-Agent Challenge benchmark — no StarCraft II binary required):

- **mmm** — 1 Medivac + 2 Marauders + 7 Marines (team of 10) vs mirror; heterogeneous cooperation requiring heal micro (≈5M env steps).
- **2s3z** — 2 Stalkers + 3 Zealots (team of 5) vs mirror; medium heterogeneous team (≈5M env steps).
- **3s5z** — 3 Stalkers + 5 Zealots (team of 8) vs mirror; larger team, harder (≈5M env steps).

The default critic is a simple `(state ⊕ agent-one-hot) → 3-layer MLP → V` that ignores per-agent observations. It is a working baseline but leaves room for smarter architectures that integrate per-agent features, attention, or state conditioning.

## Interface
Your `CustomCritic` must:
- Inherit from `nn.Module`.
- Accept `(scheme, args)` in `__init__`, where:
  - `scheme["state"]["vshape"]` — global state dim
  - `scheme["obs"]["vshape"]` — per-agent observation dim
  - `args.n_agents`, `args.n_actions`, `args.hidden_dim`, `args.obs_last_action`, `args.obs_individual_obs`
- Set `self.output_type = "v"` in `__init__`.
- Implement `forward(self, batch, t=None)` where:
  - `batch["state"]` — shape `(B, T, state_dim)`
  - `batch["obs"]` — shape `(B, T, n_agents, obs_dim)`
  - `batch.batch_size`, `batch.max_seq_length`, `batch.device`
  - `t=None` means "whole sequence"; otherwise `t` is an integer
  - **Returns** `q` with shape `(B, T, n_agents, 1)` — the learner later does `.squeeze(3)`, so the trailing singleton is mandatory.

## Reference Implementations
- **IPPO critic** (`ippo_critic.edit.py`): per-agent MLP over `batch["obs"]` ⊕ agent-one-hot; no centralization. Floor baseline from Yu et al. 2022's IPPO ablation. Also see `epymarl/src/modules/critics/ac.py`.
- **MAPPO critic** (`mappo_critic.edit.py`): shared MLP over `(batch["state"] ⊕ agent-one-hot)`. Standard MAPPO central V from Yu et al. 2022. Also see `epymarl/src/modules/critics/centralV.py`.
- **MAT-style attention critic** (`mat_critic.edit.py`): projects per-agent features (obs ⊕ state broadcast) into tokens, then a single `TransformerEncoder` layer with self-attention across the agent axis produces a per-agent value. Adapted from Wen et al. 2022 "Multi-Agent Transformer" (arXiv 2205.14953) — **critic-only** form; the MAPPO actor is kept unchanged.

## Evaluation
Final performance is measured by **test win rate** (`battle_won_mean`) averaged over 32 test episodes with the greedy policy, evaluated separately on all three SMAC maps and recorded to the leaderboard under setup-specific metric keys:

- Primary: `test_battle_won_mean_<map>`
- Secondary: `test_return_mean_<map>`

Higher is better. A strong centralized critic should generalize across maps of varying difficulty.
