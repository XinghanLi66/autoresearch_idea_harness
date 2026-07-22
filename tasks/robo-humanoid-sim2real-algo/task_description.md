# Humanoid Robot Sim2Real: Algorithm Design

## Objective
Design novel reinforcement learning algorithms for humanoid robot locomotion that achieve robust sim-to-real transfer. You will implement custom algorithm components in the PPO (Proximal Policy Optimization) framework that enable policies to follow diverse velocity commands with natural, stable gaits.

## Research Question
**What algorithm implementations (network architecture, policy optimization, experience replay) lead to policies that can successfully execute diverse locomotion commands in sim2sim transfer (Isaac Gym → MuJoCo)?**

The key challenge: standard PPO implementations often struggle with diverse command following and sim-to-real transfer. Your algorithm modifications should improve:
- Policy robustness across varied commands (different speeds, directions, turning rates)
- Sample efficiency during training
- Generalization from simulation to simulation (Isaac Gym → MuJoCo)
- Natural, energy-efficient gaits
- Stable transitions between different commands

## Background
The humanoid locomotion task requires the robot to track 3D velocity commands:
- `vx`: forward/backward velocity (m/s)
- `vy`: lateral velocity (m/s)
- `dyaw`: yaw angular velocity (rad/s)

**Standard PPO algorithm** consists of three main components:
1. **Actor-Critic Network** (`actor_critic.py`): Neural network architecture with separate actor (policy) and critic (value function) heads
2. **PPO Optimizer** (`ppo.py`): Policy optimization using clipped surrogate objective, value function loss, and entropy regularization
3. **Rollout Storage** (`rollout_storage.py`): Experience buffer for collecting and processing trajectories

**The problem**: Standard implementations often:
- Struggle with diverse command distributions
- Have poor sample efficiency on complex locomotion tasks
- Fail to transfer between simulators (Isaac Gym → MuJoCo)
- Produce unnatural or unstable gaits
- Require extensive hyperparameter tuning

## Task
Implement custom algorithm components in the **EDITABLE SECTIONS**:

**1. Actor-Critic Network: `actor_critic_custom.py` (lines 36-128)**

**In `ActorCritic.__init__` method**:
- Design custom network architecture (layer sizes, activation functions)
- Add normalization layers (LayerNorm, BatchNorm, custom)
- Implement custom initialization schemes
- Add auxiliary heads or features

**In `ActorCritic.act` method**:
- Modify action sampling strategy
- Add custom exploration mechanisms
- Implement action post-processing

**In `ActorCritic.evaluate_actions` method**:
- Customize value function computation
- Modify action log probability calculation
- Add auxiliary losses or regularization

**2. PPO Optimizer: `ppo_custom.py` (lines 39-185)**

**In `PPO.__init__` method**:
- Configure optimizer settings
- Set up learning rate schedules
- Initialize custom training components

**In `PPO.update` method**:
- Modify policy loss computation (clipping strategy, advantage normalization)
- Customize value function loss (Huber loss, clipping, multi-step returns)
- Adjust entropy regularization
- Implement custom gradient clipping or normalization
- Add auxiliary losses (e.g., behavioral cloning, imitation)

**3. Rollout Storage: `rollout_storage_custom.py` (lines 32-182)**

**In `RolloutStorage.__init__` method**:
- Design custom buffer structure
- Add additional tracking tensors

**In `RolloutStorage.add_transitions` method**:
- Customize how experiences are stored
- Add data augmentation or preprocessing

**In `RolloutStorage.compute_returns` method**:
- Modify advantage estimation (GAE parameters, normalization)
- Implement custom return computation (n-step, λ-returns)
- Add reward shaping or preprocessing

**4. Training command distribution: `humanoid_config_custom.py` (lines 17-20)**

The training command ranges are **editable**, but the default values mirror the
official XBot recipe: `vx ∈ [-0.3, 0.6]`, `vy ∈ [-0.3, 0.3]`,
`dyaw ∈ [-0.3, 0.3]`. Keep these defaults for paper-aligned comparisons; widen
them only as an explicit algorithmic choice.

**5. PPO hyperparameters: `humanoid_config_custom.py` (lines 29-34)**

PPO algorithm hyperparameters (`learning_rate`, `entropy_coef`,
`num_learning_epochs`, `gamma`, `lam`, `num_mini_batches`) are **editable** per
baseline so different algorithm variants (e.g., adaptive-KL, layernorm) can use
their own training recipe without editing the algorithm code. Architecture and
infrastructure constants (`num_envs`, `max_iterations`, `num_steps_per_env`,
network sizes) remain fixed for fair comparison.

**Fixed components**:
- Environment and reward functions
- Training: 4096 parallel environments, official XBot iteration budget
- Observation/action spaces

## Evaluation
Trained in Isaac Gym (4096 parallel envs, official XBot iteration budget), then evaluated on **100 diverse random commands in MuJoCo (sim2sim transfer)**:

**Test procedure**:
1. Sample 100 random commands from ranges:
   - `vx`: [-0.5, 1.0] m/s
   - `vy`: [-0.4, 0.4] m/s
   - `dyaw`: [-0.5, 0.5] rad/s
2. For each command, run 10-second episode in MuJoCo
3. **Success criteria** (per command):
   - Robot doesn't fall (base height > 0.3m, |roll|, |pitch| < 0.5 rad)
   - Average velocity tracking error < 0.5 (linear-norm + |yaw| error)

**Metrics**:
- `success_rate`: Fraction of commands meeting both criteria above (primary metric)
- `avg_vel_error`: Average velocity tracking error across all 100 commands
- `fall_rate`: Fraction of commands where the robot fell during the episode

**Performance targets**: Good algorithms should achieve:
- High success rate (>70%) across diverse commands
- Low tracking error (<0.4 combined)
- Low fall rate (<20%)
- Robust sim2sim transfer (Isaac Gym → MuJoCo)

## Reference Implementation
One baseline is provided:

**default**: Standard PPO implementation from humanoid-gym
- 3-layer MLP with [512, 256, 128] hidden units
- Standard PPO loss with clipping (ε=0.2)
- Actor [512, 256, 128], critic [768, 256, 128]
- GAE with λ=0.9, γ=0.994
- Baseline for comparison

## Hints
- **Network architecture**: Deeper networks, normalization layers, or residual connections may improve learning
- **Advantage normalization**: Proper normalization can stabilize training
- **Value function**: Accurate value estimates improve policy learning
- **Exploration**: Entropy regularization or noise injection can help exploration
- **Sample efficiency**: Better advantage estimation or multi-step returns can improve efficiency
- **Sim2sim transfer**: Algorithms that learn robust features transfer better
- Consider:
  - Adaptive learning rates or schedules
  - Custom loss weightings (policy vs value vs entropy)
  - Gradient clipping strategies
  - Observation/action normalization
  - Auxiliary tasks or losses
