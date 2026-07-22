# Robo-Diffusion Tasks Plan

## Overview
Based on CleanDiffuser repository (https://github.com/CleanDiffuserTeam/CleanDiffuser), create MLS-Bench tasks for diffusion models in robot decision-making.

## Design Principles
- **Atomic**: Each task focuses on ONE research question
- **Generalizable**: Tested on multiple D4RL environments
- **Rigorous codebase**: Use template + edit_ops pattern (rigorous_codebase=true)
- **Clear scope**: Fixed vs editable components clearly defined
- **Multiple baselines**: 3 baselines per task, ALL from CleanDiffuser repo

## Task List

### Task 1: robo-diffusion-network-backbone
**Network Architecture Design**

**Research Question**: What neural network architecture better models the diffusion process for robot action sequences?

**Editable**:
- Network architecture choice and design
- Layer depth, width, activation functions
- Normalization methods
- Time embedding methods
- Condition injection methods

**Fixed**:
- Diffusion model: DDPM
- Training hyperparameters
- Sampling method
- Task environments

**Baselines (from CleanDiffuser)**:
1. **Pearce_MLP**: Simple MLP (Imitating Human Behaviour with Diffusion Models)
2. **Chi_UNet1d**: 1D UNet (Diffusion Policy)
3. **DiT1d**: Diffusion Transformer (AlignDiff)

**Evaluation**: D4RL (hopper-medium-v2, walker2d-medium-v2, halfcheetah-medium-v2)

---

### Task 2: robo-diffusion-guidance
**Guided Sampling Strategy Design**

**Research Question**: How to better guide diffusion models to generate action sequences satisfying specific conditions (e.g., target states, rewards)?

**Editable**:
- Classifier-free guidance weights and strategies
- Classifier guidance design
- Guidance application timestep ranges
- Hybrid guidance strategies

**Fixed**:
- Network architecture
- Diffusion model
- Training method

**Baselines (from CleanDiffuser)**:
1. **no_guidance**: No guidance sampling
2. **cfg**: Classifier-free guidance (standard implementation)
3. **classifier_guidance**: Classifier guidance (standard implementation)

**Evaluation**: D4RL (hopper-medium-v2, walker2d-medium-v2, halfcheetah-medium-v2)

---

### Task 3: robo-diffusion-sampling-method
**Sampling Algorithm Design**

**Research Question**: What sampling algorithm achieves better efficiency while maintaining generation quality?

**Evaluation Dimensions**:
- Sampling steps (inference speed)
- Success rate (task performance)

**Editable**:
- Sampling algorithm type
- Number of sampling steps
- Sampling strategy parameters

**Fixed**:
- Network architecture
- Diffusion model training
- Task environments

**Baselines (from CleanDiffuser)**:
1. **DDPM**: DDPM sampling (standard implementation)
2. **DDIM**: DDIM sampling (standard implementation)
3. **DPM-Solver**: DPM-Solver++ (standard implementation)

**Evaluation**: D4RL (hopper-medium-v2, walker2d-medium-v2, halfcheetah-medium-v2)

---

### Task 4: robo-diffusion-planner
**Planning Algorithm Design**

**Research Question**: What planning algorithm better performs long-horizon trajectory planning?

**Editable**:
- Planning algorithm core logic
- Trajectory optimization strategy
- Horizon and planning frequency
- Return conditioning method

**Fixed**:
- Network architecture (standard architecture for planners)
- Diffusion model
- Evaluation environments

**Baselines (from CleanDiffuser)**:
1. **Diffuser**: Planning with Diffusion for Flexible Behavior Synthesis
2. **Decision_Diffuser**: Is Conditional Generative Modeling all you need for Decision-Making?
3. **AdaptDiffuser** or **DiffuserLite**: More efficient planning variants

**Evaluation**: D4RL (hopper-medium/medium-replay, walker2d-medium/medium-replay, halfcheetah-medium/medium-replay)

---

### Task 5: robo-diffusion-policy
**Policy Algorithm Design**

**Research Question**: What policy algorithm better performs online decision-making and action generation?

**Editable**:
- Policy algorithm core logic
- Q-function design (if used)
- Action generation strategy
- Training objective

**Fixed**:
- Network architecture (standard architecture for policies)
- Diffusion model
- Evaluation environments

**Baselines (from CleanDiffuser)**:
1. **DQL**: Diffusion Policies as an Expressive Policy Class for Offline RL
2. **IDQL**: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies
3. **Diffusion_Policy** or **SfBC**: Visuomotor Policy Learning / High-Fidelity Generative Behavior Modeling

**Evaluation**: D4RL (hopper-medium/medium-expert, walker2d-medium/medium-expert, halfcheetah-medium/medium-expert)

---

## Planner vs Policy Distinction

**Planner (Trajectory Planning)**:
- Focus on long-horizon trajectory planning
- Generate complete trajectories (state-action sequences)
- Suitable for offline RL, emphasize return conditioning
- Examples: Diffuser, Decision Diffuser

**Policy (Action Generation)**:
- Focus on single-step or short-horizon action generation
- State → action mapping
- May combine Q-learning or actor-critic
- Examples: DQL, IDQL, Diffusion Policy

---

## Task Comparison

| Task | Research Dimension | Baseline Source | Evaluation Focus |
|------|-------------------|----------------|------------------|
| **network-backbone** | Network architecture | Pearce_MLP, Chi_UNet1d, DiT1d | Expressiveness |
| **guidance** | Conditional generation | no_guidance, CFG, classifier_guidance | Conditional control |
| **sampling-method** | Sampling efficiency | DDPM, DDIM, DPM-Solver | Speed vs quality |
| **planner** | Planning algorithm | Diffuser, Decision_Diffuser, AdaptDiffuser | Long-horizon planning |
| **policy** | Policy algorithm | DQL, IDQL, Diffusion_Policy | Online decision-making |

---

## Implementation Priority

**Phase 1 (Core)**:
1. ✅ robo-diffusion-network-backbone - Most fundamental, highest impact
2. robo-diffusion-planner - Core planning algorithm

**Phase 2 (Important)**:
3. robo-diffusion-policy - Core policy algorithm
4. robo-diffusion-guidance - Conditional generation

**Phase 3 (Optimization)**:
5. robo-diffusion-sampling-method - Efficiency optimization

---

## Next Steps

- [x] Create planning branch
- [x] Write plan document
- [ ] Clone CleanDiffuser to vendor/external_packages/
- [ ] Analyze CleanDiffuser code structure
- [ ] Create vendor/pkg_configs/cleandiffuser/config.json
- [ ] Write pre_edit.py for CleanDiffuser
- [ ] Create first task branch: robo-diffusion-network-backbone
- [ ] Implement first task

---

## Notes

- All baselines MUST come from CleanDiffuser repo implementations
- Use rigorous_codebase=true for all tasks
- Ensure flexibility in task design
- Follow MLS-Bench conventions (config.json, parser.py, task_description.md, etc.)
