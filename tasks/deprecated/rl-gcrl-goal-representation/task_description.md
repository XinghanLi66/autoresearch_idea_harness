# Goal Representation for Goal-Conditioned RL

## Objective
Design a novel goal representation module for goal-conditioned reinforcement learning that improves over using raw state observations as goals. The representation should encode goals into a learned vector space before they are passed to the actor and value function networks.

## Background
In goal-conditioned RL, an agent learns to reach arbitrary goal states from a dataset of offline trajectories. Standard approaches simply concatenate raw goal observations with state observations as input to actor and value networks. However, raw goal observations may contain exogenous noise and irrelevant features that hinder learning.

The `GoalRepresentation` module you must implement sits between raw goal observations and the downstream GCIVL (Goal-Conditioned Implicit V-Learning) agent. It has three key methods:
- `setup()`: Initialize any neural network layers or parameters
- `encode_goal(goals)`: Map raw goal observations to a learned representation vector
- `compute_rep_loss(observations, goals, next_observations, rewards, masks, actions)`: Compute an auxiliary training loss for learning the representation (return 0.0 if no auxiliary loss is needed)

The encoded goals replace raw goals everywhere: in the value function V(s, phi(g)), the actor pi(a|s, phi(g)), and the target networks. The auxiliary rep loss is added to the total training objective.

## Evaluation
Success rate averaged across multiple goal-reaching tasks in each environment. Higher is better. The agent is evaluated on three OGBench environments:
- **antmaze-large-navigate-v0**: Ant robot navigating a large maze to reach goal positions
- **cube-single-noisy-v0**: Robotic manipulation of a single cube to reach goal configurations (noisy dataset)
- **pointmaze-large-navigate-v0**: Point robot navigating a large maze

## Available Utilities
You can import from the codebase:
- `MLP`, `ensemblize` from `utils.networks` — standard MLP and ensemble wrappers
- `GCBilinearValue` from `utils.networks` — bilinear inner product value parameterization
- `jax`, `jax.numpy`, `flax.linen` — JAX/Flax neural network primitives
- All standard JAX operations (gradients, vmap, etc.)

## Hints
- Consider what information about a goal is actually useful for reaching it. Raw state features may contain irrelevant dimensions.
- Temporal distance or reachability structure between states and goals could be a useful learning signal.
- Both symmetric (phi(s), phi(g) share parameters) and asymmetric (separate phi, psi networks) representations are valid approaches.
- The auxiliary loss should encourage the representation to capture goal-relevant structure from the offline dataset.
- Balance the representation loss magnitude with the main GCIVL losses.
