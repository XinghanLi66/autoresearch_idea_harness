# JEPA World Model Planning: Algorithm Design

## Objective
Design a novel planning algorithm that leverages a learned JEPA (Joint Embedding Predictive Architecture) world model for goal-conditioned navigation. The task evaluates planning performance on the Two Rooms environment, where an agent must navigate around walls and through doors to reach a target location.

## Research Question
Can you design a planning algorithm that outperforms standard derivative-free methods (CEM, MPPI) by better exploiting the structure of a learned JEPA world model?

## What You Can Modify
You must implement the `CustomPlanner` class in `custom_planner.py` within the editable region on lines 323-367. The class extends the `Planner` abstract base class and must implement the `plan()` method. The JEPA world model checkpoint is fixed and provided by the evaluation environment; the task is to improve planning, not to retrain the model.

## Interface

### CustomPlanner Constructor
```python
def __init__(self, unroll, action_dim=2, plan_length=15,
             num_samples=200, n_iters=20, **kwargs):
```
- `unroll`: Function to forward-simulate through the world model
- `action_dim`: Dimensionality of action space (2 for x/y movement)
- `plan_length`: Maximum planning horizon
- `num_samples`: Number of action samples (adjustable)
- `n_iters`: Number of optimization iterations (adjustable)

### plan() Method
```python
def plan(self, obs_init, steps_left=None, eval_mode=True,
         t0=False, plan_vis_path=None) -> PlanningResult:
```
- `obs_init`: Initial observation encoding `[1, C, 1, H, W]`
- `steps_left`: Remaining steps in the episode
- Returns: `PlanningResult(actions=Tensor[T, A], ...)`

### Available Methods (Inherited)
- `self.unroll(obs_init, actions)`: Forward-simulate actions through the world model.
  - `obs_init`: `[1, C, 1, H, W]` initial observation encoding
  - `actions`: `[B, A, T]` batch of action sequences
  - Returns: `[B, D, T+1, H, W]` predicted state encodings
- `self.objective(encodings)`: Compute cost for predicted state encodings.
  - `encodings`: `[B, D, T, H, W]`
  - Returns: `[B]` cost per sample (lower is better)
- `self.cost_function(actions, obs_init)`: Convenience method that calls `unroll` then `objective`.
  - Returns: `[B]` cost per sample

## Evaluation
- Environment: Two Rooms (65x65 grid with wall and door)
- Episodes: 20 with random start and goal positions controlled by the MLS-Bench SEED
- Max steps per episode: 200
- Success threshold: Euclidean distance < 4.5 from goal
- Benchmarks: Three planning horizons (30, 60, 90 steps) test the algorithm across short, medium, and long-range planning
- Metric: `success_rate` (fraction of successful episodes) per horizon
