# Implicit Q-Learning: Value Function Loss Design for Offline Robot Learning

## Research Question
Design an improved value function loss for Implicit Q-Learning (IQL) in offline robot manipulation. IQL avoids querying out-of-distribution actions by learning V(s) via asymmetric regression against Q(s,a) estimates. The loss function determines how V(s) approximates the upper quantile of Q-values, directly affecting policy quality.

## What You Can Modify
The `custom_vf_loss` function (lines 21-39) in `custom_iql_vf.py`. This function computes the loss for training the value network V(s).

Interface:
- **Input**:
  - `vf_pred: [B, 1]` -- predicted state values V(s)
  - `q_target: [B, 1]` -- target Q-values Q(s,a) from the critic ensemble (detached)
  - `quantile: float` -- asymmetry parameter tau (default 0.9)
- **Output**: scalar loss tensor

You may restructure the function body, add helper computations, and use any PyTorch operations.

## Evaluation
- **Metric**: `success_rate` -- rollout success rate on the task (higher is better)
- **Tasks**: Lift, Can, Square (robot manipulation with proficient human demonstrations)
- **Dataset**: ~200 demonstrations with (s, a, r, s', done) transitions
- **Training**: IQL with Q-ensemble (2 critics), GMM actor (5 modes), 2000 epochs x 100 steps
- **Hyperparameters**: discount=0.99, target_tau=0.01, adv_beta=1.0, vf_quantile=0.9
- **Rollout evaluation**: 50 episodes per task, horizon 400 steps, every 50 epochs

## Background
IQL's key insight is learning V(s) without maximizing over actions:
- **Expectile regression** (default): asymmetric L2 loss that pushes V(s) toward the tau-th expectile of Q(s,a). When `tau=0.9`, overestimation (V > Q) is penalized 9x more than underestimation.
- The value function feeds into the actor via advantage weighting: `w(s,a) = exp((Q(s,a) - V(s)) / beta)`, so V(s) quality directly impacts policy learning.
