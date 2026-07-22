# Offline RL: Partially Observable MDP on NetHack (POMDP)


## Objective
Design and implement an offline RL algorithm that handles partial observability using recurrent memory. Your code goes in the `Model` and `OfflineAlgorithm` classes in `custom_nethack_pomdp.py`. Three reference implementations (BC, CQL, IQL with chaotic LSTM) are provided as read-only.

## Background
NetHack is a procedurally generated roguelike with 121 discrete actions and terminal-based (80x24 character grid) observations. It is inherently a POMDP — the agent cannot see the full map, hidden enemies, or trap states from the current observation alone, requiring temporal memory.

## Constraints
- The `ModelBackbone` (observation encoders + LSTM) is FIXED and instantiated by the training loop. Your `Model.__init__` receives it as the `backbone` argument and must store it as `self.backbone`. Do not replace or modify the backbone.
- The LSTM dimensions (rnn_hidden_dim=2048, rnn_layers=2) are fixed by the `TrainConfig` and cannot be overridden.
- Output head parameters (everything beyond the backbone) must not exceed 2,000,000. This is enforced at runtime.
- Focus on algorithmic innovation (loss functions, training procedures, output head design) rather than scaling up network capacity.
- Do NOT simply copy a reference implementation with minor changes

## Evaluation
Trained and evaluated on Monk, Valkyrie, Ranger. Additional held-out NetHack characters (not shown during intermediate testing) are used to assess generalization. Metric: normalized score (your score / AutoAscend bot mean score).
