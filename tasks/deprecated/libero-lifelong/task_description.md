# LIBERO Lifelong Learning: Continual Robot Task Learning

## Objective
Design a better lifelong/continual learning algorithm that reduces catastrophic
forgetting when learning 10 sequential robot manipulation tasks from the
LIBERO-Spatial benchmark.

## Background
The LIBERO-Spatial benchmark consists of 10 pick-and-place tasks that share the
same action (pick up a black bowl and place it on a plate) but differ in spatial
configurations. Tasks are learned sequentially: the agent trains on task 0,
then task 1, etc. After training on all 10 tasks, the agent is evaluated on
every task. The key challenge is **catastrophic forgetting** — performance on
earlier tasks degrades as new tasks are learned.

The default `Sequential` baseline simply fine-tunes on each new task without any
forgetting mitigation. Training uses behavior cloning (BC) with a transformer
policy, AdamW optimizer, cosine annealing LR, for 50 epochs per task.

## Your Task
Modify the `Custom` class in `custom.py` (lines 37-65) to implement a lifelong
learning strategy. You can override any of these methods:

- `__init__(self, n_tasks, cfg, **policy_kwargs)`: Initialize algorithm state
  (buffers, regularizers, etc.)
- `start_task(self, task)`: Called before training on each new task. Use to set
  up task-specific state (e.g., replay buffers).
- `observe(self, data)`: Called each training step. Compute loss, do backprop,
  update weights. Must return the loss value (float).
- `end_task(self, dataset, task_id, benchmark, env=None)`: Called after training
  on each task. Use for post-processing (compute Fisher information, store replay
  data, etc.)

## Available Utilities
- `self.policy`: The BC transformer policy. Use `self.policy.compute_loss(data)`
  for the behavior cloning loss.
- `self.optimizer`: AdamW optimizer (initialized in parent `start_task`).
- `self.scheduler`: Cosine annealing LR scheduler.
- `self.map_tensor_to_device(data)`: Move data dict to GPU.
- `self.loss_scale`, `self.cfg`, `self.current_task`, `self.n_tasks`: Config and state.
- `cycle(dl)`: Infinite iterator over a DataLoader.
- `merge_datas(x, y)`: Concatenate two nested data dicts.
- `TruncatedSequenceDataset(dataset, n)`: Truncate a dataset to `n` sequences.
- `DataLoader`, `ConcatDataset`, `RandomSampler`: PyTorch data utilities.
- `TensorUtils.map_tensor(data, fn)`: Apply function to all tensors in nested dict.
- `safe_device(tensor, device)`: Move tensor to device safely.

## Metric
- **avg_final_success** (higher is better): Average success rate across all 10
  tasks, evaluated after training on the final task.

