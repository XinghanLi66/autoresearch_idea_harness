# Federated Learning Aggregation Strategy Design

## Research Question
Design a novel server-side aggregation strategy for federated learning that achieves better convergence and higher test accuracy under heterogeneous (non-IID) data distributions across clients.

## Background
Federated Learning (FL) trains a shared global model across many clients without centralizing data. The canonical algorithm, FedAvg, simply averages client model parameters weighted by sample count. However, when client data distributions are heterogeneous (non-IID), FedAvg suffers from "client drift" where local updates diverge, leading to slow convergence or poor final accuracy. Research has produced several improvements: FedProx adds a proximal penalty to local objectives, SCAFFOLD uses control variates for variance reduction, and methods like FedNova normalize updates by local steps. The aggregation strategy — how the server combines client updates into the global model — is the core algorithmic component that determines convergence behavior.

## Task
Modify the `ServerAggregator` class in `custom_fl_aggregation.py`. You must implement the `aggregate()` method that takes the current global model state, a list of client updates (model parameters + metadata), and returns the new global model state. You may also customize client selection via `select_clients()`.

## Interface
```python
class ServerAggregator:
    def __init__(self, global_model, args):
        # Initialize aggregation state (momentum buffers, control variates, etc.)

    def aggregate(self, global_state_dict, client_updates, round_num):
        # global_state_dict: OrderedDict of current global model parameters
        # client_updates: list of (state_dict, num_samples, avg_loss) tuples
        # round_num: current communication round (0-indexed)
        # Returns: OrderedDict of updated global model parameters

    def select_clients(self, num_available, num_to_select, round_num):
        # Returns: list of client indices to participate this round
```

## Evaluation
The aggregation strategy is evaluated on three benchmarks with non-IID data:
1. **CIFAR-10** with Dirichlet split (alpha=0.1) — 100 clients, image classification
2. **FEMNIST** (EMNIST ByClass) with Dirichlet split — 100 clients, character recognition
3. **Shakespeare** (next character prediction) — naturally non-IID by speaker

Metric: **test accuracy** after 200 communication rounds (higher is better). Each round, 10 of 100 clients are selected, each trains for 5 local epochs with SGD (lr=0.01).

