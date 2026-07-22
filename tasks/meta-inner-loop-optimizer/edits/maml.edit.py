"""MAML baseline — rigorous codebase edit ops.

Vanilla MAML (Finn et al., 2017): fixed learning rate SGD applied to
all model parameters in the inner loop.

Reference: learn2learn/algorithms/maml.py
Paper: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
       (Finn, Abbeel, Levine, ICML 2017)

Reported accuracy (ConvNet-4, learn2learn benchmarks):
  miniImageNet 5-way 1-shot: ~46-48%
  miniImageNet 5-way 5-shot: ~63%
  CIFAR-FS 5-way 5-shot: ~65%
"""

_FILE = "learn2learn/custom_maml.py"

_MAML = """\
class InnerLoopOptimizer:
    \"\"\"MAML inner-loop optimizer (Finn et al., 2017).

    Vanilla SGD with a fixed learning rate applied uniformly to all
    model parameters. This is the standard MAML inner loop.
    \"\"\"

    def __init__(self, model: nn.Module, inner_lr: float = INNER_LR):
        self.inner_lr = inner_lr

    def adapt(self, model: nn.Module, support_x: Tensor, support_y: Tensor,
              n_steps: int) -> nn.Module:
        model.train()
        for _ in range(n_steps):
            loss = F.cross_entropy(model(support_x), support_y)
            grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            model = l2l.algorithms.maml.maml_update(
                model, lr=self.inner_lr, grads=grads
            )
        return model

    def meta_parameters(self) -> List[Tensor]:
        return []

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 177,
        "end_line": 254,
        "content": _MAML,
    },
]
