"""A-GEM (Averaged Gradient Episodic Memory) baseline — rigorous codebase edit ops.

Replaces lines 37-65 with helper functions (project, store_grad, overwrite_grad)
and an A-GEM implementation that combines experience replay with gradient
projection to prevent interference with previous tasks.

Line numbers reference the post-mid_edit file (custom.py created from template).
"""

_FILE = "LIBERO/libero/lifelong/algos/custom.py"

_AGEM_IMPL = """\

def _project(gxy, ger):
    \"\"\"Project gradient gxy so it does not conflict with ger.\"\"\"
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


def _store_grad(params, grads, grad_dims):
    \"\"\"Store parameter gradients into a flat vector.\"\"\"
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def _overwrite_grad(params, newgrad, grad_dims):
    \"\"\"Overwrite parameter gradients from a flat vector.\"\"\"
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[: count + 1])
            this_grad = newgrad[begin:end].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


class Custom(Sequential):
    \"\"\"A-GEM (Averaged Gradient Episodic Memory) lifelong learning algorithm.\"\"\"

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.n_memories = 1000
        self.datasets = []
        self.buffer = None

        self.grad_dims = []
        for pp in self.policy.parameters():
            self.grad_dims.append(pp.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            buffers = [
                TruncatedSequenceDataset(dataset, self.n_memories)
                for dataset in self.datasets
            ]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=True,
                )
            )

    def observe(self, data):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (loss * self.loss_scale).backward()

        if self.buffer is not None:
            _store_grad(self.policy.parameters, self.grad_xy, self.grad_dims)
            buf_data = next(self.buffer)
            self.policy.zero_grad()

            buf_data = self.map_tensor_to_device(buf_data)
            buf_loss = self.policy.compute_loss(buf_data)
            buf_loss.backward()
            _store_grad(self.policy.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = _project(gxy=self.grad_xy, ger=self.grad_er)
                _overwrite_grad(self.policy.parameters, g_tilde, self.grad_dims)
            else:
                _overwrite_grad(self.policy.parameters, self.grad_xy, self.grad_dims)

        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark, env=None):
        self.datasets.append(dataset)
"""

# Ordered bottom-to-top (single op here).
OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 37,
        "end_line": 65,
        "content": _AGEM_IMPL,
    },
]
