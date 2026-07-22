"""ER (Experience Replay) baseline — rigorous codebase edit ops.

Replaces the Custom class (lines 37-65) with an Experience Replay
implementation that stores data from previous tasks and replays it
alongside current task data during training.

Line numbers reference the post-mid_edit file (custom.py created from template).
"""

_FILE = "LIBERO/libero/lifelong/algos/custom.py"

_ER_IMPL = """\

class Custom(Sequential):
    \"\"\"ER (Experience Replay) lifelong learning algorithm.\"\"\"

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.n_memories = 1000
        self.datasets = []
        self.buffer = None

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
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
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
        "content": _ER_IMPL,
    },
]
