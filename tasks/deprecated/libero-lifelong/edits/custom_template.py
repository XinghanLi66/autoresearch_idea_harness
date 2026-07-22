"""Custom lifelong learning algorithm for LIBERO sequential task learning."""

import collections

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *


def cycle(dl):
    """Infinite cycle over a dataloader."""
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    """Recursively merge (concatenate) two nested data structures."""
    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = type(x)()
        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, (torch.FloatTensor, torch.LongTensor)):
        return torch.cat([x, y], 0)


# ── Custom algorithm implementation (editable) ─────────────────────────────


class Custom(Sequential):
    """Custom lifelong learning algorithm.

    Override __init__, start_task, observe, and/or end_task to implement
    your lifelong learning strategy. The goal is to minimize catastrophic
    forgetting across sequential robot manipulation tasks.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

    def start_task(self, task):
        super().start_task(task)

    def observe(self, data):
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
        pass
