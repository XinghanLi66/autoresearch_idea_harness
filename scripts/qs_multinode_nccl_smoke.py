#!/usr/bin/env python3
"""Multi-node NCCL all-reduce smoke (Phase 1 of the 64+ GPU effort).

Launched by torchrun (one process per GPU) across N QS pods. Verifies:
  1. Rendezvous: every rank joins the process group (global world = N_pods * 4).
  2. Correctness: an all-reduce of each rank's rank-id equals world*(world-1)/2 on all ranks.
  3. Fabric bandwidth: a 1 GB all-reduce loop; rank-0 prints alg/bus bandwidth. IB-class busbw is
     hundreds of GB/s; single-digit GB/s means it fell back to TCP sockets (misconfigured NCCL_IB_*).

Run: torchrun --nnodes=$PET_NNODES --node-rank=$PET_NODE_RANK --nproc-per-node=4 \
       --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT scripts/qs_multinode_nccl_smoke.py
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    host = os.uname().nodename

    # 1+2. rendezvous + correctness
    t = torch.tensor([float(rank)], device=dev)
    dist.all_reduce(t)
    expected = world * (world - 1) / 2.0
    ok = abs(t.item() - expected) < 1e-3
    print(f"[nccl-smoke] rank={rank}/{world} local_rank={local_rank} host={host} "
          f"allreduce_sum={t.item():.1f} expected={expected:.1f} OK={ok}", flush=True)

    # 3. bandwidth (1 GB tensor, timed all-reduce loop)
    dist.barrier()
    n = 256 * 1024 * 1024  # 256M fp32 = 1 GiB
    big = torch.ones(n, device=dev)
    iters = 10
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    dist.barrier()
    dt = (time.time() - t0) / iters
    size_bytes = n * 4
    algbw = size_bytes / dt / 1e9
    busbw = algbw * 2 * (world - 1) / world  # ring all-reduce bus-bandwidth formula
    if rank == 0:
        print(f"[nccl-smoke] all-reduce {size_bytes/1e9:.2f}GB x{iters}: {dt*1e3:.1f} ms/iter | "
              f"algbw={algbw:.1f} GB/s busbw={busbw:.1f} GB/s | world={world} "
              f"(IB-class = hundreds GB/s; TCP = single digits)", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
