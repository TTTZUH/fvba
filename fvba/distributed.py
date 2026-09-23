from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def get_context(enabled: bool = False) -> DistributedContext:
    requested = enabled or "RANK" in os.environ
    if not requested:
        return DistributedContext(False, 0, 1, 0)
    if not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return DistributedContext(True, dist.get_rank(), dist.get_world_size(), local_rank)


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        dist.barrier()
