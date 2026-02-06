"""Utility functions for GRPO training."""

import torch
import torch.distributed as dist


def gather_tensor(tensor):
    """Gather tensor from all processes."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)
