"""Distributed training, learning-rate schedules, EMA, and atomic checkpoints."""
from __future__ import annotations

import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def seed_all(seed: int) -> None:
    seed = seed + rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_dtype(name: str) -> torch.dtype:
    n = (name or "").lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def jax_warmup_constant_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup from 1e-6 to base_lr, then constant. ``total_steps`` unused."""
    del total_steps
    if step < warmup_steps:
        start_lr = 1e-6
        alpha = float(step + 1) / max(warmup_steps, 1)
        return start_lr + alpha * (base_lr - start_lr)
    return base_lr


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 1e-6) -> float:
    """Linear warmup followed by cosine decay to min_lr."""
    import math
    if step < warmup_steps:
        return base_lr * float(step + 1) / max(warmup_steps, 1)
    span = max(total_steps - warmup_steps, 1)
    frac = min(max(float(step - warmup_steps) / span, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * frac))


def lr_at(schedule: str, step: int, total_steps: int, warmup_steps: int, base_lr: float,
          min_lr: float = 1e-6) -> float:
    if schedule == "warmup_constant":
        return jax_warmup_constant_lr(step, total_steps, warmup_steps, base_lr)
    if schedule == "cosine":
        return cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr)
    raise ValueError(f"unknown lr_schedule={schedule!r}; choices: warmup_constant, cosine")


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    src = model.module if hasattr(model, "module") else model
    src = getattr(src, "_orig_mod", src)  # torch.compile wrapper
    for ema_p, p in zip(ema_model.parameters(), src.parameters()):
        ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), src.buffers()):
        ema_b.copy_(b)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Remove DDP and torch.compile wrappers before checkpointing or inference."""
    m = model.module if hasattr(model, "module") else model
    return getattr(m, "_orig_mod", m)


def atomic_torch_save(obj: object, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    with zipfile.ZipFile(tmp) as zf:
        if not any(name.endswith("/.data/serialization_id") for name in zf.namelist()):
            raise RuntimeError(f"checkpoint archive is missing serialization metadata: {tmp}")
    tmp.replace(path)
    dir_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
