"""Muon/AdamW optimizer construction and learning-rate scheduling.

The default recipe uses Muon for matrix weights and AdamW for the rest."""
from __future__ import annotations

import logging

import torch

from .config import TrainConfig

logger = logging.getLogger(__name__)


def _split_params(model: torch.nn.Module):
    """(matrix params, everything else). Newton-Schulz needs true 2-D matrices."""
    matrices, others = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = (p.ndim == 2
                     and "embed" not in name.lower()
                     and "mask_token" not in name)
        (matrices if is_matrix else others).append(p)
    return matrices, others


def build_optimizer(cfg: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "muon":
        from .muon import Muon
        matrices, others = _split_params(model)
        logger.info("Muon: %d matrix params, %d AdamW params",
                    len(matrices), len(others))
        return Muon([
            {"params": matrices, "use_muon": True, "lr": cfg.muon_lr,
             "momentum": cfg.muon_momentum, "ns_steps": cfg.muon_ns_steps,
             "weight_decay": cfg.weight_decay, "nesterov": True},
            {"params": others, "use_muon": False, "lr": cfg.learning_rate,
             "betas": (cfg.adam_beta1, cfg.adam_beta2), "eps": 1e-8,
             "weight_decay": cfg.weight_decay},
        ])
    raise ValueError(f"optimizer must be adamw|muon, got {cfg.optimizer!r}")


def set_lr(optimizer: torch.optim.Optimizer, cfg: TrainConfig, lr: float) -> None:
    """Apply the schedule. Muon groups keep their own (much larger) base lr, scaled
    by the same schedule shape so warmup/decay applies to both consistently."""
    scale = lr / max(cfg.learning_rate, 1e-12)
    for g in optimizer.param_groups:
        if g.get("use_muon", False):
            g["lr"] = cfg.muon_lr * scale
        else:
            g["lr"] = lr
