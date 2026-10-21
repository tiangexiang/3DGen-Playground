"""GaussianVerse datasets with frozen T5 conditioning.

Reuses the shared PLY loader, Hilbert ordering, normalization, and decoded-data
cache. Splits are deterministic and keyed by caption stem."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .text import T5TokenStore

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from dataloaders.text_3dgen_loader import (
    DC_ONLY_FEATURE_INDICES,
    Text3DGenDataset,
)

logger = logging.getLogger(__name__)


class T5Text3DGenDataset(Text3DGenDataset):
    """Use frozen T5 tokens through the shared text-dataset cache.

    An empty class map bypasses the parent's CLIP requirement. Reset label mode
    afterwards; _fetch_cond supplies T5 conditioning."""

    def __init__(self, base_dataset, t5_store: T5TokenStore, **kwargs):
        super().__init__(base_dataset, class_map={}, **kwargs)
        self.t5 = t5_store
        self.label_mode = False
        self.class_map = None
        self.text_dim = t5_store.dim

    def _fetch_cond(self, hash_key: str, sample=None):
        return self.t5.get(caption_key_for(self.base_dataset, hash_key))


def caption_key_for(base_dataset, hash_key: str) -> str:
    """Map an object hash to its caption key: the archive path without .tar.gz."""
    return base_dataset.obj_data[hash_key].split(".tar.gz")[0]


def _split_keys(keys: list[str], heldout_size: int) -> tuple[list[str], list[str]]:
    """Reserve the last heldout_size sorted keys for validation."""
    if heldout_size <= 0:
        return list(keys), []
    if heldout_size >= len(keys):
        raise ValueError(f"heldout_size={heldout_size} >= dataset size {len(keys)}")
    return list(keys[:-heldout_size]), list(keys[-heldout_size:])


def build_base_dataset(cfg: TrainConfig) -> Standard3DGenDataset:
    return Standard3DGenDataset(
        obj_list=[cfg.obj_list],
        gs_path=cfg.gs_path,
        caption_path=cfg.caption_path,
        mean_file=cfg.resolved("mean_file"),
        std_file=cfg.resolved("std_file"),
        sphere2plane_path=cfg.resolved("sphere2plane_path"),
        exclude_keys_file=cfg.resolved("exclude_keys_file"),
        rank_transform_file=cfg.resolved("rank_transform_file"),
        clip_thresholds_file=cfg.resolved("clip_thresholds_file"),
        text_embed_path=None,          # T5 tokens come from T5TokenStore, not here
        token_order=cfg.token_order,
        hilbert_bits=cfg.hilbert_bits,
    )


def build_dataset(cfg: TrainConfig, split: str = "train", *, t5_store: T5TokenStore | None = None,
                  return_full_for_render: bool = False):
    """Build a deterministic split from objects with cached T5 tokens."""
    if t5_store is None:
        t5_store = T5TokenStore(cfg.t5_token_dir, expect_len=cfg.prompt_length,
                                expect_dim=cfg.txt_input_size)
    base = build_base_dataset(cfg)

    n_before = len(base.keys)
    covered = [k for k in base.keys if caption_key_for(base, k) in t5_store]
    if len(covered) < n_before:
        logger.warning(
            "Dropping %d of %d objects with no T5 encoding (%.2f%%). If this is not "
            "~0, the encode is incomplete -- check keys.json in %s.",
            n_before - len(covered), n_before,
            100 * (n_before - len(covered)) / max(n_before, 1), cfg.t5_token_dir,
        )
    if not covered:
        raise RuntimeError(
            f"No object in the dataset has a T5 encoding. This is almost always a "
            f"key-space mismatch, not an empty encode: dataset keys are obj_list "
            f"KEYS, while {cfg.t5_token_dir} is keyed by caption stem (obj_list "
            f"VALUES minus '.tar.gz'). Sample dataset key={base.keys[0]!r} -> "
            f"caption_key={caption_key_for(base, base.keys[0])!r}."
        )
    # Sort by CAPTION key so the held-out split is stable regardless of obj_list
    # ordering, then map back.
    covered = sorted(covered, key=lambda k: caption_key_for(base, k))
    train_keys, held_keys = _split_keys(covered, cfg.heldout_size)
    keep = set(train_keys if split == "train" else held_keys)
    if not keep:
        raise ValueError(f"split={split!r} is empty")

    base.keys = [k for k in base.keys if k in keep]
    base.obj_data = {k: v for k, v in base.obj_data.items() if k in keep}
    logger.info("split=%s -> %d objects", split, len(base.keys))

    feature_indices = (torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
                       if cfg.sh_degree0_only else None)
    return T5Text3DGenDataset(
        base,
        t5_store,
        feature_indices=feature_indices,
        return_full_for_render=return_full_for_render,
        lazy_cache_to_cpu=cfg.lazy_cache_to_cpu,
        cache_root=cfg.cache_dir,
        # Match AMP to halve cache storage for bf16 training.
        cache_dtype=(torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float32),
    )


def collate(batch):
    """Collate ``(pc, cond, [pc_full,] hash_key)`` tuples."""
    has_full = len(batch[0]) == 4
    pc = torch.stack([b[0] for b in batch])
    tokens = torch.stack([b[1]["tokens"] for b in batch])
    mask = torch.stack([b[1]["mask"] for b in batch])
    out = {"pc": pc, "tokens": tokens, "mask": mask,
           "key": [b[-1] for b in batch]}
    if has_full:
        out["pc_full"] = torch.stack([b[2] for b in batch])
    return out


def make_loader(cfg: TrainConfig, dataset, *, shuffle: bool = True,
                batch_size: int | None = None, distributed: bool = False,
                num_workers: int | None = None) -> DataLoader:
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        shuffle = False
    nw = cfg.num_workers if num_workers is None else num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=cfg.dataloader_prefetch if nw > 0 else None,
        drop_last=True,
    )


def check_training_data(cfg: TrainConfig) -> None:
    """Fail loudly at launch, not 40 minutes into a job, when inputs are missing."""
    problems = []
    for label, path in (("obj_list", cfg.obj_list), ("caption_path", cfg.caption_path),
                        ("gs_path", cfg.gs_path)):
        if not Path(path).exists():
            problems.append(f"{label} does not exist: {path}")
    for key in TrainConfig._ASSET_KEYS:
        p = cfg.resolved(key)
        if p is not None and not Path(p).exists():
            problems.append(f"{key} does not exist: {p}")
    tok_dir = Path(cfg.t5_token_dir)
    if not tok_dir.is_dir():
        problems.append(f"t5_token_dir does not exist: {tok_dir} "
                        "(run python prepare.py --config-file YOUR_CONFIG)")
    else:
        for f in ("keys.json", "tokens.npy", "mask.npy"):
            if not (tok_dir / f).exists():
                problems.append(f"t5_token_dir is missing {f}: {tok_dir}")
    if problems:
        raise RuntimeError("Training data check failed:\n  - " + "\n  - ".join(problems))
