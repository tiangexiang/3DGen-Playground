"""Read the FLAN-T5 token directory written by ``tools/encode_t5_tokens.py``.

Opened with ``mmap_mode='r'`` so every dataloader worker shares one page cache
instead of each faulting in its own copy of a 50 GB array.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class T5TokenStore:
    """Caption-stem -> (tokens (L,D) fp16, mask (L,) bool) lookup.

    Accepts either a single encode directory or a list of shard directories
    (``--num-shards`` writes one per process); shards are concatenated by key.
    """

    def __init__(self, roots: str | Path | list[str | Path], expect_len: int | None = None,
                 expect_dim: int | None = None):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        self.roots = [Path(r) for r in roots]
        self._tokens: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self.key_to_row: dict[str, tuple[int, int]] = {}
        self.meta: dict = {}

        for shard, root in enumerate(self.roots):
            if not root.is_dir():
                raise FileNotFoundError(
                    f"T5 token directory not found: {root}. Run "
                    "tools/encode_t5_tokens.py first."
                )
            keys_path = root / "keys.json"
            if not keys_path.exists():
                raise FileNotFoundError(
                    f"{root} has no keys.json -- the encode never reached its first "
                    "flush, so the arrays are unwritten. Re-run the encoder."
                )
            keys = json.loads(keys_path.read_text())
            tok = np.load(root / "tokens.npy", mmap_mode="r")
            msk = np.load(root / "mask.npy", mmap_mode="r")
            meta_path = root / "meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if not self.meta:
                self.meta = meta

            # A partial encode leaves keys.json SHORTER than the memmap. Trusting
            # the array length instead would silently feed uninitialised zeros as
            # captions, which trains but conditions on nothing.
            if len(keys) < tok.shape[0]:
                pass  # keys.json is authoritative; extra rows are simply unreferenced
            elif len(keys) > tok.shape[0]:
                raise ValueError(
                    f"{root}: keys.json lists {len(keys)} keys but tokens.npy has "
                    f"{tok.shape[0]} rows."
                )
            self._tokens.append(tok)
            self._masks.append(msk)
            for row, k in enumerate(keys):
                self.key_to_row[k] = (shard, row)

        self.seq_len = int(self._tokens[0].shape[1])
        self.dim = int(self._tokens[0].shape[2])
        for t in self._tokens[1:]:
            if (t.shape[1], t.shape[2]) != (self.seq_len, self.dim):
                raise ValueError("shard token arrays disagree on (seq_len, dim)")
        if expect_len is not None and self.seq_len != expect_len:
            raise ValueError(
                f"T5 tokens have seq_len={self.seq_len} but config prompt_length="
                f"{expect_len}. The model's RoPE position space and mask_token "
                "geometry are sized from the config, so these must agree."
            )
        if expect_dim is not None and self.dim != expect_dim:
            raise ValueError(
                f"T5 tokens have dim={self.dim} but config txt_input_size={expect_dim}."
            )

    def __len__(self) -> int:
        return len(self.key_to_row)

    def __contains__(self, key: str) -> bool:
        return key in self.key_to_row

    def get(self, key: str) -> dict[str, torch.Tensor]:
        shard, row = self.key_to_row[key]
        # .copy() is required: these are read-only mmaps and torch.from_numpy on a
        # read-only buffer yields a tensor that silently shares (and may later be
        # written through) the page cache.
        tokens = torch.from_numpy(np.asarray(self._tokens[shard][row]).copy())
        mask = torch.from_numpy(np.asarray(self._masks[shard][row]).copy()).to(torch.bool)
        return {"tokens": tokens, "mask": mask}

    def describe(self) -> str:
        return (f"T5TokenStore({len(self):,} captions, seq_len={self.seq_len}, "
                f"dim={self.dim}, model={self.meta.get('t5_name', '?')}, "
                f"shards={len(self.roots)})")
