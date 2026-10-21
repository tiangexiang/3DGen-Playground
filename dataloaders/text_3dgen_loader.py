"""Text-conditioned Gaussian splats with shared CPU caching.

Pairs Standard3DGenDataset samples with precomputed CLIP embeddings, optionally
selecting feature channels. T5 conditioning is supplied by a subclass.

Cache files: pc.dat (selected grids), optional pc_full.dat (full SH grids),
ready.dat (lazy-fill flags), hash_keys.json (row keys), and meta.json (layout).
Cache keys include preprocessing inputs and feature selection."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloaders.standard_3dgen_loader import (
    Standard3DGenDataset,
    _apply_rank_transform_numpy,
    _normalize_point_cloud_numpy,
    extract_directory_info,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger(__name__)

FULL_3DGS_FEATURE_DIM = 59
# DC-only features: xyz, opacity, RGB DC coefficients, log-scales, quaternion.
# GaussianVerse stores DC colour in BGR order; indices 36,20,4 restore RGB.
# Statistics and cache keys use the same feature indices.
DC_ONLY_FEATURE_INDICES = (0, 1, 2, 3, 36, 20, 4, 52, 53, 54, 55, 56, 57, 58)

# Text caches omit class labels and include clipping thresholds in their keys.
PRELOAD_CACHE_VERSION = 2  # bumped: clip thresholds now part of the cache key
LAZY_CACHE_LOCK_STRIPES = 256
_PRELOAD_WORKER_STATE: dict = {}


# ---------------------------------------------------------------------------
# Cache primitives (ported with simplifications from class_3dgen_loader)
# ---------------------------------------------------------------------------

def _default_preload_cache_root() -> Path:
    override = os.environ.get("DGEN_CACHE_ROOT")
    candidate = Path(override) if override else Path.home() / "3dgen_cache" / "preload_text"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"preload_to_cpu requires a writable cache directory at {candidate} "
            "(set DGEN_CACHE_ROOT to override)"
        ) from exc
    if not os.access(candidate, os.W_OK):
        raise RuntimeError(
            f"preload_to_cpu requires a writable cache directory at {candidate} "
            "(set DGEN_CACHE_ROOT to override)"
        )
    return candidate


def _hash_strings(hasher: "hashlib._Hash", values) -> None:
    for value in values:
        encoded = value.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)


def _hash_array(hasher: "hashlib._Hash", array) -> None:
    arr = np.ascontiguousarray(np.asarray(array))
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.view(np.uint8).tobytes())


def _torch_dtype_to_numpy(dtype: torch.dtype) -> np.dtype:
    return np.dtype(torch.empty((), dtype=dtype).numpy().dtype)


def _torch_dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported cache dtype: {dtype}")


def _dtype_name_to_torch(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported cache dtype name: {name}")


def _element_size_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _open_tensor_memmap(
    path: str,
    mode: str,
    tensor_dtype: torch.dtype,
    shape: tuple[int, ...],
) -> tuple[np.memmap, torch.Tensor]:
    if tensor_dtype == torch.bfloat16:
        backing = np.memmap(path, mode=mode, dtype=np.uint16, shape=shape)
        tensor = torch.from_numpy(backing).view(torch.bfloat16)
    else:
        backing = np.memmap(path, mode=mode, dtype=_torch_dtype_to_numpy(tensor_dtype), shape=shape)
        tensor = torch.from_numpy(backing)
    return backing, tensor


# ---------------------------------------------------------------------------
# Per-sample build (PLY -> rank -> norm -> feature select -> reshape)
# ---------------------------------------------------------------------------

def _load_preload_point_cloud(
    base_dataset: Standard3DGenDataset, real_idx: int
) -> np.ndarray:
    hash_key = base_dataset.keys[real_idx]
    tar_gz_path = base_dataset.obj_data[hash_key]
    directory_number, filename = extract_directory_info(tar_gz_path)
    point_cloud, _ = base_dataset._load_3dgs_data(directory_number, filename)

    if getattr(base_dataset, "rank_channels", None) is not None:
        point_cloud = _apply_rank_transform_numpy(
            point_cloud,
            base_dataset.rank_channels,
            base_dataset.rank_data_quantiles,
            base_dataset.rank_gauss_quantiles,
        )
    if base_dataset.mean is not None and base_dataset.std is not None:
        point_cloud = _normalize_point_cloud_numpy(point_cloud, base_dataset.mean, base_dataset.std)
    return point_cloud.astype(np.float32, copy=False)


def _select_features_numpy(
    point_cloud: np.ndarray, feature_indices: Optional[np.ndarray]
) -> np.ndarray:
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if feature_indices is None:
        return point_cloud
    if point_cloud.ndim == 2:
        return point_cloud[:, feature_indices]
    if point_cloud.ndim == 3:
        return point_cloud[feature_indices]
    raise ValueError(f"Expected pc with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")


def _plane_to_grid_numpy(point_cloud: np.ndarray) -> np.ndarray:
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if point_cloud.ndim == 3:
        return np.ascontiguousarray(point_cloud)
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected pc with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")
    n, d = point_cloud.shape
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"N={n} is not a perfect square")
    return np.ascontiguousarray(point_cloud.reshape(side, side, d).transpose(2, 0, 1))


def _plane_to_grid_torch(point_cloud: torch.Tensor) -> torch.Tensor:
    if point_cloud.ndim == 4 and point_cloud.shape[0] == 1:
        point_cloud = point_cloud[0]
    if point_cloud.ndim == 3:
        return point_cloud.contiguous()
    if point_cloud.ndim != 2:
        raise ValueError(f"Expected pc with 2 or 3 dims, got shape {tuple(point_cloud.shape)}")
    n, d = point_cloud.shape
    side = int(math.isqrt(n))
    assert side * side == n, f"N={n} is not a perfect square"
    return point_cloud.reshape(side, side, d).permute(2, 0, 1).contiguous()


def _select_features_torch(
    pc: torch.Tensor, feature_indices: Optional[torch.Tensor]
) -> torch.Tensor:
    if pc.ndim == 4 and pc.shape[0] == 1:
        pc = pc[0]
    if feature_indices is None:
        return pc
    if pc.ndim == 2:
        return pc[:, feature_indices]
    if pc.ndim == 3:
        return pc[feature_indices]
    raise ValueError(f"Expected pc with 2 or 3 dims, got shape {tuple(pc.shape)}")


def _build_preload_grids(
    base_dataset: Standard3DGenDataset,
    real_idx: int,
    feature_indices: Optional[np.ndarray],
    return_full_for_render: bool,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    pc_full = _load_preload_point_cloud(base_dataset, real_idx)
    pc = _select_features_numpy(pc_full, feature_indices)
    pc_plane = _plane_to_grid_numpy(pc)
    if return_full_for_render:
        return pc_plane, _plane_to_grid_numpy(pc_full)
    return pc_plane, None


# ---------------------------------------------------------------------------
# Parallel preload worker (pickled into child processes)
# ---------------------------------------------------------------------------

def _init_preload_worker(
    base_dataset: Standard3DGenDataset,
    feature_indices: Optional[np.ndarray],
    return_full_for_render: bool,
    pc_path: str,
    pc_shape: tuple[int, ...],
    pc_dtype_name: str,
    pc_full_path: Optional[str],
    pc_full_shape: Optional[tuple[int, ...]],
    pc_full_dtype_name: Optional[str],
) -> None:
    global _PRELOAD_WORKER_STATE
    _PRELOAD_WORKER_STATE = {
        "base_dataset": base_dataset,
        "feature_indices": feature_indices,
        "return_full_for_render": return_full_for_render,
        "pc_tensor": _open_tensor_memmap(
            pc_path, mode="r+", tensor_dtype=_dtype_name_to_torch(pc_dtype_name),
            shape=tuple(pc_shape),
        )[1],
        "pc_full_tensor": None,
    }
    if return_full_for_render:
        _PRELOAD_WORKER_STATE["pc_full_tensor"] = _open_tensor_memmap(
            pc_full_path, mode="r+",
            tensor_dtype=_dtype_name_to_torch(pc_full_dtype_name),
            shape=tuple(pc_full_shape),
        )[1]


def _preload_worker_write_sample(task: tuple[int, int]) -> int:
    slot, real_idx = task
    state = _PRELOAD_WORKER_STATE
    pc_plane, pc_full_plane = _build_preload_grids(
        base_dataset=state["base_dataset"],
        real_idx=real_idx,
        feature_indices=state["feature_indices"],
        return_full_for_render=state["return_full_for_render"],
    )
    state["pc_tensor"][slot].copy_(torch.from_numpy(pc_plane).to(dtype=state["pc_tensor"].dtype))
    if state["return_full_for_render"]:
        state["pc_full_tensor"][slot].copy_(
            torch.from_numpy(pc_full_plane).to(dtype=state["pc_full_tensor"].dtype)
        )
    return slot


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Text3DGenDataset(Dataset):
    """Wraps Standard3DGenDataset for text-conditional training.

    Requires the underlying ``base_dataset`` to have been constructed with
    ``text_embed_path`` set — it must return a ``text_pooled`` (D,) tensor per
    sample (EOS-pooled CLIP vector) for AdaLN conditioning.

    Returns either
        ``(pc, text_pooled, hash_key)`` or
        ``(pc, text_pooled, pc_full, hash_key)``
    when ``return_full_for_render=True`` and feature selection is active.

    Caching: when ``preload_to_cpu`` or ``lazy_cache_to_cpu`` is set, the
    point-cloud grids (which require PLY parse + reorder + rank transform +
    normalize) are mmapped into a shared cache directory. Pooled vectors are
    NOT cached here — they already live as a single COW-shared numpy array on
    the base dataset.
    """

    def __init__(
        self,
        base_dataset: Standard3DGenDataset,
        feature_indices: Optional[torch.Tensor] = None,
        return_full_for_render: bool = False,
        preload_to_cpu: bool = False,
        lazy_cache_to_cpu: bool = False,
        cache_dtype: torch.dtype = torch.float32,
        preload_max_samples: int = 0,
        preload_workers: int = 0,
        class_map: Optional[dict] = None,
        return_text_tokens: bool = False,
        cache_root: str | Path | None = None,
    ):
        # Three conditioning modes:
        #  - class_map given  -> DISCRETE class conditioning: __getitem__ returns a
        #    (scalar long) class id in place of text_pooled (model uses LabelEmbedder).
        #  - return_text_tokens -> CROSS-ATTENTION text conditioning: the cond slot
        #    carries a dict {'pooled', 'tokens', 'mask'} so the model can attend to
        #    individual caption tokens. Default collate batches the dict per key.
        #  - else             -> pooled-AdaLN text conditioning: base_dataset must
        #    carry text_pooled.
        self.class_map = class_map
        self.label_mode = class_map is not None
        self.return_text_tokens = bool(return_text_tokens and not self.label_mode)
        if not self.label_mode and base_dataset.text_pooled is None:
            raise ValueError(
                "Text3DGenDataset requires Standard3DGenDataset to be constructed "
                "with text_embed_path set (or pass class_map for class conditioning); "
                "got base_dataset.text_pooled=None."
            )
        if self.return_text_tokens and base_dataset.text_tokens is None:
            raise ValueError(
                "return_text_tokens=True needs per-token CLIP embeddings, but the "
                "text_embed_path supplied no tokens.npy. Re-run "
                "object_classification/encode_text_embeddings.py to emit "
                "tokens.npy/mask.npy alongside pooled.npy."
            )
        if preload_to_cpu and lazy_cache_to_cpu:
            raise ValueError("preload_to_cpu and lazy_cache_to_cpu are mutually exclusive")
        if preload_max_samples < 0:
            raise ValueError("preload_max_samples must be >= 0")
        if getattr(base_dataset, "point_cloud_order", "plane") != "plane":
            raise ValueError("Text3DGenDataset expects a plane-ordered base dataset")

        self.base_dataset = base_dataset
        self.feature_indices = feature_indices
        self.return_full_for_render = bool(return_full_for_render and feature_indices is not None)
        self.preload_to_cpu = preload_to_cpu
        self.lazy_cache_to_cpu = lazy_cache_to_cpu
        self.cache_root = Path(cache_root).expanduser() if cache_root is not None else None
        self.cache_dtype = cache_dtype
        self.preload_max_samples = preload_max_samples
        self.preload_workers = preload_workers
        self.text_dim = None if self.label_mode else int(base_dataset.text_pooled.shape[1])

        self._feature_indices_np = (
            self.feature_indices.cpu().numpy() if self.feature_indices is not None else None
        )

        # Cache state
        self.cached_pc: Optional[torch.Tensor] = None
        self.cached_pc_full: Optional[torch.Tensor] = None
        self.cached_hash_keys: Optional[list[str]] = None
        self.cached_ready: Optional[np.memmap] = None
        self.cached_sample_count: int = 0
        self.cache_dir: Optional[Path] = None
        self.cache_mode: Optional[str] = None
        self._cache_backing: dict = {}

        logger.info(
            f"Text3DGenDataset: {len(base_dataset):,} samples, text_dim={self.text_dim}, "
            f"feature_selection={'on' if feature_indices is not None else 'off'}, "
            f"return_full_for_render={self.return_full_for_render}, "
            f"preload={'eager' if preload_to_cpu else ('lazy' if lazy_cache_to_cpu else 'off')}"
        )

        if self.preload_to_cpu:
            self._attach_or_build_cache(mode="preload")
        elif self.lazy_cache_to_cpu:
            self._attach_or_build_cache(mode="lazy")

    def __len__(self) -> int:
        if self.preload_to_cpu and self.cached_sample_count > 0:
            return self.cached_sample_count
        return len(self.base_dataset)

    # ----- Cache build / attach --------------------------------------------

    def _cache_key(self, mode: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"text3dgen_preload_v{PRELOAD_CACHE_VERSION}".encode("utf-8"))
        hasher.update(f"mode:{mode}".encode("utf-8"))
        _hash_strings(hasher, self.base_dataset.keys)
        _hash_strings(
            hasher, [self.base_dataset.obj_data[k] for k in self.base_dataset.keys]
        )
        if self.feature_indices is None:
            hasher.update(b"feature_indices:none")
        else:
            _hash_array(hasher, self.feature_indices.cpu().numpy())
        for attr_name, prefix in (("mean", b"mean"), ("std", b"std")):
            v = getattr(self.base_dataset, attr_name)
            if v is None:
                hasher.update(prefix + b":none")
            else:
                _hash_array(hasher, v)
        rank_channels = getattr(self.base_dataset, "rank_channels", None)
        if rank_channels is None:
            hasher.update(b"rank:none")
        else:
            hasher.update(b"rank:on")
            _hash_array(hasher, np.asarray(rank_channels, dtype=np.int64))
            _hash_array(hasher, np.asarray(self.base_dataset.rank_data_quantiles))
            _hash_array(hasher, np.asarray(self.base_dataset.rank_gauss_quantiles))
        clip_channels = getattr(self.base_dataset, "clip_channels", None)
        if clip_channels is None:
            hasher.update(b"clip:none")
        else:
            hasher.update(b"clip:on")
            _hash_array(hasher, np.asarray(clip_channels, dtype=np.int64))
            _hash_array(hasher, np.asarray(self.base_dataset.clip_lower))
            _hash_array(hasher, np.asarray(self.base_dataset.clip_upper))
        token_order = getattr(self.base_dataset, "token_order", "plane2d")
        hasher.update(f"token_order:{token_order}".encode("utf-8"))
        if token_order == "hilbert1d":
            hasher.update(f"hilbert_bits:{getattr(self.base_dataset, 'hilbert_bits', 8)}".encode("utf-8"))
        else:
            sphere2plane = getattr(self.base_dataset, "sphere2plane", None)
            if sphere2plane is not None:
                _hash_array(hasher, sphere2plane)
        hasher.update(str(self.base_dataset.gs_path).encode("utf-8"))
        hasher.update(b"return_full:1" if self.return_full_for_render else b"return_full:0")
        hasher.update(f"dtype:{_torch_dtype_to_name(self.cache_dtype)}".encode("utf-8"))
        hasher.update(f"max_samples:{self.preload_max_samples}".encode("utf-8"))
        return hasher.hexdigest()[:32]

    def _cache_paths(self, mode: str) -> tuple[Path, Path]:
        cache_root = self.cache_root if self.cache_root is not None else _default_preload_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(mode)
        cache_dir = cache_root / key
        return cache_dir, cache_root / f"{key}.lock"

    def _resolved_workers(self, n: int) -> int:
        if n <= 1:
            return 1
        if self.preload_workers > 0:
            return min(self.preload_workers, n)
        cpu_count = os.cpu_count() or 1
        return min(n, max(1, cpu_count))

    def _resolved_sample_count(self, n: int) -> int:
        return min(n, self.preload_max_samples) if self.preload_max_samples > 0 else n

    def _attach_or_build_cache(self, mode: str) -> None:
        cache_dir, lock_path = self._cache_paths(mode)
        meta_path = cache_dir / "meta.json"
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if meta_path.exists():
                try:
                    self._attach(meta_path, expected_mode=mode)
                    logger.info(f"Attached to existing shared CPU {mode} cache: {cache_dir}")
                    return
                except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                    logger.warning(f"Cache at {cache_dir} invalid ({exc}); rebuilding")
                    meta_path.unlink(missing_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            if mode == "preload":
                self._build_eager(cache_dir, meta_path)
            else:
                self._build_lazy_skeleton(cache_dir, meta_path)
            self._attach(meta_path, expected_mode=mode)

    def _attach(self, meta_path: Path, *, expected_mode: str) -> None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("version") != PRELOAD_CACHE_VERSION:
            raise ValueError(f"cache version mismatch at {meta_path}")
        if meta.get("cache_mode") != expected_mode:
            raise ValueError(f"cache mode mismatch at {meta_path}")

        pc_memmap, pc_tensor = _open_tensor_memmap(
            meta["pc_path"], mode="r+",
            tensor_dtype=_dtype_name_to_torch(meta["pc_dtype"]),
            shape=tuple(meta["pc_shape"]),
        )
        ready_memmap = np.memmap(
            meta["ready_path"], mode="r+",
            dtype=np.dtype(meta["ready_dtype"]), shape=tuple(meta["ready_shape"]),
        )
        with open(meta["hash_keys_path"], "r", encoding="utf-8") as f:
            hash_keys = json.load(f)

        self.cached_pc = pc_tensor
        self.cached_hash_keys = hash_keys
        self.cached_ready = ready_memmap
        self.cached_sample_count = int(meta.get("cached_sample_count", len(hash_keys)))
        self.cache_dir = meta_path.parent
        self.cache_mode = expected_mode
        self._cache_backing = {"pc": pc_memmap, "ready": ready_memmap}

        if self.return_full_for_render:
            pc_full_memmap, pc_full_tensor = _open_tensor_memmap(
                meta["pc_full_path"], mode="r+",
                tensor_dtype=_dtype_name_to_torch(meta["pc_full_dtype"]),
                shape=tuple(meta["pc_full_shape"]),
            )
            self.cached_pc_full = pc_full_tensor
            self._cache_backing["pc_full"] = pc_full_memmap
        else:
            self.cached_pc_full = None

    def _build_eager(self, cache_dir: Path, meta_path: Path) -> None:
        t0 = time.time()
        n_total = len(self.base_dataset)
        n_preload = self._resolved_sample_count(n_total)
        worker_count = self._resolved_workers(n_preload)
        logger.info(
            f"Preloading {n_preload:,}/{n_total:,} samples into shared CPU cache "
            f"at {cache_dir} using {worker_count} worker(s)"
        )
        if n_preload == 0:
            raise ValueError("preload_to_cpu enabled but resolved sample count is 0")

        first_pc, first_pc_full = _build_preload_grids(
            self.base_dataset, real_idx=0,
            feature_indices=self._feature_indices_np,
            return_full_for_render=self.return_full_for_render,
        )

        pc_shape = (n_preload,) + tuple(first_pc.shape)
        cache_dtype_name = _torch_dtype_to_name(self.cache_dtype)
        pc_path = cache_dir / "pc.dat"
        ready_path = cache_dir / "ready.dat"
        hash_keys_path = cache_dir / "hash_keys.json"

        pc_memmap, pc_tensor = _open_tensor_memmap(
            str(pc_path), mode="w+", tensor_dtype=self.cache_dtype, shape=pc_shape
        )
        ready_memmap = np.memmap(ready_path, mode="w+", dtype=np.uint8, shape=(n_preload,))
        ready_memmap[:] = 0

        pc_full_memmap = None
        pc_full_tensor = None
        pc_full_path = None
        pc_full_shape = None
        pc_full_dtype_name = None
        if self.return_full_for_render:
            pc_full_shape = (n_preload,) + tuple(first_pc_full.shape)
            pc_full_dtype_name = cache_dtype_name
            pc_full_path = cache_dir / "pc_full.dat"
            pc_full_memmap, pc_full_tensor = _open_tensor_memmap(
                str(pc_full_path), mode="w+",
                tensor_dtype=self.cache_dtype, shape=pc_full_shape,
            )

        hash_keys = [self.base_dataset.keys[i] for i in range(n_preload)]

        def write_sample(slot: int, pc_plane: np.ndarray, pc_full_plane: Optional[np.ndarray]) -> None:
            pc_tensor[slot].copy_(torch.from_numpy(pc_plane).to(dtype=pc_tensor.dtype))
            if self.return_full_for_render:
                pc_full_tensor[slot].copy_(
                    torch.from_numpy(pc_full_plane).to(dtype=pc_full_tensor.dtype)
                )

        progress = tqdm(total=n_preload, desc="preload_to_cpu", unit="sample",
                        dynamic_ncols=True) if tqdm is not None else None
        try:
            write_sample(0, first_pc, first_pc_full)
            if progress is not None:
                progress.update(1)

            task_iter = ((slot, slot) for slot in range(1, n_preload))
            if worker_count == 1:
                for task in task_iter:
                    pc_plane, pc_full_plane = _build_preload_grids(
                        self.base_dataset, real_idx=task[1],
                        feature_indices=self._feature_indices_np,
                        return_full_for_render=self.return_full_for_render,
                    )
                    write_sample(task[0], pc_plane, pc_full_plane)
                    if progress is not None:
                        progress.update(1)
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_preload_worker,
                    initargs=(
                        self.base_dataset, self._feature_indices_np,
                        self.return_full_for_render,
                        str(pc_path), pc_shape, cache_dtype_name,
                        str(pc_full_path) if pc_full_path else None,
                        pc_full_shape, pc_full_dtype_name,
                    ),
                ) as ex:
                    pending = set()
                    max_pending = max(1, worker_count * 2)

                    def submit_next() -> bool:
                        try:
                            t = next(task_iter)
                        except StopIteration:
                            return False
                        pending.add(ex.submit(_preload_worker_write_sample, t))
                        return True

                    for _ in range(max_pending):
                        if not submit_next():
                            break
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            fut.result()
                            if progress is not None:
                                progress.update(1)
                            submit_next()
        finally:
            if progress is not None:
                progress.close()

        ready_memmap[:] = 1
        pc_memmap.flush(); ready_memmap.flush()
        if pc_full_memmap is not None:
            pc_full_memmap.flush()

        hash_keys_tmp = hash_keys_path.with_suffix(".json.tmp")
        with open(hash_keys_tmp, "w", encoding="utf-8") as f:
            json.dump(hash_keys, f)
        os.replace(hash_keys_tmp, hash_keys_path)

        meta = {
            "version": PRELOAD_CACHE_VERSION,
            "cache_mode": "preload",
            "pc_path": str(pc_path),
            "pc_dtype": cache_dtype_name,
            "pc_shape": list(pc_shape),
            "ready_path": str(ready_path),
            "ready_dtype": "uint8",
            "ready_shape": [n_preload],
            "cached_sample_count": n_preload,
            "hash_keys_path": str(hash_keys_path),
            "return_full_for_render": self.return_full_for_render,
        }
        if self.return_full_for_render:
            meta.update({
                "pc_full_path": str(pc_full_path),
                "pc_full_dtype": pc_full_dtype_name,
                "pc_full_shape": list(pc_full_shape),
            })
        meta_tmp = meta_path.with_suffix(".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)

        bytes_pc = int(np.prod(pc_shape)) * _element_size_bytes(self.cache_dtype)
        if self.return_full_for_render:
            bytes_pc += int(np.prod(pc_full_shape)) * _element_size_bytes(self.cache_dtype)
        logger.info(
            f"Finished CPU preload: {n_preload:,} samples in {time.time() - t0:.1f}s "
            f"(dtype={cache_dtype_name}, {bytes_pc / 1024**3:.2f} GiB)"
        )

    def _build_lazy_skeleton(self, cache_dir: Path, meta_path: Path) -> None:
        t0 = time.time()
        logger.info(f"Initializing lazy CPU cache skeleton at {cache_dir}")

        first_pc, first_pc_full = _build_preload_grids(
            self.base_dataset, real_idx=0,
            feature_indices=self._feature_indices_np,
            return_full_for_render=self.return_full_for_render,
        )
        n = len(self.base_dataset)
        pc_shape = (n,) + tuple(first_pc.shape)
        cache_dtype_name = _torch_dtype_to_name(self.cache_dtype)
        pc_path = cache_dir / "pc.dat"
        ready_path = cache_dir / "ready.dat"
        hash_keys_path = cache_dir / "hash_keys.json"

        pc_memmap, pc_tensor = _open_tensor_memmap(
            str(pc_path), mode="w+", tensor_dtype=self.cache_dtype, shape=pc_shape
        )
        ready_memmap = np.memmap(ready_path, mode="w+", dtype=np.uint8, shape=(n,))
        ready_memmap[:] = 0

        pc_full_memmap = None
        pc_full_tensor = None
        pc_full_path = None
        pc_full_shape = None
        pc_full_dtype_name = None
        if self.return_full_for_render:
            pc_full_shape = (n,) + tuple(first_pc_full.shape)
            pc_full_dtype_name = cache_dtype_name
            pc_full_path = cache_dir / "pc_full.dat"
            pc_full_memmap, pc_full_tensor = _open_tensor_memmap(
                str(pc_full_path), mode="w+",
                tensor_dtype=self.cache_dtype, shape=pc_full_shape,
            )

        # Seed slot 0 so __getitem__(0) doesn't pay the build cost.
        pc_tensor[0].copy_(torch.from_numpy(first_pc).to(dtype=pc_tensor.dtype))
        if self.return_full_for_render:
            pc_full_tensor[0].copy_(torch.from_numpy(first_pc_full).to(dtype=pc_full_tensor.dtype))
        ready_memmap[0] = 1

        pc_memmap.flush(); ready_memmap.flush()
        if pc_full_memmap is not None:
            pc_full_memmap.flush()

        hash_keys = [self.base_dataset.keys[i] for i in range(n)]
        hash_keys_tmp = hash_keys_path.with_suffix(".json.tmp")
        with open(hash_keys_tmp, "w", encoding="utf-8") as f:
            json.dump(hash_keys, f)
        os.replace(hash_keys_tmp, hash_keys_path)

        meta = {
            "version": PRELOAD_CACHE_VERSION,
            "cache_mode": "lazy",
            "pc_path": str(pc_path),
            "pc_dtype": cache_dtype_name,
            "pc_shape": list(pc_shape),
            "ready_path": str(ready_path),
            "ready_dtype": "uint8",
            "ready_shape": [n],
            "cached_sample_count": n,
            "hash_keys_path": str(hash_keys_path),
            "return_full_for_render": self.return_full_for_render,
        }
        if self.return_full_for_render:
            meta.update({
                "pc_full_path": str(pc_full_path),
                "pc_full_dtype": pc_full_dtype_name,
                "pc_full_shape": list(pc_full_shape),
            })
        meta_tmp = meta_path.with_suffix(".tmp")
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)

        bytes_pc = int(np.prod(pc_shape)) * _element_size_bytes(self.cache_dtype)
        if self.return_full_for_render:
            bytes_pc += int(np.prod(pc_full_shape)) * _element_size_bytes(self.cache_dtype)
        logger.info(
            f"Lazy cache skeleton ready in {time.time() - t0:.1f}s "
            f"(dtype={cache_dtype_name}, reserved {bytes_pc / 1024**3:.2f} GiB)"
        )

    # ----- Lazy fill -------------------------------------------------------

    def _lazy_lock_path(self, idx: int) -> Path:
        stripe = idx % LAZY_CACHE_LOCK_STRIPES
        return self.cache_dir / f"lazy_{stripe:03d}.lock"

    def _ensure_lazy_cached(self, idx: int) -> None:
        if not self.lazy_cache_to_cpu or self.cached_pc is None:
            return
        if int(self.cached_ready[idx]) == 1:
            return
        lock_path = self._lazy_lock_path(idx)
        with open(lock_path, "a+b") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if int(self.cached_ready[idx]) == 1:
                return
            pc_plane, pc_full_plane = _build_preload_grids(
                self.base_dataset, real_idx=idx,
                feature_indices=self._feature_indices_np,
                return_full_for_render=self.return_full_for_render,
            )
            self.cached_pc[idx].copy_(torch.from_numpy(pc_plane).to(dtype=self.cached_pc.dtype))
            if self.return_full_for_render:
                self.cached_pc_full[idx].copy_(
                    torch.from_numpy(pc_full_plane).to(dtype=self.cached_pc_full.dtype)
                )
            self.cached_ready[idx] = 1

    # ----- __getitem__ -----------------------------------------------------

    def _build_sample_uncached(self, idx: int):
        sample = self.base_dataset[idx]
        pc_full = sample['point_cloud']
        hash_key = sample['hash_key']
        cond = self._fetch_cond(hash_key, sample)

        pc = _select_features_torch(pc_full, self.feature_indices)
        pc = _plane_to_grid_torch(pc)
        if self.return_full_for_render:
            pc_full_grid = _plane_to_grid_torch(pc_full)
            return pc, cond, pc_full_grid, hash_key
        return pc, cond, hash_key

    def _fetch_cond(self, hash_key: str, sample: Optional[dict] = None):
        """Per-object conditioning: a (scalar long) class id in label mode; a dict
        of {'pooled', 'tokens', 'mask'} in cross-attention mode; else the (D,)
        text_pooled vector (reused from ``sample`` if already loaded)."""
        if self.label_mode:
            return self._fetch_label(hash_key)
        if sample is not None and 'text_pooled' in sample:
            pooled = sample['text_pooled']
        else:
            pooled = self._fetch_text_pooled(hash_key)
        if not self.return_text_tokens:
            return pooled
        tokens, mask = self._fetch_text_tokens(hash_key)
        return {'pooled': pooled, 'tokens': tokens, 'mask': mask}

    def _fetch_text_tokens(self, hash_key: str):
        """(L, D) token sequence in its on-disk dtype + (L,) bool validity mask.

        Deliberately does NOT upcast to float32 here. The sequence is the single
        largest per-sample payload in the batch — at eff-batch 512 that is 60.6 MB
        of fp16 per optimizer step, and upcasting on the CPU would double it to
        121 MB of allocate-copy plus host->device traffic. The upcast now happens
        on-GPU in ``_cond_to_model_kwargs`` instead; fp16 -> fp32 is lossless, so
        results are bit-identical to the CPU-side upcast.

        The explicit ``.copy()`` is required: ``text_tokens`` is a read-only mmap,
        and ``torch.from_numpy`` on a read-only view warns and aliases the mapping.
        """
        path = self.base_dataset.obj_data[hash_key]
        caption_key = path.split('.tar.gz')[0]
        row = self.base_dataset.text_embed_key_to_row[caption_key]
        tokens = torch.from_numpy(np.asarray(self.base_dataset.text_tokens[row]).copy())
        text_mask = self.base_dataset.text_mask
        if text_mask is None:
            # No mask array: treat every position as valid. CLIP pads with a
            # repeated EOS embedding, so this is lossy but not incorrect.
            mask = torch.ones(tokens.shape[0], dtype=torch.bool)
        else:
            mask = torch.from_numpy(
                text_mask[row].astype(np.bool_, copy=False)
            )
        return tokens, mask

    def _fetch_label(self, hash_key: str):
        path = self.base_dataset.obj_data[hash_key]
        stem = path.split('.tar.gz')[0]
        return torch.tensor(int(self.class_map[stem]), dtype=torch.long)

    def _fetch_text_pooled(self, hash_key: str):
        path = self.base_dataset.obj_data[hash_key]
        caption_key = path.split('.tar.gz')[0]
        row = self.base_dataset.text_embed_key_to_row[caption_key]
        return torch.from_numpy(
            self.base_dataset.text_pooled[row].astype(np.float32, copy=False)
        )

    def __getitem__(self, idx: int):
        for attempt in range(16):
            try:
                return self._getitem_raw(idx)
            except FileNotFoundError as exc:
                logger.warning(
                    f"Missing data at idx={idx} (attempt={attempt}): {exc}; "
                    f"falling back to neighbor index"
                )
                idx = (idx + 1) % len(self)
        raise RuntimeError(
            f"Text3DGenDataset: 16 consecutive missing-file errors at idx={idx}"
        )

    def _getitem_raw(self, idx: int):
        if self.cached_pc is not None:
            if idx >= self.cached_sample_count:
                if self.preload_to_cpu:
                    raise IndexError(
                        f"idx={idx} outside preloaded range (cached_sample_count={self.cached_sample_count})"
                    )
                return self._build_sample_uncached(idx)
            if self.lazy_cache_to_cpu and int(self.cached_ready[idx]) == 0:
                self._ensure_lazy_cached(idx)
            hash_key = self.cached_hash_keys[idx]
            cond = self._fetch_cond(hash_key)
            if self.return_full_for_render:
                return self.cached_pc[idx], cond, self.cached_pc_full[idx], hash_key
            return self.cached_pc[idx], cond, hash_key

        return self._build_sample_uncached(idx)
