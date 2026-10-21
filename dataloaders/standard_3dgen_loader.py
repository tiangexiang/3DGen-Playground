#!/usr/bin/env python3
"""
Standard 3D Generation Data Loader for GaussianVerse.

This module provides a PyTorch Dataset and DataLoader for loading 3D Gaussian Splatting data
along with optional 2D renderings and text captions for 3D generation tasks.
"""

import os
import json
import logging
import tarfile
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from plyfile import PlyData

from utils.hilbert import hilbert_sort_indices
from PIL import Image


def fov2focal(fov: float, pixels: int) -> float:
    """Convert field of view to focal length.
    
    Args:
        fov: Field of view in radians
        pixels: Image dimension in pixels
        
    Returns:
        Focal length in pixels
    """
    return pixels / (2 * math.tan(fov / 2))


def load_obj_list(obj_list_paths: List[str]) -> Dict[str, str]:
    """Load one or more obj_list JSON files and merge them.
    
    Args:
        obj_list_paths: List of paths to obj_list JSON files
        
    Returns:
        Dictionary mapping hash keys to tar.gz paths (merged from all files)
    """
    merged_data = {}
    
    for obj_list_path in obj_list_paths:
        logging.info(f"Loading obj list from {obj_list_path}")
        
        with open(obj_list_path, 'r') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} entries from {obj_list_path}")
        
        # Merge into the combined dictionary
        merged_data.update(data)
    
    logging.info(f"Total merged entries: {len(merged_data)}")
    return merged_data


def load_captions(caption_path: str) -> Dict[str, str]:
    """Load the captions JSON file.

    Args:
        caption_path: Path to captions.json

    Returns:
        Dictionary mapping foldername/filename to caption text
    """
    logging.info(f"Loading captions from {caption_path}")

    with open(caption_path, 'r') as f:
        data = json.load(f)

    logging.info(f"Loaded {len(data)} captions")
    return data


def load_text_pooled(
    text_embed_path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, int]]:
    """Load the precomputed per-caption CLIP arrays.

    ``text_embed_path`` is one of:
      * A directory containing ``keys.json`` + ``pooled.npy`` (+ optional
        ``tokens.npy`` and ``mask.npy`` for cross-attention conditioning).
        All arrays are opened with ``mmap_mode='r'`` so worker processes
        share a single page cache instead of duplicating per worker.
      * A ``.npz`` file containing ``keys`` + one of ``pooled`` or
        ``embeddings`` (the legacy field name); ``tokens`` / ``mask`` fields
        are loaded if present. Fully loaded into RAM.
      * A comma-separated list of any mix of the above; arrays are
        concatenated in order. Shards must be disjoint. Tokens/mask are
        returned only if EVERY shard supplies them.

    Returns ``(pooled, tokens, mask, key_to_row)`` — pooled (always), tokens
    and mask (optional, ``None`` if absent in source). Arrays stay in their
    on-disk dtype; upcast happens at sample-fetch time.
    """
    shard_paths = [p.strip() for p in text_embed_path.split(",") if p.strip()]
    if not shard_paths:
        raise ValueError(f"Empty text_embed_path: {text_embed_path!r}")

    keys_all: list[str] = []
    pooled_chunks: list[np.ndarray] = []
    tokens_chunks: list[Optional[np.ndarray]] = []
    mask_chunks: list[Optional[np.ndarray]] = []
    sources: list[str] = []
    for path in shard_paths:
        p = Path(path)
        tokens_arr: Optional[np.ndarray] = None
        mask_arr: Optional[np.ndarray] = None
        if p.is_dir():
            logging.info(f"mmap'ing CLIP arrays from {p}")
            keys_json = p / "keys.json"
            with open(keys_json, "r", encoding="utf-8") as f:
                k = json.load(f)
            pooled = np.load(p / "pooled.npy", mmap_mode="r")
            tokens_path = p / "tokens.npy"
            mask_path = p / "mask.npy"
            if tokens_path.exists():
                tokens_arr = np.load(tokens_path, mmap_mode="r")
            if mask_path.exists():
                mask_arr = np.load(mask_path, mmap_mode="r")
            sources.append(f"{p} (mmap)")
        else:
            logging.info(f"Loading CLIP arrays from {path}")
            data = np.load(path, allow_pickle=True)
            if "keys" not in data.files:
                raise ValueError(f"{path} is missing required field 'keys'.")
            if "pooled" in data.files:
                pooled = np.asarray(data["pooled"])
            elif "embeddings" in data.files:
                pooled = np.asarray(data["embeddings"])
            else:
                raise ValueError(
                    f"{path} must contain 'pooled' or legacy 'embeddings' field."
                )
            if "tokens" in data.files:
                tokens_arr = np.asarray(data["tokens"])
            if "mask" in data.files:
                mask_arr = np.asarray(data["mask"])
            k = data["keys"].tolist()
            sources.append(f"{path} (npz)")
        if pooled.shape[0] != len(k):
            raise ValueError(
                f"Corrupt shard {path}: keys={len(k)} pooled={pooled.shape[0]}"
            )
        if tokens_arr is not None and tokens_arr.shape[0] != len(k):
            raise ValueError(
                f"Corrupt shard {path}: keys={len(k)} tokens={tokens_arr.shape[0]}"
            )
        if mask_arr is not None and mask_arr.shape[0] != len(k):
            raise ValueError(
                f"Corrupt shard {path}: keys={len(k)} mask={mask_arr.shape[0]}"
            )
        keys_all.extend(k)
        pooled_chunks.append(pooled)
        tokens_chunks.append(tokens_arr)
        mask_chunks.append(mask_arr)

    if len(pooled_chunks) == 1:
        pooled = pooled_chunks[0]
        tokens = tokens_chunks[0]
        mask = mask_chunks[0]
    else:
        pooled = np.concatenate(pooled_chunks, axis=0)
        if all(t is not None for t in tokens_chunks):
            tokens = np.concatenate(tokens_chunks, axis=0)
        else:
            tokens = None
        if all(m is not None for m in mask_chunks):
            mask = np.concatenate(mask_chunks, axis=0)
        else:
            mask = None

    key_to_row = {k: i for i, k in enumerate(keys_all)}
    if len(key_to_row) != len(keys_all):
        raise ValueError(
            "Duplicate keys across shards — shards must be disjoint "
            "(produced by --num-shards/--shard-id without overlap)."
        )
    extras = []
    if tokens is not None:
        extras.append(f"tokens={tokens.shape}({tokens.dtype})")
    if mask is not None:
        extras.append(f"mask={mask.shape}({mask.dtype})")
    extras_str = f" + {', '.join(extras)}" if extras else ""
    logging.info(
        f"Loaded {len(keys_all):,} pooled CLIP vectors: shape={pooled.shape} "
        f"({pooled.dtype}){extras_str} from {len(shard_paths)} source(s): {', '.join(sources)}"
    )
    return pooled, tokens, mask, key_to_row


def load_null_text_token(path: str) -> np.ndarray:
    """Load the cached empty-string CLIP token from null_text_token.npz.
    Returns a (1, D) fp16 numpy array."""
    logging.info(f"Loading null text token from {path}")
    data = np.load(path)
    nt = np.asarray(data["null_token"])
    if nt.ndim != 2 or nt.shape[0] != 1:
        raise ValueError(
            f"{path} null_token must be shape (1, D); got {nt.shape}"
        )
    return nt


def load_null_text_arrays(
    path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load the empty-string CLIP encoding for both CFG pathways.

    Returns ``(null_token, null_tokens, null_mask)``:
      * ``null_token``  (1, D)  — EOS-pooled vector, for the pooled-AdaLN path.
      * ``null_tokens`` (L, D)  — the full empty-string token sequence, used as
        the unconditional KV sequence by cross-attention. ``None`` when the npz
        predates cross-attention support.
      * ``null_mask``   (L,)    — validity mask for that sequence, or ``None``.
    """
    null_token = load_null_text_token(path)
    data = np.load(path)
    null_tokens = None
    null_mask = None
    if "null_tokens" in data.files:
        null_tokens = np.asarray(data["null_tokens"])
        if null_tokens.ndim == 3 and null_tokens.shape[0] == 1:
            null_tokens = null_tokens[0]
        if null_tokens.ndim != 2:
            raise ValueError(
                f"{path} null_tokens must be (L, D) or (1, L, D); got {null_tokens.shape}"
            )
        if null_tokens.shape[1] != null_token.shape[1]:
            raise ValueError(
                f"{path} null_tokens dim {null_tokens.shape[1]} != null_token dim "
                f"{null_token.shape[1]}"
            )
    if "null_mask" in data.files:
        null_mask = np.asarray(data["null_mask"]).reshape(-1)
    return null_token, null_tokens, null_mask


def load_ply(path: str) -> np.ndarray:
    """Load PLY file and convert to numpy array format.
    
    Args:
        path: Path to PLY file
        
    Returns:
        Numpy array of shape (N, D) containing [xyz, opacity, features, scales, rotations]
    """
    max_sh_degree = 3
    plydata = PlyData.read(path)
    
    # Load positions
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    # Clamp per-axis to the rough unit-cube extent. A handful of objects have
    # stray points 100+σ from the global xyz mean — fitting failures the source
    # 3DGS optimizer never pruned. Clipped points sit at the boundary and add
    # zero pixels to the render anyway.
    xyz = np.clip(xyz, -3.0, 3.0)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # Load DC features
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # Load extra features
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    # Load scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Load rotations
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # 3DGS stores `_rotation` as a free 4D vector — only the direction matters
    # because the renderer normalizes at render time, so per-object ||q|| drift
    # is a fitting artifact that injects huge variance into per-channel stats.
    # Normalize to the unit 3-sphere here, then canonicalize sign so q and -q
    # (which encode the same rotation) don't smear the marginal of rot_w.
    rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-8)
    flip = rots[:, 0] < 0
    rots[flip] *= -1.0

    # Concatenate features
    features = np.concatenate((features_dc, features_extra), axis=-1).reshape(xyz.shape[0], -1)

    return np.concatenate((xyz, opacities, features, scales, rots), axis=1).astype(np.float32)


def extract_directory_info(tar_gz_path: str) -> Tuple[str, str]:
    """Extract directory and filename info from tar.gz path.
    
    Args:
        tar_gz_path: Path like "1923/9611649.tar.gz"
        
    Returns:
        Tuple of (directory_number, filename_without_extension)
    """
    parts = tar_gz_path.split('/')
    if len(parts) != 2:
        raise ValueError(f"Unexpected tar.gz path format: {tar_gz_path}")
    
    directory_number = parts[0]
    filename = parts[1].replace('.tar.gz', '')
    
    return directory_number, filename


def load_sphere2plane(sphere2plane_path: str) -> np.ndarray:
    """Load and validate the plane-to-sphere permutation stored in sphere2plane.npy."""
    arr = np.load(sphere2plane_path).astype(np.int64, copy=False)
    if arr.ndim != 1:
        raise ValueError(f"sphere2plane must be 1D, got shape {arr.shape}")
    expected = np.arange(arr.shape[0], dtype=arr.dtype)
    if not np.array_equal(np.sort(arr), expected):
        raise ValueError(f"sphere2plane at {sphere2plane_path} is not a valid permutation")
    return arr


def load_rank_transform_payload(path: str) -> dict:
    """Load a Gaussian rank-transform payload built by data/build_rank_transform.py.

    Returns a dict with `channels` (list[int], indices into the full 59-channel
    feature space), `data_quantiles` (np.ndarray of shape (C, K), float32), and
    `gauss_quantiles` (np.ndarray of shape (K,), float32). Both quantile arrays
    are strictly monotonic so 1D linear interpolation is well-defined in either
    direction.
    """
    payload = torch.load(path, weights_only=False)
    data_q = payload["data_quantiles"]
    gauss_q = payload["gauss_quantiles"]
    if isinstance(data_q, torch.Tensor):
        data_q = data_q.cpu().numpy()
    if isinstance(gauss_q, torch.Tensor):
        gauss_q = gauss_q.cpu().numpy()
    return {
        "channels": [int(c) for c in payload["channels"]],
        "data_quantiles": np.asarray(data_q, dtype=np.float32),
        "gauss_quantiles": np.asarray(gauss_q, dtype=np.float32),
    }


def load_clip_thresholds_payload(path: str) -> dict:
    """Load per-channel hard-clip thresholds built by data/build_clip_thresholds.py.

    Returns a dict with `channels` (list[int], indices into the full 59-channel
    feature space), `lower` and `upper` (np.ndarray of shape (C,), float32). The
    clip is applied in the raw data domain (right after load_ply), so when it is
    combined with the rank transform it MUST run first — clip then rank.
    """
    payload = torch.load(path, weights_only=False)
    lower = payload["lower"]
    upper = payload["upper"]
    if isinstance(lower, torch.Tensor):
        lower = lower.cpu().numpy()
    if isinstance(upper, torch.Tensor):
        upper = upper.cpu().numpy()
    return {
        "channels": [int(c) for c in payload["channels"]],
        "lower": np.asarray(lower, dtype=np.float32),
        "upper": np.asarray(upper, dtype=np.float32),
    }


def _apply_clip_numpy(
    point_cloud: np.ndarray,
    clip_channels: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Hard-clip listed channels to [lower, upper]; other channels untouched.

    Accepts (C, H, W) or (N, C). Mirrors `_apply_rank_transform_numpy`'s shape
    handling so it can run on either the plane grid or a flat point list.
    """
    if clip_channels.size == 0:
        return point_cloud
    out = point_cloud.copy()
    if point_cloud.ndim == 3:
        for i, c in enumerate(clip_channels):
            out[c] = np.clip(point_cloud[c], lower[i], upper[i])
    elif point_cloud.ndim == 2:
        for i, c in enumerate(clip_channels):
            out[:, c] = np.clip(point_cloud[:, c], lower[i], upper[i])
    else:
        raise ValueError(
            f"Unsupported point_cloud shape for clip: {point_cloud.shape}"
        )
    return out


def _apply_rank_transform_numpy(
    point_cloud: np.ndarray,
    rank_channels: np.ndarray,
    data_quantiles: np.ndarray,
    gauss_quantiles: np.ndarray,
) -> np.ndarray:
    """Forward Gaussian rank transform: data domain -> N(0,1) on listed channels.

    Accepts (C, H, W) or (N, C). Channels not in `rank_channels` are untouched.
    """
    if rank_channels.size == 0:
        return point_cloud
    out = point_cloud.copy()
    if point_cloud.ndim == 3:
        for i, c in enumerate(rank_channels):
            out[c] = np.interp(point_cloud[c], data_quantiles[i], gauss_quantiles)
    elif point_cloud.ndim == 2:
        for i, c in enumerate(rank_channels):
            out[:, c] = np.interp(point_cloud[:, c], data_quantiles[i], gauss_quantiles)
    else:
        raise ValueError(
            f"Unsupported point_cloud shape for rank transform: {point_cloud.shape}"
        )
    return out


def _normalize_point_cloud_numpy(
    point_cloud: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Normalize a point cloud while preserving its original shape."""
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    if point_cloud.ndim == 2:
        if mean_arr.ndim == 1:
            mean_arr = mean_arr[None, :]
            std_arr = std_arr[None, :]
        elif mean_arr.ndim == 3 and mean_arr.shape[1:] == (1, 1):
            mean_arr = mean_arr[:, 0, 0][None, :]
            std_arr = std_arr[:, 0, 0][None, :]
        else:
            raise ValueError(
                f"Unsupported normalization shape for flat point cloud: mean={mean_arr.shape}, std={std_arr.shape}"
            )
    elif point_cloud.ndim == 3:
        if mean_arr.ndim == 1:
            mean_arr = mean_arr[:, None, None]
            std_arr = std_arr[:, None, None]
        elif mean_arr.ndim == 3:
            pass
        else:
            raise ValueError(
                f"Unsupported normalization shape for plane point cloud: mean={mean_arr.shape}, std={std_arr.shape}"
            )
    else:
        raise ValueError(f"Unsupported point_cloud shape for normalization: {point_cloud.shape}")

    return (point_cloud - mean_arr) / (std_arr + 1e-8)


class Standard3DGenDataset(Dataset):
    """PyTorch Dataset for 3D object generation.
    
    This dataset loads 3DGS data, captions, and optionally 2D renderings for
    training 3D generative models.
    """
    
    def __init__(
        self,
        obj_list: List[str],
        gs_path: str,
        caption_path: Optional[str] = None,
        rendering_path: Optional[str] = None,
        num_images: int = 1,
        mean_file: Optional[str] = None,
        std_file: Optional[str] = None,
        sphere2plane_path: str = "data/sphere2plane.npy",
        exclude_keys_file: Optional[str] = None,
        rank_transform_file: Optional[str] = None,
        clip_thresholds_file: Optional[str] = None,
        text_embed_path: Optional[str] = None,
        token_order: str = "plane2d",
        hilbert_bits: int = 8,
    ):
        """Initialize the dataset.
        
        Args:
            obj_list: List of paths to obj_list JSON files
            gs_path: Root path for downloaded 3DGS data
            caption_path: Path to preprocessed caption file
            rendering_path: Root path for downloaded 2D renderings (optional)
            num_images: Number of images to fetch at each step (only when rendering_path is not None)
            mean_file: Path to downloaded GS mean file
            std_file: Path to downloaded GS std file
            sphere2plane_path: Path to sphere2plane.npy permutation file
        """
        super().__init__()
        
        self.gs_path = Path(gs_path)
        self.rendering_path = Path(rendering_path) if rendering_path is not None else None
        self.num_images = num_images
        self.sphere2plane_path = Path(sphere2plane_path)
        logging.info(f"Loading sphere2plane permutation from {self.sphere2plane_path}")
        self.sphere2plane = load_sphere2plane(str(self.sphere2plane_path))
        self.point_cloud_order = "plane"

        # Splat -> grid-slot ordering. "plane2d": the canonical sphere2plane raster
        # (each 2D atlas patch is a token). "hilbert1d": sort splats along a 3D
        # Hilbert curve over their xyz so a contiguous 1D chunk is a spatially
        # compact token (the sphere2plane permutation is ignored). The (C,H,W)
        # container shape is identical either way — only the fill order differs.
        if token_order not in ("plane2d", "hilbert1d"):
            raise ValueError(f"token_order must be 'plane2d' or 'hilbert1d', got {token_order!r}")
        self.token_order = token_order
        self.hilbert_bits = int(hilbert_bits)
        
        # Load object list
        self.obj_data = load_obj_list(obj_list)

        if exclude_keys_file is not None:
            with open(exclude_keys_file, "r") as f:
                exclude_keys = set(json.load(f))
            before = len(self.obj_data)
            self.obj_data = {k: v for k, v in self.obj_data.items() if k not in exclude_keys}
            logging.info(
                f"Excluded {before - len(self.obj_data)} of {before} keys via {exclude_keys_file}"
            )

        self.keys = list(self.obj_data.keys())
        
        # Load captions
        if caption_path is not None:
            self.captions = load_captions(caption_path)
        else:
            self.captions = {}

        # Load precomputed EOS-pooled CLIP vectors (optional). When set, every
        # sample also returns the (text_dim,) pooled vector for AdaLN
        # conditioning. Keys are caption stems "chunk/file" matching
        # captions.json; we drop obj_data entries that lack an embedding so
        # __getitem__ never returns silent nulls.
        self.text_pooled: Optional[np.ndarray] = None
        self.text_tokens: Optional[np.ndarray] = None
        self.text_mask: Optional[np.ndarray] = None
        self.text_embed_key_to_row: Dict[str, int] = {}
        if text_embed_path is not None:
            (
                self.text_pooled,
                self.text_tokens,
                self.text_mask,
                self.text_embed_key_to_row,
            ) = load_text_pooled(text_embed_path)
            before = len(self.obj_data)
            self.obj_data = {
                h: p for h, p in self.obj_data.items()
                if p.split('.tar.gz')[0] in self.text_embed_key_to_row
            }
            dropped = before - len(self.obj_data)
            if dropped:
                logging.info(
                    f"Dropped {dropped} of {before} obj entries missing a text embedding"
                )
            self.keys = list(self.obj_data.keys())
        
        # Load normalization statistics if provided
        self.mean = None
        self.std = None
        if mean_file is not None and std_file is not None:
            logging.info(f"Loading normalization statistics from {mean_file} and {std_file}")
            self.mean = torch.load(mean_file).cpu().numpy().astype(np.float32)
            self.std = torch.load(std_file).cpu().numpy().astype(np.float32)
        else:
            logging.warning("Normalization is NOT enabled, mean or std file not provided.")

        # Load per-channel hard-clip thresholds if provided. Applied in __getitem__
        # right after load_ply and BEFORE the rank transform / standardize, so when
        # a channel is both clipped and rank-transformed the order is clip -> rank.
        self.clip_channels: Optional[np.ndarray] = None
        self.clip_lower: Optional[np.ndarray] = None
        self.clip_upper: Optional[np.ndarray] = None
        if clip_thresholds_file is not None:
            logging.info(f"Loading clip thresholds from {clip_thresholds_file}")
            clip_payload = load_clip_thresholds_payload(clip_thresholds_file)
            self.clip_channels = np.asarray(clip_payload["channels"], dtype=np.int64)
            self.clip_lower = clip_payload["lower"]
            self.clip_upper = clip_payload["upper"]

        # Load Gaussian rank-transform tables if provided. Channels listed in the
        # payload are mapped to N(0,1) before the mean/std standardize step; we
        # then force mean=0, std=1 on those channels so the standardize is a
        # no-op there (the tables built post-rank are exactly that by
        # construction, and it lets the rest of the pipeline stay generic).
        self.rank_channels: Optional[np.ndarray] = None
        self.rank_data_quantiles: Optional[np.ndarray] = None
        self.rank_gauss_quantiles: Optional[np.ndarray] = None
        if rank_transform_file is not None:
            logging.info(f"Loading Gaussian rank-transform tables from {rank_transform_file}")
            payload = load_rank_transform_payload(rank_transform_file)
            self.rank_channels = np.asarray(payload["channels"], dtype=np.int64)
            self.rank_data_quantiles = payload["data_quantiles"]
            self.rank_gauss_quantiles = payload["gauss_quantiles"]
            if self.mean is not None and self.std is not None:
                self.mean = self.mean.copy()
                self.std = self.std.copy()
                self.mean[self.rank_channels] = 0.0
                self.std[self.rank_channels] = 1.0
            else:
                logging.warning(
                    "rank_transform_file is set without mean/std; non-rank channels "
                    "will pass through un-standardized."
                )

        logging.info(f"Initialized dataset with {len(self.keys)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.keys)
    
    def _load_3dgs_data(self, directory_number: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load 3DGS data (point cloud and gs2sphere).
        
        Args:
            directory_number: Directory number (e.g., "1923")
            filename: Filename without extension (e.g., "9611649")
            
        Returns:
            Tuple of (point_cloud, gs2sphere) as numpy arrays in 2D plane order
            point_cloud: (C, H, W)
        """
        data_dir = self.gs_path / directory_number / filename
        
        # Load gs2sphere indices
        gs2sphere_path = data_dir / 'gs2sphere.npy'
        gs2sphere = np.load(str(gs2sphere_path))
        
        # Load point cloud
        ply_path = data_dir / 'point_cloud.ply'
        point_cloud = load_ply(str(ply_path))
        sphere2plane = self.sphere2plane
        
        # gs2sphere maps gaussian index to sphere index. Reorder to sphere order first,
        # then apply sphere2plane to get the final flat plane ordering.
        if gs2sphere.ndim != 1:
            raise ValueError(f"Expected 1D gs2sphere, got shape {gs2sphere.shape}")
        if gs2sphere.shape[0] != point_cloud.shape[0]:
            raise ValueError(
                f"Point count mismatch: point_cloud={point_cloud.shape[0]} vs gs2sphere={gs2sphere.shape[0]}"
            )
        if sphere2plane.shape[0] != point_cloud.shape[0]:
            raise ValueError(
                f"sphere2plane has {sphere2plane.shape[0]} entries, expected {point_cloud.shape[0]}"
            )

        if self.token_order == "hilbert1d":
            # Order splats along a 3D Hilbert curve over their (clamped, metric) xyz
            # (columns 0:3 of load_ply). Contiguous runs are spatially compact, so a
            # row-major reshape into (C,H,W) makes each 1D chunk a compact token. The
            # canonical lexsort/gs2sphere/sphere2plane hops are bypassed entirely.
            order = hilbert_sort_indices(point_cloud[:, :3], bits=self.hilbert_bits)
            point_cloud = point_cloud[order]
        else:
            sorted_indices = np.lexsort((point_cloud[:, 2], point_cloud[:, 1], point_cloud[:, 0]))
            point_cloud = point_cloud[sorted_indices]
            point_cloud = point_cloud[gs2sphere]
            point_cloud = point_cloud[sphere2plane]

        side = int(np.sqrt(point_cloud.shape[0]))
        point_cloud = point_cloud.reshape(side, -1, point_cloud.shape[-1])
        point_cloud = point_cloud.transpose(2, 0, 1)
        
        return point_cloud, gs2sphere
    
    def _load_renderings(self, directory_number: str, filename: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Load 2D renderings and camera parameters from tar.gz file.
        
        Args:
            directory_number: Directory number (e.g., "1923")
            filename: Filename without extension (e.g., "9611649")
            
        Returns:
            Tuple of (images, cameras):
                - images: Numpy array of shape (num_images, H, W, C) or None
                - cameras: Dictionary containing camera parameters (K, R, t) or None
        """
        if self.rendering_path is None:
            return None, None
        
        tar_path = self.rendering_path / directory_number / f"{filename}.tar.gz"
        
        if not tar_path.exists():
            logging.warning(f"Rendering tar file not found: {tar_path}")
            return None, None
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Find all available frame indices
                frame_indices = []
                for member in tar.getmembers():
                    if '.png' in member.name and 'campos_512_v1' in member.name:
                        # Extract frame index from path like 'campos_512_v1/00000/00000.png'
                        parts = member.name.split('/')
                        if len(parts) >= 3:
                            try:
                                frame_idx = int(parts[-2])
                                if frame_idx not in frame_indices:
                                    frame_indices.append(frame_idx)
                            except ValueError:
                                continue
                
                if len(frame_indices) == 0:
                    logging.warning(f"No valid frames found in {tar_path}")
                    return None, None
                
                frame_indices.sort()
                
                # Randomly select num_images frames
                if len(frame_indices) < self.num_images:
                    selected_indices = frame_indices
                else:
                    selected_indices = sorted(np.random.choice(frame_indices, self.num_images, replace=False).tolist())
                
                # Load selected frames and camera parameters
                images = []
                cameras = {
                    'K': [],      # Intrinsic matrices
                    'R': [],      # Rotation matrices
                    't': [],      # Translation vectors
                    'c2w': [],    # Camera-to-world matrices
                    'fov_x': [],  # Field of view x
                    'fov_y': [],  # Field of view y
                }
                
                for frame_idx in selected_indices:
                    # Load PNG image
                    png_member_name = f'campos_512_v1/{frame_idx:05d}/{frame_idx:05d}.png'
                    try:
                        png_member = tar.getmember(png_member_name)
                        png_data = tar.extractfile(png_member).read()
                        image = Image.open(BytesIO(png_data))
                        image_array = np.array(image)[:, :, :4]  # RGBA
                        # Convert to RGB and normalize to [0, 1]
                        image_rgb = image_array[:, :, :3].astype(np.float32) / 255.0
                        images.append(image_rgb)
                    except Exception as e:
                        logging.warning(f"Failed to load image for frame {frame_idx}: {e}")
                        continue
                    
                    # Load JSON camera parameters
                    json_member_name = f'campos_512_v1/{frame_idx:05d}/{frame_idx:05d}.json'
                    try:
                        json_member = tar.getmember(json_member_name)
                        json_data = tar.extractfile(json_member).read().decode('utf-8')
                        meta = json.loads(json_data)
                        
                        # Build camera-to-world matrix
                        c2w = np.eye(4, dtype=np.float32)
                        c2w[:3, 0] = np.array(meta['x'], dtype=np.float32)
                        c2w[:3, 1] = np.array(meta['y'], dtype=np.float32)
                        c2w[:3, 2] = np.array(meta['z'], dtype=np.float32)
                        c2w[:3, 3] = np.array(meta['origin'], dtype=np.float32)
                        
                        # Get field of view
                        fov_x = meta['x_fov']
                        fov_y = meta['y_fov']
                        
                        # Compute focal length and intrinsic matrix
                        focal = fov2focal(fov_x, 512)
                        K = np.eye(3, dtype=np.float32)
                        K[0, 0] = K[1, 1] = focal
                        K[0, 2] = K[1, 2] = 256.0
                        
                        # Get world-to-camera transform
                        w2c = np.linalg.inv(c2w)
                        R = w2c[:3, :3].astype(np.float32)
                        t = w2c[:3, 3].astype(np.float32)
                        
                        cameras['K'].append(K)
                        cameras['R'].append(R)
                        cameras['t'].append(t)
                        cameras['c2w'].append(c2w)
                        cameras['fov_x'].append(fov_x)
                        cameras['fov_y'].append(fov_y)
                        
                    except Exception as e:
                        logging.warning(f"Failed to load camera for frame {frame_idx}: {e}")
                        continue
                
                if len(images) == 0:
                    logging.warning(f"No images successfully loaded from {tar_path}")
                    return None, None
                
                # Convert lists to numpy arrays
                images_array = np.stack(images, axis=0)
                cameras_array = {
                    'K': np.stack(cameras['K'], axis=0),
                    'R': np.stack(cameras['R'], axis=0),
                    't': np.stack(cameras['t'], axis=0),
                    'c2w': np.stack(cameras['c2w'], axis=0),
                    'fov_x': np.array(cameras['fov_x'], dtype=np.float32),
                    'fov_y': np.array(cameras['fov_y'], dtype=np.float32),
                }
                
                return images_array, cameras_array
                
        except Exception as e:
            logging.error(f"Failed to load renderings from {tar_path}: {e}")
            return None, None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - point_cloud: 3DGS point cloud data (N, D)
                - caption: Text caption
                - hash_key: Unique identifier
                - images: 2D renderings (optional, if rendering_path is not None)
                - cameras: Camera parameters (optional, if rendering_path is not None)
                    - K: Intrinsic matrices (num_images, 3, 3)
                    - R: Rotation matrices (num_images, 3, 3)
                    - t: Translation vectors (num_images, 3)
                    - c2w: Camera-to-world matrices (num_images, 4, 4)
                    - fov_x: Field of view x (num_images,)
                    - fov_y: Field of view y (num_images,)
        """
        hash_key = self.keys[idx]
        tar_gz_path = self.obj_data[hash_key]
        
        # Extract directory info
        directory_number, filename = extract_directory_info(tar_gz_path)
        
        # Load 3DGS data
        point_cloud, gs2sphere = self._load_3dgs_data(directory_number, filename)

        # Hard-clip selected channels (if loaded). Must run BEFORE the rank
        # transform so the rank tables (built on the clipped stream) see the
        # same distribution.
        if self.clip_channels is not None:
            point_cloud = _apply_clip_numpy(
                point_cloud,
                self.clip_channels,
                self.clip_lower,
                self.clip_upper,
            )

        # Apply Gaussian rank transform on selected channels (if loaded)
        if self.rank_channels is not None:
            point_cloud = _apply_rank_transform_numpy(
                point_cloud,
                self.rank_channels,
                self.rank_data_quantiles,
                self.rank_gauss_quantiles,
            )

        # Normalize if enabled
        if self.mean is not None and self.std is not None:
            point_cloud = _normalize_point_cloud_numpy(point_cloud, self.mean, self.std)
        
        # Get caption (key is directory/filename without .tar.gz)
        caption_key = tar_gz_path.split('.tar.gz')[0]
        caption = self.captions.get(caption_key, "")

        # Prepare output
        sample = {
            'point_cloud': torch.from_numpy(point_cloud),
            'caption': caption,
            'hash_key': hash_key,
            'tar_gz_path': tar_gz_path,
        }

        if self.text_pooled is not None:
            row = self.text_embed_key_to_row[caption_key]
            # fp16 on disk -> fp32 here so downstream arithmetic stays accurate.
            sample['text_pooled'] = torch.from_numpy(
                self.text_pooled[row].astype(np.float32, copy=False)
            )
        
        # Load renderings and cameras if path is provided
        if self.rendering_path is not None:
            images, cameras = self._load_renderings(directory_number, filename)
            if images is not None:
                sample['images'] = torch.from_numpy(images)
            if cameras is not None:
                # Convert camera parameters to torch tensors
                sample['cameras'] = {
                    'K': torch.from_numpy(cameras['K']),
                    'R': torch.from_numpy(cameras['R']),
                    't': torch.from_numpy(cameras['t']),
                    'c2w': torch.from_numpy(cameras['c2w']),
                    'fov_x': torch.from_numpy(cameras['fov_x']),
                    'fov_y': torch.from_numpy(cameras['fov_y']),
                }
        
        return sample


def create_dataloader(
    obj_list: List[str],
    gs_path: str,
    caption_path: Optional[str] = None,
    rendering_path: Optional[str] = None,
    num_images: int = 1,
    mean_file: Optional[str] = None,
    std_file: Optional[str] = None,
    sphere2plane_path: str = "data/sphere2plane.npy",
    rank_transform_file: Optional[str] = None,
    clip_thresholds_file: Optional[str] = None,
    text_embed_path: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Create a PyTorch DataLoader for the Standard3DGen dataset.
    
    Args:
        obj_list: List of paths to obj_list JSON files
        gs_path: Root path for downloaded 3DGS data
        caption_path: Path to preprocessed caption file
        rendering_path: Root path for downloaded 2D renderings (optional)
        num_images: Number of images to fetch at each step
        mean_file: Path to downloaded GS mean file (normalization will be applied if provided)
        std_file: Path to downloaded GS std file (normalization will be applied if provided)
        sphere2plane_path: Path to sphere2plane.npy permutation file
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        PyTorch DataLoader instance
    """
    dataset = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        caption_path=caption_path,
        rendering_path=rendering_path,
        num_images=num_images,
        mean_file=mean_file,
        std_file=std_file,
        sphere2plane_path=sphere2plane_path,
        rank_transform_file=rank_transform_file,
        clip_thresholds_file=clip_thresholds_file,
        text_embed_path=text_embed_path,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    
    return dataset, dataloader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Standard3DGen DataLoader")
    parser.add_argument("--obj_list", type=str, nargs='+', required=True,
                       help="Paths to obj_list JSON files")
    parser.add_argument("--gs_path", type=str, required=True,
                       help="Path to 3DGS data directory")
    parser.add_argument("--caption_path", type=str, required=True,
                       help="Path to captions JSON file")
    parser.add_argument("--rendering_path", type=str, default=None,
                       help="Path to renderings directory")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images per sample")
    parser.add_argument("--mean_file", type=str, default=None,
                       help="Path to mean file")
    parser.add_argument("--std_file", type=str, default=None,
                       help="Path to std file")
    parser.add_argument("--sphere2plane_path", type=str, default="data/sphere2plane.npy",
                       help="Path to sphere2plane.npy permutation file")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create dataloader
    dataset,dataloader = create_dataloader(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        caption_path=args.caption_path,
        rendering_path=args.rendering_path,
        num_images=args.num_images,
        mean_file=args.mean_file,
        std_file=args.std_file,
        sphere2plane_path=args.sphere2plane_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Test loading a few batches
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
        
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Keys: {batch.keys()}")
        print(f"Point cloud shape: {batch['point_cloud'].shape}")
        print(f"Captions: {batch['caption']}")
        
        if 'images' in batch:
            print(f"Images shape: {batch['images'].shape}")
        
        if 'cameras' in batch:
            print(f"Cameras available:")
            print(f"  K shape: {batch['cameras']['K'].shape}")
            print(f"  R shape: {batch['cameras']['R'].shape}")
            print(f"  t shape: {batch['cameras']['t'].shape}")
            print(f"  c2w shape: {batch['cameras']['c2w'].shape}")
    
    print("\nDataLoader test completed successfully!")
