"""Precompute per-(sample, camera) render-camera zoom factors for the render loss.

For every object in the dataset and every reference camera, this projects the object's
3D Gaussian centers through the camera at the BASE FOV (zoom=1.0, train_render_size=224)
and computes the tightest zoom factor that scales the projected bbox to fill
``target_bbox_fill`` (default 0.85) of the frame width. The result is consumed at training
time by ``utils.gsplat_render_util._render_gsplat_batch`` to scale per-batch-element focal
lengths, giving every render a tight per-object framing instead of the current ~26%
bbox_fill at uniform FOV (= ~74% of every L1/LPIPS pixel being trivially-correct background).

Output: a single ``.pt`` at ``data/stats/per_sample_render_fov.pt`` with:
  - ``hash_keys``: list[str], length N, in the dataset's iteration order
  - ``zoom_factors``: torch.Tensor (N, num_cameras) float32
  - ``meta``: dict of build params for reproducibility / staleness checks

Implementation: multiprocessing pool. Each worker holds a ``Standard3DGenDataset`` (raw
mode — no mean/std, no rank, no clip; we want WORLD COORDINATES) and a (R, T) array for
all reference cameras + the base K. Per object: load atlas → reshape to (N, 3) world xyz
+ (N,) opacity logits → for all 52 cameras at once via batched matmul, project + filter +
compute max-distance-from-image-center → derive zoom = (target_fill * W/2) / max_dist.

Run from repo root with the project venv activated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import Standard3DGenDataset
from utils.gsplat_render_util import (
    _camera_intrinsics_from_ref,
    _camera_viewmat_from_ref,
    _fov2focal,
    _load_reference_cameras,
)


# Atlas channel layout. Matches Standard3DGenDataset raw output: (C=59, H=128, W=128).
XYZ_CHANNELS = (0, 1, 2)
OPACITY_CHANNEL = 3


# --- Worker globals (one set per worker process) ---
_DATASET: Optional[Standard3DGenDataset] = None
_R_T: Optional[np.ndarray] = None             # (num_cam, 3, 4) [R | T]
_K_BASE: Optional[np.ndarray] = None          # (3, 3)
_IMG_SIZE: int = 0
_TARGET_FILL: float = 0.0
_OPACITY_LOGIT_MIN: float = 0.0
_ZOOM_MIN: float = 1.0
_ZOOM_MAX: float = 5.0


def _worker_init(obj_list, gs_path, sphere2plane_path, exclude_keys_file,
                 r_t_arr, k_base_arr, img_size,
                 target_fill, opacity_logit_min, zoom_min, zoom_max):
    global _DATASET, _R_T, _K_BASE, _IMG_SIZE, _TARGET_FILL
    global _OPACITY_LOGIT_MIN, _ZOOM_MIN, _ZOOM_MAX

    logging.getLogger().setLevel(logging.WARNING)
    # Raw load: no mean/std, no rank, no clip. We want WORLD COORDINATES for projection.
    # xyz channels (0:3) are not touched by clip or rank transforms in this project.
    _DATASET = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        mean_file=None,
        std_file=None,
        sphere2plane_path=sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )
    _R_T = r_t_arr.astype(np.float32, copy=False)        # (num_cam, 3, 4)
    _K_BASE = k_base_arr.astype(np.float32, copy=False)   # (3, 3)
    _IMG_SIZE = int(img_size)
    _TARGET_FILL = float(target_fill)
    _OPACITY_LOGIT_MIN = float(opacity_logit_min)
    _ZOOM_MIN = float(zoom_min)
    _ZOOM_MAX = float(zoom_max)


def _project_and_compute_zooms(xyz: np.ndarray) -> np.ndarray:
    """xyz: (N, 3) world coords (opacity-filtered).  Returns (num_cam,) zoom factors.

    Vectorized across all cameras. Camera back-of-frame points (z<=0) are dropped per
    camera. If no points remain for a camera, zoom defaults to 1.0 (no change)."""
    num_cam = int(_R_T.shape[0])
    img_half = _IMG_SIZE / 2.0
    fx = float(_K_BASE[0, 0])
    fy = float(_K_BASE[1, 1])
    cx = float(_K_BASE[0, 2])
    cy = float(_K_BASE[1, 2])

    # Homogeneous coords: (N, 4) with trailing 1.
    n = xyz.shape[0]
    xyzw = np.ones((n, 4), dtype=np.float32)
    xyzw[:, :3] = xyz
    # Apply each camera's [R | T]: (num_cam, 3, 4) @ (4, N) -> (num_cam, 3, N).
    cam_coords = np.matmul(_R_T, xyzw.T)                 # (num_cam, 3, N)
    z = cam_coords[:, 2, :]                              # (num_cam, N)
    in_front = z > 1e-4                                  # (num_cam, N)
    # Avoid divide-by-zero on points-behind by stuffing them with z=1 (the mask drops them).
    safe_z = np.where(in_front, z, np.ones_like(z))
    u = fx * cam_coords[:, 0, :] / safe_z + cx           # (num_cam, N)
    v = fy * cam_coords[:, 1, :] / safe_z + cy           # (num_cam, N)
    # Max distance from image center to any valid projected point. INF for invalid
    # points so they don't contribute to the max.
    dx = np.where(in_front, np.abs(u - img_half), np.full_like(u, -np.inf))
    dy = np.where(in_front, np.abs(v - img_half), np.full_like(v, -np.inf))
    max_dist = np.maximum(dx.max(axis=1), dy.max(axis=1))  # (num_cam,)
    has_points = np.isfinite(max_dist) & (max_dist > 0)
    # zoom = (target_fill * W/2) / max_dist;  default 1.0 where no valid points.
    safe_max = np.where(has_points, max_dist, np.ones_like(max_dist))
    zoom = (_TARGET_FILL * img_half) / safe_max
    zoom = np.where(has_points, zoom, np.ones_like(zoom))
    zoom = np.clip(zoom, _ZOOM_MIN, _ZOOM_MAX).astype(np.float32)
    return zoom


def _worker_fn(idx: int):
    hash_key = _DATASET.keys[idx]
    sample = _DATASET[idx]
    pc = sample["point_cloud"]
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()
    pc = pc.astype(np.float32)
    if pc.ndim != 3:
        raise RuntimeError(f"expected (C, H, W) point cloud, got shape {pc.shape}")
    # Reshape to (N=H*W, C). The renderer's sphere2plane ordering doesn't matter for
    # bbox computation (the projected SET is reordering-invariant), so we skip it.
    C, H, W = pc.shape
    pc_flat = pc.reshape(C, H * W).T                   # (N, C)
    xyz = pc_flat[:, list(XYZ_CHANNELS)]               # (N, 3) world coords
    opacity_logit = pc_flat[:, OPACITY_CHANNEL]        # (N,)
    live = opacity_logit > _OPACITY_LOGIT_MIN
    if not live.any():
        # Totally-transparent atlas: keep zooms at 1.0 for safety.
        zooms = np.ones(int(_R_T.shape[0]), dtype=np.float32)
    else:
        zooms = _project_and_compute_zooms(xyz[live])
    return idx, hash_key, zooms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obj_list", required=True, nargs="+")
    p.add_argument("--gs_path", required=True)
    p.add_argument("--sphere2plane_path", required=True)
    p.add_argument("--ref_camera_tar", required=True,
                   help="Path to ref_camera.tar.gz (same one used by training).")
    p.add_argument("--exclude_keys_file", default="data/outlier_keys_8sigma.json",
                   help="JSON list of hash_keys to skip (set to '' to disable).")
    p.add_argument("--output", default="data/stats/per_sample_render_fov.pt",
                   help="Output .pt path.")
    p.add_argument("--train_render_size", type=int, default=224,
                   help="Base render resolution (matches the training config). Zoom is "
                        "computed at this size; at inference time it scales consistently.")
    p.add_argument("--target_bbox_fill", type=float, default=0.85,
                   help="Fraction of the frame the object's tight bbox should occupy "
                        "after zoom.  0.85 = 7.5%% margin per side as safety.")
    p.add_argument("--opacity_min", type=float, default=0.05,
                   help="Minimum (post-sigmoid) opacity to consider a Gaussian alive. "
                        "Below this it's treated as transparent and ignored for the bbox.")
    p.add_argument("--zoom_min", type=float, default=1.0,
                   help="Lower clamp on the per-(sample, camera) zoom factor.")
    p.add_argument("--zoom_max", type=float, default=5.0,
                   help="Upper clamp; prevents pathological tiny-object cases from being "
                        "zoomed past LPIPS's meaningful scale.")
    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on objects scanned (testing only).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    exclude_keys_file = args.exclude_keys_file or None

    # Build the dataset once on main to know N and the hash order.
    ds = Standard3DGenDataset(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        mean_file=None, std_file=None,
        sphere2plane_path=args.sphere2plane_path,
        exclude_keys_file=exclude_keys_file,
    )
    n_total = len(ds)
    n = n_total if args.limit is None else min(args.limit, n_total)

    # Load reference cameras + build the base K matrix (zoom=1.0).
    ref_cams = _load_reference_cameras(args.ref_camera_tar)
    num_cam = len(ref_cams)
    # (num_cam, 3, 4): [R | T] per camera.
    r_t = np.zeros((num_cam, 3, 4), dtype=np.float32)
    for c, ref_cam in enumerate(ref_cams):
        viewmat = _camera_viewmat_from_ref(ref_cam).cpu().numpy()  # (4, 4) w2c
        r_t[c, :, :3] = viewmat[:3, :3]
        r_t[c, :, 3] = viewmat[:3, 3]
    # K is the same shape across cameras here (square renders, single fov per cam).
    # Use cam 0 as the prototype — confirm all cameras agree on fov/size, then build K.
    fovx = float(ref_cams[0]["fovx"]); fovy = float(ref_cams[0]["fovy"])
    for ref_cam in ref_cams[1:]:
        if abs(float(ref_cam["fovx"]) - fovx) > 1e-6 or abs(float(ref_cam["fovy"]) - fovy) > 1e-6:
            raise ValueError("Reference cameras have non-uniform FOV; precompute assumes "
                             "uniform FOV across cameras. Fix or extend the script.")
    fx = _fov2focal(fovx, args.train_render_size)
    fy = _fov2focal(fovy, args.train_render_size)
    k_base = np.array([
        [fx, 0.0, args.train_render_size / 2.0],
        [0.0, fy, args.train_render_size / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    opacity_logit_min = float(math.log(args.opacity_min / (1.0 - args.opacity_min)))
    print(f"Building per-sample render zoom for {n:,} objects × {num_cam} cameras")
    print(f"  base FOV: {math.degrees(fovx):.2f}° ({fovx:.4f} rad), render size {args.train_render_size}")
    print(f"  target_bbox_fill={args.target_bbox_fill}, zoom range [{args.zoom_min}, {args.zoom_max}]")
    print(f"  opacity_min={args.opacity_min} (logit > {opacity_logit_min:.3f})")
    print(f"  output: {args.output}")
    print(f"  workers: {args.num_workers}")
    if exclude_keys_file:
        print(f"  excluding objects listed in: {exclude_keys_file}")

    # Pre-allocate the (N, num_cam) zoom tensor and a hash list (filled in dataset order).
    zoom_factors = np.ones((n, num_cam), dtype=np.float32)
    hash_keys: list[Optional[str]] = [None] * n

    indices = list(range(n))
    t0 = time.time()
    log_every = max(1, n // 100)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_workers,
        initializer=_worker_init,
        initargs=(args.obj_list, args.gs_path, args.sphere2plane_path, exclude_keys_file,
                  r_t, k_base, args.train_render_size,
                  args.target_bbox_fill, opacity_logit_min, args.zoom_min, args.zoom_max),
    ) as pool:
        for done, (idx, hash_key, zooms) in enumerate(
            pool.imap_unordered(_worker_fn, indices, chunksize=8), 1
        ):
            zoom_factors[idx] = zooms
            hash_keys[idx] = hash_key
            if done % log_every == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                running_mean = zoom_factors[:done].mean()
                running_p99 = float(np.quantile(zoom_factors[:done].max(axis=1), 0.99))
                print(
                    f"  {done:>7d}/{n}  {rate:>5.1f} obj/s  eta {eta / 60:>5.1f} min  "
                    f"running mean zoom={running_mean:.3f}  p99 per-obj-max={running_p99:.2f}"
                )

    if any(h is None for h in hash_keys):
        missing = sum(1 for h in hash_keys if h is None)
        raise RuntimeError(f"{missing} objects had no result returned from the pool")

    # --- Stats + save ---
    z_flat = zoom_factors.reshape(-1)
    print("\n=== Done. Zoom-factor distribution (over all sample×camera pairs) ===")
    for q in (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(q*100):>2d}: {np.quantile(z_flat, q):.3f}")
    print(f"  min: {z_flat.min():.3f}   mean: {z_flat.mean():.3f}   max: {z_flat.max():.3f}")
    n_at_min = int((z_flat <= args.zoom_min + 1e-6).sum())
    n_at_max = int((z_flat >= args.zoom_max - 1e-6).sum())
    print(f"  at-min ({args.zoom_min}): {n_at_min:,} pairs ({100*n_at_min/z_flat.size:.2f}%)")
    print(f"  at-max ({args.zoom_max}): {n_at_max:,} pairs ({100*n_at_max/z_flat.size:.2f}%)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hash_keys": list(hash_keys),
        "zoom_factors": torch.from_numpy(zoom_factors),
        "meta": {
            "ref_camera_tar": str(args.ref_camera_tar),
            "train_render_size": int(args.train_render_size),
            "target_bbox_fill": float(args.target_bbox_fill),
            "opacity_min": float(args.opacity_min),
            "zoom_min": float(args.zoom_min),
            "zoom_max": float(args.zoom_max),
            "num_objects": int(n),
            "num_cameras": int(num_cam),
            "base_fovx_rad": float(fovx),
            "base_fovy_rad": float(fovy),
            "exclude_keys_file": str(args.exclude_keys_file or ""),
            "built_with": "data/build_per_sample_render_fov.py",
        },
    }
    torch.save(payload, out_path)
    print(f"\nSaved {n:,} × {num_cam} zoom factors -> {out_path}  "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    # Also write a tiny JSON twin of the meta for quick inspection without loading the .pt.
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(payload["meta"], f, indent=2)
    print(f"Meta JSON: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
