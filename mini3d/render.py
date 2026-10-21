"""Render Gaussian containers for training losses, sampling, and evaluation.

This module assembles cameras, normalization tables, and optional LPIPS.
Hilbert-ordered containers flatten directly; plane layouts use the stored permutation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import TrainConfig

from utils.gsplat_render_util import (
    _denormalize_point_cloud,
    _load_reference_cameras,
    _plane_to_point_cloud_batch,
    _point_clouds_to_gsplat_inputs,
    _prepare_train_cameras,
    _render_gsplat_batch,
    _try_import_lpips,
    _try_import_renderer,
    load_rank_transform_payload_torch,
)
from dataloaders.standard_3dgen_loader import load_sphere2plane

logger = logging.getLogger(__name__)


@dataclass
class RenderContext:
    device: torch.device
    train_cameras: dict[str, Any]
    renderer_tuple: Any
    lpips_fn: Any = None
    norm_mean: torch.Tensor | None = None
    norm_std: torch.Tensor | None = None
    norm_mean_full: torch.Tensor | None = None
    norm_std_full: torch.Tensor | None = None
    rank_transform_tables: dict | None = None
    plane_to_sphere: torch.Tensor | None = None
    dc_only: bool = True
    num_cameras: int = 0
    zoom_table: torch.Tensor | None = None        # (N_obj, num_cam)
    zoom_row: dict = field(default_factory=dict)  # hash_key -> row
    meta: dict = field(default_factory=dict)

    def zooms_for(self, keys: list[str]) -> torch.Tensor | None:
        """(B, num_cam) per-sample zoom factors, or None when disabled.

        Missing keys fall back to 1.0 (base FOV) rather than raising: the zoom
        table covers the non-outlier objects, and a KeyError mid-training would
        kill a 48h job over a framing detail.
        """
        if self.zoom_table is None:
            return None
        rows = [self.zoom_row.get(k, -1) for k in keys]
        out = torch.ones((len(keys), self.zoom_table.shape[1]),
                         device=self.zoom_table.device, dtype=self.zoom_table.dtype)
        have = [i for i, r in enumerate(rows) if r >= 0]
        if have:
            idx = torch.tensor([rows[i] for i in have], device=self.zoom_table.device)
            out[torch.tensor(have, device=self.zoom_table.device)] = \
                self.zoom_table.index_select(0, idx)
        return out


def build_render_context(cfg: TrainConfig, device: torch.device, *,
                         want_lpips: bool = False) -> RenderContext:
    renderer_tuple = _try_import_renderer()
    if renderer_tuple is None or isinstance(renderer_tuple, Exception):
        raise RuntimeError(
            "gsplat is not importable, so nothing can be rendered. Validation "
            f"grids and CMMD both depend on it. Import error: {renderer_tuple}"
        )
    ref_cameras = _load_reference_cameras(cfg.resolved("ref_camera_tar"))
    train_cameras = _prepare_train_cameras(ref_cameras, cfg.train_render_size, device)

    rank_tables = load_rank_transform_payload_torch(cfg.resolved("rank_transform_file"),
                                                    device=device)
    norm_mean_full = torch.load(cfg.resolved("mean_file"), weights_only=True).float().to(device)
    norm_std_full = torch.load(cfg.resolved("std_file"), weights_only=True).float().to(device)
    if rank_tables is not None:
        # Rank-transformed channels are already N(0,1); force their stats to (0,1)
        # so the destandardize is a no-op and the inverse rank transform owns the
        # whole round-trip. Applying both would double-transform them.
        idx = torch.tensor(rank_tables["channels"], dtype=torch.long, device=device)
        norm_mean_full = norm_mean_full.clone(); norm_std_full = norm_std_full.clone()
        norm_mean_full[idx] = 0.0
        norm_std_full[idx] = 1.0

    if cfg.sh_degree0_only:
        from dataloaders.text_3dgen_loader import DC_ONLY_FEATURE_INDICES
        fi = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long, device=device)
        norm_mean, norm_std = norm_mean_full[fi], norm_std_full[fi]
    else:
        norm_mean, norm_std = norm_mean_full, norm_std_full

    plane_to_sphere = None
    if cfg.token_order == "hilbert1d":
        logger.info("token_order=hilbert1d: render uses flat container order (no sphere2plane)")
    else:
        plane_to_sphere = load_sphere2plane(cfg.resolved("sphere2plane_path"),
                                            cfg.container_size ** 2)

    lpips_fn = None
    if want_lpips:
        # _try_import_lpips() returns the MODULE on success and the EXCEPTION on
        # failure -- never None. So a `is None` check both (a) passes the bare
        # module through as if it were a network, which then raises when the loss
        # calls it, and (b) treats an import failure as success. Mirror the
        # playground trainer: type-check, then instantiate the network.
        probe = _try_import_lpips()
        if isinstance(probe, Exception):
            raise RuntimeError(f"LPIPS is required by this configuration: {probe}") from probe
        else:
            # spatial=True is MANDATORY when fg_phi > 0: the coverage weighting
            # needs a per-pixel LPIPS map to apply the foreground weight to, and
            # the loss raises rather than silently averaging if it gets a scalar.
            spatial = float(cfg.render_fg_phi) > 0.0
            lpips_fn = probe.LPIPS(net=cfg.lpips_net, spatial=spatial).to(device).eval()
            for prm in lpips_fn.parameters():
                prm.requires_grad_(False)
            logger.info("LPIPS(net=%s, spatial=%s) ready", cfg.lpips_net, spatial)

    zoom_table, zoom_row = None, {}
    if cfg.use_per_sample_zoom:
        payload = torch.load(cfg.resolved("per_sample_zoom_file"), weights_only=False)
        zoom_table = payload["zoom_factors"].to(device).float()
        zoom_row = {k: i for i, k in enumerate(payload["hash_keys"])}
        n_cam = int(train_cameras["viewmats"].shape[0])
        if zoom_table.shape[1] != n_cam:
            raise ValueError(
                f"zoom table has {zoom_table.shape[1]} cameras but the camera bundle has "
                f"{n_cam}; the zoom factors are indexed BY CAMERA, so a mismatch silently "
                "applies the wrong framing."
            )
        logger.info("per-sample zoom: %d objects x %d cameras", *zoom_table.shape)

    return RenderContext(
        zoom_table=zoom_table, zoom_row=zoom_row,
        device=device, train_cameras=train_cameras, renderer_tuple=renderer_tuple,
        lpips_fn=lpips_fn, norm_mean=norm_mean, norm_std=norm_std,
        norm_mean_full=norm_mean_full, norm_std_full=norm_std_full,
        rank_transform_tables=rank_tables, plane_to_sphere=plane_to_sphere,
        dc_only=cfg.sh_degree0_only,
        num_cameras=int(train_cameras["viewmats"].shape[0]),
        meta={"train_render_size": cfg.train_render_size},
    )


@torch.no_grad()
def render_containers(ctx: RenderContext, x: torch.Tensor, cam_indices: list[int] | None = None,
                      *, full_channels: bool = False,
                      keys: list[str] | None = None) -> torch.Tensor:
    """Render normalized containers ``(B, C, H, W)`` -> ``(B, V, H, W, 3)`` in [0,1].

    NOTE the layout conversion. ``_render_gsplat_batch`` returns CHANNELS-FIRST
    ``(B, V, 3, H, W)``; every consumer here (grid saving, CLIP features for CMMD)
    wants channels-last. Converting once, here, keeps that contract in one place --
    a channels-first array reaching PIL silently produces a garbage image rather
    than an error, because 3xH still looks like a valid 2-D shape.

    ``full_channels`` selects the 59-channel norm stats (for ground-truth
    containers) instead of the 14-channel DC-only ones.
    """
    cam_indices = cam_indices or [0]
    mean = ctx.norm_mean_full if full_channels else ctx.norm_mean
    std = ctx.norm_std_full if full_channels else ctx.norm_std
    pc = _plane_to_point_cloud_batch(x.float(), ctx.plane_to_sphere)
    pc = _denormalize_point_cloud(pc, mean, std)
    is_dc = pc.shape[-1] == 14
    gaussians = _point_clouds_to_gsplat_inputs(
        pc.to(ctx.device), dc_only=is_dc, detach_input=True,
        rank_transform_tables=ctx.rank_transform_tables,
    )
    if ctx.dc_only and not is_dc:
        gaussians = {**gaussians, "colors": gaussians["colors"][..., :1, :], "sh_degree": 0}
    zooms = None
    if keys is not None:
        z = ctx.zooms_for(keys)
        if z is not None:
            zooms = z.index_select(1, torch.tensor(cam_indices, device=z.device))
    out = _render_gsplat_batch(ctx.renderer_tuple, gaussians, ctx.train_cameras,
                               cam_indices, ctx.device, per_sample_zooms=zooms)
    return out.permute(0, 1, 3, 4, 2).contiguous()          # (B,V,3,H,W) -> (B,V,H,W,3)


def save_grid(images: torch.Tensor, path: str, nrow: int = 4) -> None:
    """Save ``(N, H, W, 3)`` or ``(N, V, H, W, 3)`` in [0,1] as a PNG grid.

    Channels-first input is detected and converted, so a caller that forgets to
    permute gets a correct image rather than a scrambled one.
    """
    from pathlib import Path

    import numpy as np
    from PIL import Image

    x = images.detach().float().clamp(0, 1).cpu()
    if x.ndim == 5:
        x = x.reshape(-1, *x.shape[2:])
    if x.ndim == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        x = x.permute(0, 2, 3, 1)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"save_grid expects (...,H,W,3); got {tuple(images.shape)}")
    n, h, w, _ = x.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    canvas = np.zeros((nrows * h, ncol * w, 3), dtype=np.float32)
    arr = x.numpy()
    for i in range(n):
        r, c = divmod(i, ncol)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = arr[i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((canvas * 255).round().astype(np.uint8)).save(path)
