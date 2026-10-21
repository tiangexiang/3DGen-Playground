"""Shared gsplat rendering helpers used across training, inference, and eval scripts."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import tarfile
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dataloaders.text_3dgen_loader import DC_ONLY_FEATURE_INDICES

logger = logging.getLogger(__name__)

# Cache the inverse of the sphere-to-plane permutation so _plane_to_point_cloud_batch
# doesn't recompute it on every call during render-loss steps.
# Keyed by (tensor data_ptr, device_str); the permutation tensor is held alive by
# the caller (train_cameras / module state) for the full training run.
_SPHERE_TO_PLANE_INV_CACHE: dict = {}

RENDER_OPACITY_RAW_MIN = -12.0
RENDER_OPACITY_RAW_MAX = 12.0
RENDER_SCALE_RAW_MIN = -12.0
RENDER_SCALE_RAW_MAX = 8.0
RENDER_QUAT_EPS = 1e-8

# Clamp on per-sample GT coverage before it becomes a divisor in the fg_phi reweighting.
# Bounds chosen from the measured distribution (jit/measure_render_coverage.py, 1,024 objects):
# p05 = 0.053, p95 = 0.465. The floor caps the foreground upweight at phi/0.04 (12.5x at
# phi=0.5) so a near-empty or degenerate render cannot dominate the batch gradient; the ceiling
# keeps very large objects from being down-weighted below their plain contribution.
_FG_PHI_COV_MIN = 0.04
_FG_PHI_COV_MAX = 0.70


def _try_import_renderer():
    try:
        import gsplat
        return gsplat
    except Exception as exc:  # pragma: no cover - runtime dependency probe
        return exc


def _try_import_lpips():
    try:
        import lpips
        return lpips
    except Exception as exc:  # pragma: no cover - runtime dependency probe
        return exc


def _fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))


def _load_reference_cameras(ref_camera_tar: str) -> list[dict[str, Any]]:
    cams = []
    with tarfile.open(ref_camera_tar, "r:gz") as tar:
        json_members = [member for member in tar.getmembers() if member.name.endswith(".json")]
        json_members.sort(key=lambda member: member.name)
        for member in json_members:
            meta = json.loads(tar.extractfile(member).read().decode("utf-8"))
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = np.array(meta["x"], dtype=np.float32)
            c2w[:3, 1] = np.array(meta["y"], dtype=np.float32)
            c2w[:3, 2] = np.array(meta["z"], dtype=np.float32)
            c2w[:3, 3] = np.array(meta["origin"], dtype=np.float32)
            w2c = np.linalg.inv(c2w).astype(np.float32)
            cams.append(
                {
                    "R": w2c[:3, :3],
                    "T": w2c[:3, 3],
                    "fovx": float(meta["x_fov"]),
                    "fovy": float(meta["y_fov"]),
                    "width": int(meta.get("width", 512)),
                    "height": int(meta.get("height", 512)),
                }
            )
    if not cams:
        raise ValueError(f"No camera json found in {ref_camera_tar}")
    return cams


def _camera_viewmat_from_ref(ref_cam: dict[str, Any]) -> torch.Tensor:
    """Build a world-to-camera matrix matching the legacy renderer contract."""
    viewmat = np.zeros((4, 4), dtype=np.float32)
    viewmat[:3, :3] = np.asarray(ref_cam["R"], dtype=np.float32)
    viewmat[:3, 3] = np.asarray(ref_cam["T"], dtype=np.float32)
    viewmat[3, 3] = 1.0
    return torch.from_numpy(viewmat)


def _camera_intrinsics_from_ref(ref_cam: dict[str, Any]) -> torch.Tensor:
    width = int(ref_cam["width"])
    height = int(ref_cam["height"])
    fx = _fov2focal(float(ref_cam["fovx"]), width)
    fy = _fov2focal(float(ref_cam["fovy"]), height)
    return torch.tensor(
        [
            [fx, 0.0, width * 0.5],
            [0.0, fy, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _prepare_train_cameras(
    ref_cameras: list[dict[str, Any]],
    train_render_size: int,
    device: torch.device,
    zoom_factor: float = 1.0,
) -> dict[str, Any]:
    """Build the camera bundle for training renders.

    ``zoom_factor > 1`` narrows the effective FOV (equivalent to increasing focal length
    by the same factor) so the object fills more of the frame — fixes the situation where
    most rendered pixels are background and dilute the per-pixel L1/LPIPS loss-mean. View
    matrices are untouched (camera doesn't move); only the projection cone narrows. Mapping:
    new_fov/2 = atan(tan(old_fov/2) / zoom). 1.0 = no change (back-compat default).
    """
    if zoom_factor <= 0:
        raise ValueError(f"zoom_factor must be positive, got {zoom_factor}")
    viewmats = []
    intrinsics = []
    for ref_cam in ref_cameras:
        ref_small = dict(ref_cam)
        ref_small["width"] = int(train_render_size)
        ref_small["height"] = int(train_render_size)
        if zoom_factor != 1.0:
            ref_small["fovx"] = 2.0 * math.atan(math.tan(float(ref_cam["fovx"]) / 2.0) / zoom_factor)
            ref_small["fovy"] = 2.0 * math.atan(math.tan(float(ref_cam["fovy"]) / 2.0) / zoom_factor)
        viewmats.append(_camera_viewmat_from_ref(ref_small))
        intrinsics.append(_camera_intrinsics_from_ref(ref_small))
    return {
        "viewmats": torch.stack(viewmats, dim=0).to(device=device),
        "Ks": torch.stack(intrinsics, dim=0).to(device=device),
        "width": int(train_render_size),
        "height": int(train_render_size),
    }


def _plane_to_point_cloud_batch(
    planes: torch.Tensor,
    plane_to_sphere: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert plane grids (B, D, H, W) to flat point clouds (B, N, D)."""
    if planes.ndim == 3:
        planes = planes.unsqueeze(0)
    if planes.ndim != 4:
        raise ValueError(f"Expected 3D or 4D plane tensor, got shape {tuple(planes.shape)}")

    batch, channels, height, width = planes.shape
    num_points = height * width
    flat = planes.permute(0, 2, 3, 1).reshape(batch, num_points, channels)
    if plane_to_sphere is None:
        return flat
    cache_key = (plane_to_sphere.data_ptr(), str(planes.device))
    sphere_to_plane = _SPHERE_TO_PLANE_INV_CACHE.get(cache_key)
    if sphere_to_plane is None:
        perm = plane_to_sphere.to(device=planes.device)
        sphere_to_plane = torch.empty_like(perm)
        sphere_to_plane[perm] = torch.arange(num_points, device=planes.device, dtype=perm.dtype)
        _SPHERE_TO_PLANE_INV_CACHE[cache_key] = sphere_to_plane
    return flat.index_select(1, sphere_to_plane)


def load_rank_transform_payload_torch(
    path: str,
    device: Optional[torch.device] = None,
) -> Optional[dict]:
    """Load a Gaussian rank-transform payload as torch tensors for the inverse path.

    Returns a dict with `channels` (list[int], full 59-channel feature space),
    `data_quantiles` (C, K) and `gauss_quantiles` (K,) as float32 tensors.
    Returns None if `path` is None.
    """
    if path is None:
        return None
    payload = torch.load(path, weights_only=False)
    data_q = payload["data_quantiles"]
    gauss_q = payload["gauss_quantiles"]
    if not isinstance(data_q, torch.Tensor):
        data_q = torch.as_tensor(data_q)
    if not isinstance(gauss_q, torch.Tensor):
        gauss_q = torch.as_tensor(gauss_q)
    data_q = data_q.float()
    gauss_q = gauss_q.float()
    if device is not None:
        data_q = data_q.to(device=device)
        gauss_q = gauss_q.to(device=device)
    return {
        "channels": [int(c) for c in payload["channels"]],
        "data_quantiles": data_q,
        "gauss_quantiles": gauss_q,
    }


def _resolve_rank_layout_for_active_space(
    rank_full_channels: list[int],
    dc_only: bool,
) -> list[tuple[int, int]]:
    """Return [(active_idx, row_idx)] pairs locating each rank channel in the
    active feature layout (full or DC-only) used by the renderer."""
    if not dc_only:
        return [(c, i) for i, c in enumerate(rank_full_channels)]
    full_to_dc = {full: pos for pos, full in enumerate(DC_ONLY_FEATURE_INDICES)}
    layout = []
    for i, c in enumerate(rank_full_channels):
        if c not in full_to_dc:
            raise ValueError(
                f"Rank-transform channel {c} is absent from the DC-only feature set; "
                f"rebuild rank_quantiles.pt with channels intersecting "
                f"DC_ONLY_FEATURE_INDICES = {DC_ONLY_FEATURE_INDICES}"
            )
        layout.append((full_to_dc[c], i))
    return layout


def _apply_rank_inverse_for_render(
    point_cloud: torch.Tensor,
    layout: list[tuple[int, int]],
    gauss_q: torch.Tensor,
    data_q: torch.Tensor,
) -> torch.Tensor:
    """Inverse Gaussian rank transform: N(0,1) -> data domain via 1D linear interp.

    Operates on the trailing feature dim. `layout` lists which active-space
    channel indices to update and which row of `data_q` provides their breakpoints.
    Values outside the empirical gauss range are saturated to the table endpoints.
    """
    if not layout:
        return point_cloud
    K = gauss_q.shape[0]
    g_min = gauss_q[0]
    g_max = gauss_q[-1]
    # Functional rebuild: replace per-channel columns and re-stack. Avoids
    # in-place setitem on a clone, which would bump the clone's version and
    # invalidate prior-iteration views captured for backward.
    columns = list(point_cloud.unbind(dim=-1))
    for active_idx, row_idx in layout:
        zc = columns[active_idx]
        zc_clamped = torch.clamp(zc, min=g_min, max=g_max)
        idx = torch.searchsorted(gauss_q, zc_clamped.contiguous())
        idx = idx.clamp(min=1, max=K - 1)
        lo = idx - 1
        hi = idx
        g_lo = gauss_q[lo]
        g_hi = gauss_q[hi]
        d_row = data_q[row_idx]
        d_lo = d_row[lo]
        d_hi = d_row[hi]
        denom = (g_hi - g_lo).clamp_min(1e-12)
        t = (zc_clamped - g_lo) / denom
        columns[active_idx] = d_lo + t * (d_hi - d_lo)
    return torch.stack(columns, dim=-1)


def _normalize_quaternions_with_identity_fallback(rotations_raw: torch.Tensor) -> torch.Tensor:
    quat_norms = rotations_raw.norm(dim=-1, keepdim=True)
    quats = rotations_raw / quat_norms.clamp_min(RENDER_QUAT_EPS)
    identity_quat = torch.zeros_like(quats)
    identity_quat[..., 0] = 1.0
    return torch.where(quat_norms > RENDER_QUAT_EPS, quats, identity_quat)


def _denormalize_point_cloud(
    point_cloud: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
) -> torch.Tensor:
    if mean is None or std is None:
        return point_cloud
    mean_t = mean.to(device=point_cloud.device, dtype=point_cloud.dtype)
    std_t = std.to(device=point_cloud.device, dtype=point_cloud.dtype)
    expand_shape = (1,) * (point_cloud.ndim - 1) + (point_cloud.shape[-1],)
    return point_cloud * (std_t.view(expand_shape) + 1e-8) + mean_t.view(expand_shape)


def _constrain_denormalized_point_cloud_for_render(
    point_cloud: torch.Tensor,
    *,
    dc_only: bool = False,
    rank_transform_tables: Optional[dict] = None,
) -> torch.Tensor:
    """Project denormalized predictions into canonical 3DGS parameter space for rendering.

    When `rank_transform_tables` is provided, channels listed in the tables are
    inverse-rank-transformed (N(0,1) -> data domain) before the sigmoid/exp
    activations apply. The tables' channel indices are interpreted in the full
    59-channel feature space and remapped to the active layout via
    `_resolve_rank_layout_for_active_space`.
    """
    if dc_only:
        scale_slice = slice(7, 10)
        rotation_slice = slice(10, 14)
    else:
        scale_slice = slice(52, 55)
        rotation_slice = slice(55, 59)

    safe_point_cloud = torch.nan_to_num(point_cloud, nan=0.0, posinf=0.0, neginf=0.0)
    if rank_transform_tables is not None:
        layout = _resolve_rank_layout_for_active_space(
            rank_transform_tables["channels"], dc_only=dc_only
        )
        gauss_q = rank_transform_tables["gauss_quantiles"].to(
            device=safe_point_cloud.device, dtype=safe_point_cloud.dtype
        )
        data_q = rank_transform_tables["data_quantiles"].to(
            device=safe_point_cloud.device, dtype=safe_point_cloud.dtype
        )
        safe_point_cloud = _apply_rank_inverse_for_render(
            safe_point_cloud, layout, gauss_q, data_q
        )
    xyz = safe_point_cloud[..., :3]
    opacity = torch.sigmoid(
        safe_point_cloud[..., 3:4].clamp(RENDER_OPACITY_RAW_MIN, RENDER_OPACITY_RAW_MAX)
    )
    middle = safe_point_cloud[..., 4:scale_slice.start]
    scales = torch.exp(
        safe_point_cloud[..., scale_slice].clamp(RENDER_SCALE_RAW_MIN, RENDER_SCALE_RAW_MAX)
    )
    rotations = _normalize_quaternions_with_identity_fallback(
        safe_point_cloud[..., rotation_slice]
    )
    tail = safe_point_cloud[..., rotation_slice.stop:]

    return torch.cat((xyz, opacity, middle, scales, rotations, tail), dim=-1)


def _point_clouds_to_gsplat_inputs(
    point_clouds: torch.Tensor,
    *,
    dc_only: bool = False,
    detach_input: bool = False,
    semantic_values: bool = False,
    rank_transform_tables: Optional[dict] = None,
) -> dict[str, torch.Tensor | int]:
    """Convert batched denormalized 3DGS point clouds to gsplat-native tensors."""
    del semantic_values
    if detach_input:
        point_clouds = point_clouds.detach()
    if point_clouds.ndim == 2:
        point_clouds = point_clouds.unsqueeze(0)
    if point_clouds.ndim != 3:
        raise ValueError(f"Expected point clouds with shape (B, N, D), got {tuple(point_clouds.shape)}")

    feature_dim = point_clouds.shape[-1]
    expected_dim = 14 if dc_only else 59
    if dc_only and feature_dim != expected_dim:
        raise ValueError(f"Expected {expected_dim} DC-only features, got {feature_dim}")
    if not dc_only and feature_dim < expected_dim:
        raise ValueError(f"Expected at least {expected_dim} full features, got {feature_dim}")

    pc = _constrain_denormalized_point_cloud_for_render(
        point_clouds[..., :expected_dim].to(dtype=torch.float32).contiguous(),
        dc_only=dc_only,
        rank_transform_tables=rank_transform_tables,
    )
    means = pc[..., 0:3]

    if dc_only:
        colors = pc[..., 4:7].unsqueeze(-2).contiguous()
        scales = pc[..., 7:10]
        quats = pc[..., 10:14]
        sh_degree = 0
    else:
        features = pc[..., 4:52]
        sh_coeffs = features.reshape(*pc.shape[:-1], 3, 16)
        features_dc = sh_coeffs[..., 0].unsqueeze(-2)
        features_rest = sh_coeffs[..., 1:].transpose(-1, -2)
        colors = torch.cat((features_dc, features_rest), dim=-2).contiguous()
        scales = pc[..., 52:55]
        quats = pc[..., 55:59]
        sh_degree = 3

    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": pc[..., 3],
        "colors": colors,
        "sh_degree": sh_degree,
    }


def _render_gsplat_batch(
    renderer_module,
    gaussian_inputs: dict[str, torch.Tensor | int],
    camera_bundle: dict[str, Any],
    cam_indices: list[int],
    device: torch.device,
    return_alpha: bool = False,
    per_sample_zooms: Optional[torch.Tensor] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Render a batch of Gaussian sets across a batch of cameras with gsplat.

    ``per_sample_zooms`` (optional, shape ``(B, num_cam_chosen)``): per-batch-element,
    per-chosen-camera zoom factor applied to (fx, fy) of the intrinsics. cx/cy are left at
    image center (no recentering). gsplat's kernel indexes Ks[bid, cid] per element, so
    per-sample varying intrinsics are natively supported. Pre-computed offline by
    ``data/build_per_sample_render_fov.py``.
    """
    batch_size = gaussian_inputs["means"].shape[0]
    cam_idx = torch.tensor(cam_indices, device=device, dtype=torch.long)
    viewmats = camera_bundle["viewmats"].index_select(0, cam_idx)
    intrinsics = camera_bundle["Ks"].index_select(0, cam_idx)
    num_cam = int(viewmats.shape[0])

    viewmats = viewmats.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    if per_sample_zooms is not None:
        # Scale (fx, fy) per (batch, cam) — cx, cy stay at image center (no recentering;
        # the offset distribution is small, see jit/measure_alpha_coverage.py findings).
        zoom = per_sample_zooms.to(device=device, dtype=intrinsics.dtype)   # (B, num_cam)
        intrinsics = intrinsics.clone()
        intrinsics[..., 0, 0] = intrinsics[..., 0, 0] * zoom
        intrinsics[..., 1, 1] = intrinsics[..., 1, 1] * zoom
    backgrounds = torch.zeros((batch_size, num_cam, 3), dtype=torch.float32, device=device)

    renders, alphas, _ = renderer_module.rasterization(
        means=gaussian_inputs["means"],
        quats=gaussian_inputs["quats"],
        scales=gaussian_inputs["scales"],
        opacities=gaussian_inputs["opacities"],
        colors=gaussian_inputs["colors"],
        viewmats=viewmats,
        Ks=intrinsics,
        width=int(camera_bundle["width"]),
        height=int(camera_bundle["height"]),
        sh_degree=gaussian_inputs["sh_degree"],
        backgrounds=backgrounds,
        packed=False,
        render_mode="RGB",
    )
    render_rgb = renders.permute(0, 1, 4, 2, 3).clamp(0.0, 1.0).contiguous()
    if not return_alpha:
        return render_rgb

    render_alpha = alphas.permute(0, 1, 4, 2, 3).contiguous()
    return render_rgb, render_alpha


def _compute_render_loss_for_batch(
    x0_pred: torch.Tensor,
    x_gt_full: torch.Tensor,
    norm_mean_pred: Optional[torch.Tensor],
    norm_std_pred: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: dict[str, Any],
    renderer_tuple,
    lpips_fn: Optional[nn.Module],
    num_cam: int,
    device: torch.device,
    dc_only: bool = False,
    plane_to_sphere: Optional[torch.Tensor] = None,
    sample_weights: Optional[torch.Tensor] = None,
    rank_transform_tables: Optional[dict] = None,
    per_sample_cam_zooms: Optional[torch.Tensor] = None,
    fg_phi: float = 0.0,
    diag: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable RGB, alpha-mask, and LPIPS losses over the full batch with gsplat.

    Args:
        sample_weights: Optional per-sample weights of shape (B,). When provided, each loss
            is computed as a weighted average (sum(w * per_sample) / sum(w)) instead of a
            uniform mean.  The caller is responsible for detaching these from the graph.
        rank_transform_tables: Optional Gaussian rank-transform tables; when present
            both pred and GT point clouds are inverse-rank-transformed on the listed
            channels before the renderer's sigmoid/exp activations.
        fg_phi: Share of each sample's render weight given to foreground pixels, making every
            term coverage-invariant. 0 (default) disables it and preserves the exact previous
            behaviour. Requires lpips_fn built with spatial=True. See the block below.
        diag: Optional dict; when given, receives the UNWEIGHTED value of each term
            ('render_*_plain') plus mean coverage and mean foreground upweight. Passed out this
            way rather than as extra return values so existing 3-tuple call sites keep working.
    """
    batch_size = x_gt_full.shape[0]
    view_count = max(1, int(num_cam))
    total_cams = int(train_cameras["viewmats"].shape[0])
    cam_indices = random.sample(range(total_cams), min(view_count, total_cams))
    # Select per-sample zooms for the chosen subset of cameras. Same indices for GT/pred
    # so both sides render through identical intrinsics (otherwise the loss is meaningless).
    sample_zooms = None
    if per_sample_cam_zooms is not None:
        cam_idx_t = torch.tensor(cam_indices, device=per_sample_cam_zooms.device, dtype=torch.long)
        sample_zooms = per_sample_cam_zooms.index_select(1, cam_idx_t)   # (B, num_cam_chosen)

    gt_pc_norm = _plane_to_point_cloud_batch(x_gt_full.float(), plane_to_sphere)
    gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
    gt_is_dc_only = gt_pc_raw.shape[-1] == 14
    gt_gaussians = _point_clouds_to_gsplat_inputs(
        gt_pc_raw.to(device), dc_only=gt_is_dc_only, detach_input=True,
        rank_transform_tables=rank_transform_tables,
    )
    if dc_only and not gt_is_dc_only:
        # Drop higher-order SH from GT so both sides render at the same SH degree;
        # otherwise view-dependent specular effects in the GT create a loss that the
        # DC-only prediction structurally cannot minimise.
        gt_gaussians = {**gt_gaussians, "colors": gt_gaussians["colors"][..., :1, :], "sh_degree": 0}

    pred_pc_norm = _plane_to_point_cloud_batch(x0_pred.float(), plane_to_sphere)
    pred_pc_raw = _denormalize_point_cloud(pred_pc_norm, norm_mean_pred, norm_std_pred)
    pred_gaussians = _point_clouds_to_gsplat_inputs(
        pred_pc_raw.to(device),
        dc_only=dc_only,
        detach_input=False,
        rank_transform_tables=rank_transform_tables,
    )

    with torch.no_grad():
        target, target_alpha = _render_gsplat_batch(
            renderer_tuple,
            gt_gaussians,
            train_cameras,
            cam_indices,
            device,
            return_alpha=True,
            per_sample_zooms=sample_zooms,
        )
    pred, pred_alpha = _render_gsplat_batch(
        renderer_tuple,
        pred_gaussians,
        train_cameras,
        cam_indices,
        device,
        return_alpha=True,
        per_sample_zooms=sample_zooms,
    )

    # ---- coverage-normalised foreground/background weighting --------------------------------
    # The plain per-frame-pixel mean weights each sample by how much of the frame its object
    # actually paints, so the cost of DELETING an object equals its coverage. Measured on 1,024
    # GT objects (jit/measure_render_coverage.py): coverage spans 8.7x p05->p95 (0.053 -> 0.465)
    # and is driven by aspect ratio (corr(log aspect, log coverage) = -0.77), because the framing
    # equalises bounding-box fill, not coverage. Thin objects are therefore ~5-9x cheaper to
    # delete -- the thin-geometry deletion seen from step 38k to 100k of the render-06 arm.
    #
    # fg_phi gives the foreground a fixed share `phi` of each sample's weight regardless of
    # coverage:   w_fg = phi/c,  w_bg = (1-phi)/(1-c),  w = a*w_fg + (1-a)*w_bg
    # sum(w) == N by construction, so the weighted mean stays on the same scale as the plain one.
    # fg_phi == c reproduces the plain mean EXACTLY (both weights collapse to 1), and fg_phi <= 0
    # disables the whole path -- that identity is what makes this a safe generalisation rather
    # than a rewrite, and it is asserted in tests/test_fg_phi_identity.py.
    #
    # Background keeps weight w_bg rather than being masked to zero on purpose: alpha is
    # colour-blind (it penalises presence, not appearance), so a hard mask would let the model
    # paint arbitrary colour outside the silhouette at no cost from the RGB/LPIPS terms.
    use_phi = float(fg_phi) > 0.0
    w_map = None
    if use_phi:
        cov = target_alpha.mean(dim=(1, 2, 3, 4), keepdim=True).clamp(
            min=_FG_PHI_COV_MIN, max=_FG_PHI_COV_MAX)
        w_map = (target_alpha * (float(fg_phi) / cov)
                 + (1.0 - target_alpha) * ((1.0 - float(fg_phi)) / (1.0 - cov)))

    l1_err = torch.abs(pred - target)                    # (B, cams, 3, H, W)
    alpha_err = torch.abs(pred_alpha - target_alpha)     # (B, cams, 1, H, W)
    l1_plain = l1_err.mean(dim=(1, 2, 3, 4))
    alpha_plain = alpha_err.mean(dim=(1, 2, 3, 4))
    if use_phi:
        l1_per_sample = (w_map * l1_err).mean(dim=(1, 2, 3, 4))
        alpha_l1_per_sample = (w_map * alpha_err).mean(dim=(1, 2, 3, 4))
    else:
        l1_per_sample, alpha_l1_per_sample = l1_plain, alpha_plain

    def _reduce(per_sample):
        if sample_weights is not None:
            return (per_sample * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
        return per_sample.mean()

    l1_loss = _reduce(l1_per_sample)
    alpha_l1_loss = _reduce(alpha_l1_per_sample)

    lpips_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    lpips_plain = None
    if lpips_fn is not None:
        n_cam = len(cam_indices)
        pred_n = (pred * 2.0 - 1.0).reshape(
            batch_size * n_cam, 3, pred.shape[-2], pred.shape[-1]
        )
        target_n = (target * 2.0 - 1.0).reshape(
            batch_size * n_cam, 3, target.shape[-2], target.shape[-1]
        )
        out = lpips_fn(pred_n, target_n)
        if use_phi:
            # lpips_fn must be built with spatial=True: returns (B*C, 1, H, W) at INPUT
            # resolution, so the alpha weight map applies directly with no interpolation.
            # Its plain mean equals the scalar LPIPS exactly, so the diagnostic below stays
            # comparable to runs that never used this path.
            if out.dim() != 4 or out.shape[-1] != pred.shape[-1]:
                raise RuntimeError(
                    f"render_fg_phi>0 requires LPIPS(spatial=True); got output {tuple(out.shape)} "
                    f"for input {tuple(pred_n.shape)}")
            lmap = out.reshape(batch_size, n_cam, 1, out.shape[-2], out.shape[-1])
            lpips_plain = lmap.mean(dim=(1, 2, 3, 4))
            lpips_per_sample = (w_map * lmap).mean(dim=(1, 2, 3, 4))
        else:
            # (B*C, 1, 1, 1) → (B*C,) → (B, C) → mean over cameras
            lpips_per_sample = out.flatten().view(batch_size, n_cam).mean(dim=1)
            lpips_plain = lpips_per_sample
        lpips_loss = _reduce(lpips_per_sample)

    if diag is not None:
        # Unweighted values, for cross-arm comparison against runs trained without fg_phi.
        diag["render_l1_plain"] = float(_reduce(l1_plain).detach())
        diag["render_alpha_plain"] = float(_reduce(alpha_plain).detach())
        if lpips_plain is not None:
            diag["render_lpips_plain"] = float(_reduce(lpips_plain).detach())
        if use_phi:
            diag["fg_cov_mean"] = float(cov.mean().detach())
            diag["fg_w_mean"] = float((float(fg_phi) / cov).mean().detach())

    return l1_loss, alpha_l1_loss, lpips_loss


def _render_variance_score(renders: torch.Tensor) -> torch.Tensor:
    """Score rendered views by how visually informative they are."""
    return renders.float().flatten(start_dim=2).std(dim=-1)


def _select_preview_sample_and_views(
    x_gt_full: torch.Tensor,
    plane_to_sphere: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: dict[str, Any],
    renderer_tuple,
    device: torch.device,
    num_cam: int,
    rank_transform_tables: Optional[dict] = None,
) -> tuple[int, list[int]]:
    """Pick a preview sample and views that avoid flat, uninformative GT renders."""
    batch_size = x_gt_full.shape[0]
    total_cams = int(train_cameras["viewmats"].shape[0])
    view_count = max(1, int(num_cam))

    if batch_size <= 1:
        candidate_sample_indices = [0]
    else:
        candidate_sample_count = min(batch_size, 8)
        candidate_sample_indices = random.sample(range(batch_size), candidate_sample_count)

    if total_cams <= 1:
        candidate_cam_indices = [0]
    else:
        candidate_cam_count = min(total_cams, max(view_count, 12))
        candidate_cam_indices = random.sample(range(total_cams), candidate_cam_count)

    gt_pc_norm = _plane_to_point_cloud_batch(
        x_gt_full[candidate_sample_indices].float(), plane_to_sphere
    )
    gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
    gt_is_dc_only = gt_pc_raw.shape[-1] == 14
    gt_gaussians = _point_clouds_to_gsplat_inputs(
        gt_pc_raw.to(device), dc_only=gt_is_dc_only, detach_input=True,
        rank_transform_tables=rank_transform_tables,
    )
    gt_views = _render_gsplat_batch(
        renderer_tuple, gt_gaussians, train_cameras, candidate_cam_indices, device
    )

    scores = _render_variance_score(gt_views)
    best_sample_pos = int(scores.max(dim=1).values.argmax().item())
    topk = torch.topk(
        scores[best_sample_pos],
        k=min(view_count, scores.shape[1]),
        largest=True,
    ).indices.tolist()
    selected_cams = [candidate_cam_indices[idx] for idx in topk]
    return candidate_sample_indices[best_sample_pos], selected_cams


def _save_training_render_preview(
    x0_pred: torch.Tensor,
    x_gt_full: torch.Tensor,
    norm_mean_pred: Optional[torch.Tensor],
    norm_std_pred: Optional[torch.Tensor],
    norm_mean_full: Optional[torch.Tensor],
    norm_std_full: Optional[torch.Tensor],
    train_cameras: dict[str, Any],
    renderer_tuple,
    output_dir: str,
    epoch: int,
    step: int,
    timesteps: torch.Tensor,
    labels: Optional[torch.Tensor],
    device: torch.device,
    num_cam: int,
    dc_only: bool = False,
    plane_to_sphere: Optional[torch.Tensor] = None,
    rank_transform_tables: Optional[dict] = None,
) -> None:
    """Save side-by-side GT/pred train-time renders for quick visual inspection."""
    sample_idx, cam_indices = _select_preview_sample_and_views(
        x_gt_full=x_gt_full,
        plane_to_sphere=plane_to_sphere,
        norm_mean_full=norm_mean_full,
        norm_std_full=norm_std_full,
        train_cameras=train_cameras,
        renderer_tuple=renderer_tuple,
        device=device,
        num_cam=num_cam,
        rank_transform_tables=rank_transform_tables,
    )
    rows = []

    with torch.no_grad():
        gt_pc_norm = _plane_to_point_cloud_batch(
            x_gt_full[sample_idx : sample_idx + 1].float(), plane_to_sphere
        )
        gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
        gt_is_dc_only = gt_pc_raw.shape[-1] == 14
        gt_gaussians = _point_clouds_to_gsplat_inputs(
            gt_pc_raw.to(device), dc_only=gt_is_dc_only, detach_input=True,
            rank_transform_tables=rank_transform_tables,
        )
        if dc_only and not gt_is_dc_only:
            gt_gaussians = {**gt_gaussians, "colors": gt_gaussians["colors"][..., :1, :], "sh_degree": 0}

        pred_pc_norm = _plane_to_point_cloud_batch(
            x0_pred[sample_idx : sample_idx + 1].float(), plane_to_sphere
        )
        pred_pc_raw = _denormalize_point_cloud(pred_pc_norm, norm_mean_pred, norm_std_pred)
        pred_gaussians = _point_clouds_to_gsplat_inputs(
            pred_pc_raw.to(device),
            dc_only=dc_only,
            detach_input=True,
            rank_transform_tables=rank_transform_tables,
        )

        target_views = _render_gsplat_batch(
            renderer_tuple, gt_gaussians, train_cameras, cam_indices, device
        )[0]
        pred_views = _render_gsplat_batch(
            renderer_tuple, pred_gaussians, train_cameras, cam_indices, device
        )[0]

        for target, pred in zip(target_views, pred_views):
            target_np = (
                target.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            pred_np = (
                pred.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0
            ).astype(np.uint8)
            column_separator = np.full((target_np.shape[0], 4, 3), 255, dtype=np.uint8)
            rows.append(np.concatenate([target_np, column_separator, pred_np], axis=1))

    if not rows:
        return

    if len(rows) == 1:
        preview = rows[0]
    else:
        row_separator = np.full((4, rows[0].shape[1], 3), 255, dtype=np.uint8)
        preview = np.concatenate(
            [piece for row in rows[:-1] for piece in (row, row_separator)] + [rows[-1]],
            axis=0,
        )

    preview_dir = os.path.join(output_dir, "dit_train_renders")
    os.makedirs(preview_dir, exist_ok=True)
    timestep = int(timesteps[sample_idx].item())
    out_path = os.path.join(
        preview_dir,
        f"epoch_{epoch:03d}_step_{step:07d}_t{timestep:04d}.png",
    )
    latest_path = os.path.join(preview_dir, "latest.png")
    Image.fromarray(preview).save(out_path)
    Image.fromarray(preview).save(latest_path)
    logger.info(
        "[train-render] saved: %s (left=gt, right=pred, t=%d)",
        out_path,
        timestep,
    )


__all__ = [
    "Image",
    "RENDER_OPACITY_RAW_MAX",
    "RENDER_OPACITY_RAW_MIN",
    "RENDER_QUAT_EPS",
    "RENDER_SCALE_RAW_MAX",
    "RENDER_SCALE_RAW_MIN",
    "_camera_intrinsics_from_ref",
    "_camera_viewmat_from_ref",
    "_apply_rank_inverse_for_render",
    "_compute_render_loss_for_batch",
    "_constrain_denormalized_point_cloud_for_render",
    "_denormalize_point_cloud",
    "_fov2focal",
    "_load_reference_cameras",
    "_normalize_quaternions_with_identity_fallback",
    "_resolve_rank_layout_for_active_space",
    "load_rank_transform_payload_torch",
    "_plane_to_point_cloud_batch",
    "_point_clouds_to_gsplat_inputs",
    "_prepare_train_cameras",
    "_render_gsplat_batch",
    "_render_variance_score",
    "_save_training_render_preview",
    "_select_preview_sample_and_views",
    "_try_import_lpips",
    "_try_import_renderer",
]
