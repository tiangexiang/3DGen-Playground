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
) -> dict[str, Any]:
    viewmats = []
    intrinsics = []
    for ref_cam in ref_cameras:
        ref_small = dict(ref_cam)
        ref_small["width"] = int(train_render_size)
        ref_small["height"] = int(train_render_size)
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
) -> torch.Tensor:
    """Project denormalized predictions into canonical 3DGS parameter space for rendering."""
    if dc_only:
        scale_slice = slice(7, 10)
        rotation_slice = slice(10, 14)
    else:
        scale_slice = slice(52, 55)
        rotation_slice = slice(55, 59)

    safe_point_cloud = torch.nan_to_num(point_cloud, nan=0.0, posinf=0.0, neginf=0.0)
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Render a batch of Gaussian sets across a batch of cameras with gsplat."""
    batch_size = gaussian_inputs["means"].shape[0]
    cam_idx = torch.tensor(cam_indices, device=device, dtype=torch.long)
    viewmats = camera_bundle["viewmats"].index_select(0, cam_idx)
    intrinsics = camera_bundle["Ks"].index_select(0, cam_idx)
    num_cam = int(viewmats.shape[0])

    viewmats = viewmats.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    intrinsics = intrinsics.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute differentiable RGB, alpha-mask, and LPIPS losses over the full batch with gsplat.

    Args:
        sample_weights: Optional per-sample weights of shape (B,). When provided, each loss
            is computed as a weighted average (sum(w * per_sample) / sum(w)) instead of a
            uniform mean.  The caller is responsible for detaching these from the graph.
    """
    batch_size = x_gt_full.shape[0]
    view_count = max(1, int(num_cam))
    total_cams = int(train_cameras["viewmats"].shape[0])
    cam_indices = random.sample(range(total_cams), min(view_count, total_cams))

    gt_pc_norm = _plane_to_point_cloud_batch(x_gt_full.float(), plane_to_sphere)
    gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
    gt_is_dc_only = gt_pc_raw.shape[-1] == 14
    gt_gaussians = _point_clouds_to_gsplat_inputs(
        gt_pc_raw.to(device), dc_only=gt_is_dc_only, detach_input=True
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
    )

    with torch.no_grad():
        target, target_alpha = _render_gsplat_batch(
            renderer_tuple,
            gt_gaussians,
            train_cameras,
            cam_indices,
            device,
            return_alpha=True,
        )
    pred, pred_alpha = _render_gsplat_batch(
        renderer_tuple,
        pred_gaussians,
        train_cameras,
        cam_indices,
        device,
        return_alpha=True,
    )

    # Per-sample losses: average over (cameras, channels, H, W) → shape (B,)
    l1_per_sample = torch.abs(pred - target).mean(dim=(1, 2, 3, 4))
    alpha_l1_per_sample = torch.abs(pred_alpha - target_alpha).mean(dim=(1, 2, 3, 4))

    if sample_weights is not None:
        w_sum = sample_weights.sum().clamp(min=1e-8)
        l1_loss = (l1_per_sample * sample_weights).sum() / w_sum
        alpha_l1_loss = (alpha_l1_per_sample * sample_weights).sum() / w_sum
    else:
        l1_loss = l1_per_sample.mean()
        alpha_l1_loss = alpha_l1_per_sample.mean()

    lpips_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    if lpips_fn is not None:
        pred_n = (pred * 2.0 - 1.0).reshape(
            batch_size * len(cam_indices), 3, pred.shape[-2], pred.shape[-1]
        )
        target_n = (target * 2.0 - 1.0).reshape(
            batch_size * len(cam_indices), 3, target.shape[-2], target.shape[-1]
        )
        # LPIPS returns (B*C, 1, 1, 1) → flatten to (B*C,) → reshape to (B, C) → mean over cameras
        lpips_per_sample = lpips_fn(pred_n, target_n).flatten().view(
            batch_size, len(cam_indices)
        ).mean(dim=1)  # (B,)
        if sample_weights is not None:
            lpips_loss = (lpips_per_sample * sample_weights).sum() / w_sum
        else:
            lpips_loss = lpips_per_sample.mean()

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
    gt_gaussians = _point_clouds_to_gsplat_inputs(
        gt_pc_raw.to(device), dc_only=False, detach_input=True
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
    labels: torch.Tensor,
    device: torch.device,
    num_cam: int,
    dc_only: bool = False,
    plane_to_sphere: Optional[torch.Tensor] = None,
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
    )
    rows = []

    with torch.no_grad():
        gt_pc_norm = _plane_to_point_cloud_batch(
            x_gt_full[sample_idx : sample_idx + 1].float(), plane_to_sphere
        )
        gt_pc_raw = _denormalize_point_cloud(gt_pc_norm, norm_mean_full, norm_std_full)
        gt_is_dc_only = gt_pc_raw.shape[-1] == 14
        gt_gaussians = _point_clouds_to_gsplat_inputs(
            gt_pc_raw.to(device), dc_only=gt_is_dc_only, detach_input=True
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
    y_label = int(labels[sample_idx].item())
    timestep = int(timesteps[sample_idx].item())
    out_path = os.path.join(
        preview_dir,
        f"epoch_{epoch:03d}_step_{step:07d}_class{y_label:03d}_t{timestep:04d}.png",
    )
    latest_path = os.path.join(preview_dir, "latest.png")
    Image.fromarray(preview).save(out_path)
    Image.fromarray(preview).save(latest_path)
    logger.info(
        "[train-render] saved: %s (left=gt, right=pred, class=%d, t=%d)",
        out_path,
        y_label,
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
    "_compute_render_loss_for_batch",
    "_constrain_denormalized_point_cloud_for_render",
    "_denormalize_point_cloud",
    "_fov2focal",
    "_load_reference_cameras",
    "_normalize_quaternions_with_identity_fallback",
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
