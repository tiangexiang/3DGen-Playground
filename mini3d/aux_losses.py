"""Point-matching and foreground-weighted render losses.

The v-space weight applies only to the diffusion reconstruction term.
Matching and render losses operate directly on clean-splat predictions."""
from __future__ import annotations

import logging

import torch

from .config import TrainConfig
from .render import RenderContext

from .matching import _chamfer_recon_loss
from utils.gsplat_render_util import _compute_render_loss_for_batch

logger = logging.getLogger(__name__)


def any_aux_enabled(cfg: TrainConfig) -> bool:
    return any(w > 0 for w in (cfg.sinkhorn_weight, cfg.render_loss_weight,
                               cfg.lpips_loss_weight, cfg.alpha_mask_loss_weight,
                               cfg.mse_hybrid_weight))


def needs_render(cfg: TrainConfig) -> bool:
    return any(w > 0 for w in (cfg.render_loss_weight, cfg.lpips_loss_weight,
                               cfg.alpha_mask_loss_weight))


def sinkhorn_term(cfg: TrainConfig, pred_x0: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Permutation-invariant reconstruction over the per-Gaussian point set."""
    recon, _ = _chamfer_recon_loss(
        pred_x0, x0, cfg.sinkhorn_mode,
        weight=cfg.chamfer_loss_weight,
        subsample=cfg.chamfer_subsample,
        patch_size=cfg.patch_size,
        rev_weight=cfg.chamfer_rev_weight,
        sinkhorn_eps=cfg.sinkhorn_epsilon,
        sinkhorn_iters=cfg.sinkhorn_iters,
        huber_delta=cfg.huber_delta,
        chunk_1d=(cfg.token_order == "hilbert1d"),
        return_diag=True,
    )
    return recon.mean()


def render_terms(cfg: TrainConfig, ctx: RenderContext, pred_x0: torch.Tensor,
                 x_gt: torch.Tensor, t: torch.Tensor,
                 keys: list[str] | None = None) -> tuple[torch.Tensor, dict]:
    """RGB, alpha, and spatial LPIPS losses over the retained noise levels.

    Prediction and target must use the same 14-channel DC color layout.
    Samples with t below render_loss_noise_cutoff are removed before rendering.
    If none remain, the zero loss retains a gradient path for distributed training.
    Returns the weighted loss and scalar diagnostics; *_plain metrics omit
    foreground reweighting."""
    keep_mask = t >= cfg.render_loss_noise_cutoff
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        # Still route through pred_x0 so every rank contributes a (zero) gradient to
        # the same parameters -- dropping the term entirely desyncs DDP's allreduce.
        z = pred_x0.sum() * 0.0
        return z, {"render_kept": 0.0}

    frac_kept = n_kept / int(keep_mask.numel())
    idx = keep_mask.nonzero(as_tuple=True)[0]
    pred_x0 = pred_x0[idx]          # tensor-indexing keeps the autograd graph
    x_gt = x_gt[idx]
    if keys is not None:
        keys = [keys[i] for i in idx.tolist()]

    # `rdiag` collects the UNWEIGHTED (_plain) values. Under fg_phi > 0 the
    # reported losses are coverage-reweighted and therefore NOT comparable across
    # arms -- the plain values are what compare against a run trained without it.
    rdiag: dict = {}
    rgb, alpha, lpips = _compute_render_loss_for_batch(
        pred_x0, x_gt,
        # Same 14-channel stats on BOTH sides -- GT is the DC-only container.
        ctx.norm_mean, ctx.norm_std, ctx.norm_mean, ctx.norm_std,
        ctx.train_cameras, ctx.renderer_tuple, ctx.lpips_fn,
        cfg.render_loss_num_cam, ctx.device,
        dc_only=ctx.dc_only, plane_to_sphere=ctx.plane_to_sphere,
        # No sample_weights: the batch IS the kept subset now, so the helper's plain
        # mean equals the weighted mean it used to compute.
        rank_transform_tables=ctx.rank_transform_tables,
        per_sample_cam_zooms=ctx.zooms_for(keys) if keys is not None else None,
        fg_phi=float(cfg.render_fg_phi),
        diag=rdiag,
    )
    total = (cfg.render_loss_weight * rgb
             + cfg.alpha_mask_loss_weight * alpha
             + cfg.lpips_loss_weight * lpips)
    diag = {"render_rgb": float(rgb.detach()), "render_alpha": float(alpha.detach()),
            "render_lpips": float(lpips.detach()),
            "render_kept": frac_kept}
    diag.update({k: float(v) for k, v in rdiag.items()})
    return total, diag


def compute(cfg: TrainConfig, ctx: RenderContext | None, aux: dict, x0: torch.Tensor,
            step: int, keys: list[str] | None = None) -> tuple[torch.Tensor, dict]:
    """Sum every enabled auxiliary term. Returns ``(loss, diagnostics)``.

    ``aux`` is the dict returned by ``diffusion.training_loss`` -- reusing its
    ``pred_x0`` means no second forward pass.
    """
    pred_x0 = aux["pred_x0"]
    total = pred_x0.sum() * 0.0
    diag: dict = {}

    if cfg.sinkhorn_weight > 0:
        s = sinkhorn_term(cfg, pred_x0, x0)
        total = total + cfg.sinkhorn_weight * s
        diag["sinkhorn"] = float(s.detach())

    if cfg.mse_hybrid_weight > 0:
        m = (pred_x0 - x0).pow(2).mean()
        total = total + cfg.mse_hybrid_weight * m
        diag["mse_hybrid"] = float(m.detach())

    if needs_render(cfg) and step >= cfg.enable_render_loss_after:
        if ctx is None:
            raise RuntimeError("render loss weights are set but no RenderContext was built")
        # x0 (14-channel DC-only), NOT x_gt_full: the DC-only and full-SH render
        # branches are RGB-reversed, so a 59-channel ground truth would make this
        # loss compare BGR against RGB. See render_terms() for the measurement.
        r, rdiag = render_terms(cfg, ctx, pred_x0, x0, aux["t"], keys=keys)
        total = total + r
        diag.update(rdiag)

    return total, diag


@torch.no_grad()
def render_metrics(cfg: TrainConfig, ctx: RenderContext, pred_x0: torch.Tensor,
                   x_gt: torch.Tensor, keys: list[str] | None = None) -> dict:
    """Unweighted render diagnostics, computed without gradients.

    Scalar LPIPS is reported when available. With spatial LPIPS, use the
    render_lpips_plain diagnostic produced by render_terms instead."""
    from utils.gsplat_render_util import _compute_render_loss_for_batch as _f
    spatial = bool(getattr(ctx.lpips_fn, "spatial", False))
    rgb, alpha, lpips = _f(
        pred_x0.detach(), x_gt,
        ctx.norm_mean, ctx.norm_std, ctx.norm_mean, ctx.norm_std,
        ctx.train_cameras, ctx.renderer_tuple, None if spatial else ctx.lpips_fn,
        cfg.render_loss_num_cam, ctx.device,
        dc_only=ctx.dc_only, plane_to_sphere=ctx.plane_to_sphere,
        rank_transform_tables=ctx.rank_transform_tables,
        per_sample_cam_zooms=ctx.zooms_for(keys) if keys is not None else None,
        fg_phi=0.0,
    )
    out = {"metric/render_l1": float(rgb), "metric/alpha_l1": float(alpha)}
    if not spatial:
        out["metric/lpips"] = float(lpips)
    return out
