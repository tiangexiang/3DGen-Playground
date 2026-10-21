"""Flow matching with clean-data prediction and a velocity-space loss.

The interpolation is x_t = t*x0 + (1-t)*noise: t=0 is noise and t=1 is clean.
The velocity loss equals MSE(pred_x0, x0) / max(1-t, v_clamp_min)^2.
Training and sampling use separate denominator clamps; exact_final emits the
clean-data prediction at the last sampling step."""
from __future__ import annotations

import torch
from torch import nn


def sample_t(batch: int, device: torch.device, schedule: str = "lognorm",
             mu: float = -0.8, sigma: float = 0.8) -> torch.Tensor:
    """Draw continuous t in (0,1). t=0 pure noise, t=1 clean."""
    if schedule == "uniform":
        return torch.rand(batch, device=device).clamp(1e-5, 1.0 - 1e-5)
    if schedule == "lognorm":
        z = torch.randn(batch, device=device) * sigma + mu
        return torch.sigmoid(z).clamp(1e-5, 1.0 - 1e-5)
    raise ValueError(f"unknown t_sample_schedule={schedule!r}; choices: lognorm, uniform")


def _bcast(t: torch.Tensor, ndim: int) -> torch.Tensor:
    return t.view(-1, *([1] * (ndim - 1)))


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None,
             noise_scale: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """x_t = t*x0 + (1-t)*eps, with eps ~ N(0, noise_scale^2)."""
    if noise is None:
        noise = torch.randn_like(x0) * noise_scale
    tv = _bcast(t, x0.ndim).to(dtype=x0.dtype)
    return x0 * tv + noise * (1.0 - tv), noise


def drop_mask_for_cfg(attn_mask: torch.Tensor, drop_rate: float) -> torch.Tensor:
    """CFG dropout = zero out the whole attention mask for a random subset.

    No null-embedding cache: an all-zero mask makes the model replace every text
    position with its learned ``mask_token``, which IS the unconditional input.
    """
    if drop_rate <= 0:
        return attn_mask
    drop = torch.rand(attn_mask.shape[0], device=attn_mask.device) < drop_rate
    return torch.where(drop[:, None], torch.zeros_like(attn_mask), attn_mask)


def training_loss(
    model: nn.Module,
    x0: torch.Tensor,
    context: torch.Tensor,
    attn_mask: torch.Tensor,
    *,
    label_drop_rate: float = 0.1,
    t_schedule: str = "lognorm",
    t_lognorm_mu: float = -0.8,
    t_lognorm_sigma: float = 0.8,
    noise_scale: float = 2.0,
    v_clamp_min: float = 0.05,
    recon_loss: str = "v_mse",
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Returns ``(loss, aux)`` where aux carries per-sample terms for the caller.

    ``aux`` includes ``pred_x0``/``x_t``/``t`` so an optional auxiliary-loss
    bridge (sinkhorn / render) can attach WITHOUT a second forward pass.
    """
    b = x0.shape[0]
    device = x0.device
    attn_mask = drop_mask_for_cfg(attn_mask, label_drop_rate)
    if t is None:
        t = sample_t(b, device, t_schedule, t_lognorm_mu, t_lognorm_sigma)
    x_t, noise = q_sample(x0, t, noise=noise, noise_scale=noise_scale)

    pred_x0 = model(x_t, t, context, attn_mask)
    if pred_x0.shape != x0.shape:
        raise ValueError(f"model returned {tuple(pred_x0.shape)}, expected {tuple(x0.shape)}")

    tv = _bcast(t, x0.ndim).to(dtype=pred_x0.dtype)
    if recon_loss == "v_mse":
        den = (1.0 - tv).clamp_min(v_clamp_min)
        per_sample = ((pred_x0 - x_t) / den - (x0 - x_t) / den).pow(2).mean(
            dim=tuple(range(1, x0.ndim))
        )
    elif recon_loss == "x0_mse":
        per_sample = (pred_x0 - x0).pow(2).mean(dim=tuple(range(1, x0.ndim)))
    else:
        raise ValueError(f"recon_loss must be v_mse|x0_mse, got {recon_loss!r}")

    loss = per_sample.mean()
    aux = {
        "loss": loss.detach(),
        "per_sample": per_sample.detach(),
        "pred_x0": pred_x0,
        "x_t": x_t,
        "t": t,
        "noise": noise,
        "attn_mask": attn_mask,
        # x0-space MSE is reported alongside the v-space objective because the
        # v-loss magnitude is dominated by its 1/(1-t)^2 weight and is therefore
        # not comparable across runs with different t-schedules or clamps.
        "x0_mse": (pred_x0.detach() - x0).pow(2).mean().detach(),
    }
    return loss, aux


@torch.no_grad()
def euler_sample(
    model: nn.Module,
    context: torch.Tensor,
    attn_mask: torch.Tensor,
    shape: tuple[int, ...],
    *,
    steps: int = 100,
    cfg_scale: float = 2.5,
    noise_scale: float = 2.0,
    v_clamp_min: float = 0.05,
    exact_final: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    progress: bool = False,
) -> torch.Tensor:
    """Euler ODE integration from t=0 (noise) to t=1 (clean).

    CFG runs the conditional and unconditional branches in ONE batched forward:
    the unconditional half is the same latent with a zeroed attention mask.

    ``v_clamp_min`` here is the SAMPLING clamp and is independent of the training
    one. With ``v_clamp_min=0`` the integration is exact for a perfect x0-oracle;
    at MiniT2I's 0.05 it leaves 0.05*0.8^5 = 1.6384% of the initial noise at 100
    steps. Never reaches t=1 inside the loop, so an unclamped denominator cannot
    divide by zero.

    ``exact_final`` emits the model's ``pred_x0`` on the last step instead of the
    integrated ``x``. This is what an x0-prediction model should return, and it
    removes the clamp residual EXACTLY, at any step count, while leaving the clamp
    free to stabilise the rest of the trajectory. That residual is not negligible
    here: ``noise_scale`` is 2.0, so 1.6384% of the initial draw is 0.033 sigma per
    channel on containers rank-transformed to ~N(0,1) -- enough to move splats whose
    opacity sits near the visibility threshold, which is exactly the thin-geometry
    failure mode. Note it grows with step count (1.44% at 50, 1.74% at 200,
    asymptotically clamp/e), so adding steps makes it worse, not better.
    Set False to reproduce MiniT2I's original behaviour.
    """
    model.eval()
    b = context.shape[0]
    device = device or context.device
    x = torch.randn(b, *shape, device=device, dtype=dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    null_mask = torch.zeros_like(attn_mask)

    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        tt = torch.full((b,), float(t0), device=device)
        if cfg_scale != 1.0:
            xx = torch.cat([x, x], dim=0)
            t2 = torch.cat([tt, tt], dim=0)
            cc = torch.cat([context, context], dim=0)
            mm = torch.cat([attn_mask, null_mask], dim=0)
            both = model(xx, t2, cc, mm)
            pred_cond, pred_uncond = both[:b], both[b:]
            pred_x0 = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
        else:
            pred_x0 = model(x, tt, context, attn_mask)
        if exact_final and i == steps - 1:
            # t1 == 1: the exact update IS pred_x0, so take it directly rather
            # than approaching it through a clamped denominator.
            x = pred_x0
        else:
            den = (1.0 - t0).clamp_min(v_clamp_min) if v_clamp_min > 0 else (1.0 - t0)
            v = (pred_x0 - x) / den
            x = x + v * (t1 - t0)
        if progress and (i + 1) % max(steps // 10, 1) == 0:
            print(f"  [sample] {i + 1}/{steps}", flush=True)
    return x
