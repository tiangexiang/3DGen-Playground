"""Legacy ODE samplers (Euler / Heun) for models trained with the DDPM path.

Models trained on the ``dev`` branch use standard DDPM forward diffusion
(discrete integer timesteps 0-999 where 0 = clean, 999 = pure noise) with
``predict_xstart=True``.  The ODE samplers below wrap such models by:

1. Mapping the continuous ODE variable ``t in [0, 1]`` to the correct
   discrete DDPM timestep for the model's sinusoidal embedding.
2. Using the velocity formula ``v = (x0_pred - x_t) / (1 - t)`` derived
   from the linear-interpolation path
   ``x_t = (1 - t) * noise + t * x_0``.
3. Integrating from t = 0 (noise) to t = 1 (clean).

The DDPM / DPM / DDIM samplers are re-exported from ``jit.sampling`` because
they already handle integer timesteps correctly and work with DDPM-trained
models out of the box.
"""

from __future__ import annotations

from typing import Optional

import torch

from jit.sampling import (
    SAMPLER_CHOICES,
    build_ddpm_diffusion,
    build_dpm_scheduler,
    resolve_sampling_shape,
    sample_with_ddpm,
    sample_with_ddim,
    sample_with_dpm,
    _validate_sampling_shape,
)


# ---------------------------------------------------------------------------
# Legacy velocity derivation (dev-branch formulation, with fixed t-mapping)
# ---------------------------------------------------------------------------

def _legacy_velocity(
    *,
    model: torch.nn.Module,
    sample: torch.Tensor,
    t_value: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    diffusion_steps: int,
    t_eps: float,
) -> torch.Tensor:
    """Evaluate velocity using the legacy (dev-branch) formulation.

    The model predicts x_0 and velocity is derived as:
        v = (x_0_pred - x_t) / (1 - t)

    The continuous ODE time ``t_value in [0, 1]`` is mapped to a discrete
    DDPM timestep via ``(1 - t_value) * (diffusion_steps - 1)`` so that
    ODE t=0 (noise) maps to DDPM step 999 (noise) and ODE t=1 (clean)
    maps to DDPM step 0 (clean).
    """
    model_dtype = next(model.parameters()).dtype
    # Map ODE time to discrete DDPM step.
    # ODE convention: t=0 -> noise, t=1 -> clean
    # DDPM convention: s=0 -> clean, s=999 -> noise
    # So: s = (1 - t) * (S - 1)
    t_discrete = ((1.0 - t_value) * (diffusion_steps - 1)).round().clamp(
        0, diffusion_steps - 1
    ).long()
    t_batch = t_discrete.expand(sample.shape[0]).to(device=sample.device)
    sample_input = sample.to(dtype=model_dtype)
    x_cond = model(sample_input, t_batch, class_labels).float()

    denom = (1.0 - t_value).clamp_min(t_eps).to(dtype=sample.dtype)
    v_cond = (x_cond - sample) / denom

    if cfg_scale == 1.0:
        return v_cond

    y_embedder = getattr(model, "y_embedder", None)
    num_classes = getattr(y_embedder, "num_classes", None)
    if num_classes is None:
        raise ValueError("JiT Euler/Heun CFG requires model.y_embedder.num_classes to be available")

    low, high = cfg_interval
    t_scalar = float(t_value.item())
    cfg_scale_interval = float(cfg_scale) if (t_scalar < high and (low == 0.0 or t_scalar > low)) else 1.0
    if cfg_scale_interval == 1.0:
        return v_cond

    null_labels = torch.full_like(class_labels, int(num_classes))
    x_uncond = model(sample_input, t_batch, null_labels).float()
    v_uncond = (x_uncond - sample) / denom
    return v_uncond + cfg_scale_interval * (v_cond - v_uncond)


# ---------------------------------------------------------------------------
# Euler and Heun steppers
# ---------------------------------------------------------------------------

def _legacy_euler_step(
    *,
    model: torch.nn.Module,
    sample: torch.Tensor,
    t_value: torch.Tensor,
    t_next: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    diffusion_steps: int,
    t_eps: float,
) -> torch.Tensor:
    velocity = _legacy_velocity(
        model=model,
        sample=sample,
        t_value=t_value,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        diffusion_steps=diffusion_steps,
        t_eps=t_eps,
    )
    step = (t_next - t_value).to(dtype=sample.dtype)
    return sample + step * velocity


def _legacy_heun_step(
    *,
    model: torch.nn.Module,
    sample: torch.Tensor,
    t_value: torch.Tensor,
    t_next: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    diffusion_steps: int,
    t_eps: float,
) -> torch.Tensor:
    velocity_t = _legacy_velocity(
        model=model,
        sample=sample,
        t_value=t_value,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        diffusion_steps=diffusion_steps,
        t_eps=t_eps,
    )
    step = (t_next - t_value).to(dtype=sample.dtype)
    sample_euler = sample + step * velocity_t
    velocity_t_next = _legacy_velocity(
        model=model,
        sample=sample_euler,
        t_value=t_next,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        diffusion_steps=diffusion_steps,
        t_eps=t_eps,
    )
    return sample + step * (0.5 * (velocity_t + velocity_t_next))


# ---------------------------------------------------------------------------
# Top-level ODE sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_legacy_ode(
    *,
    method: str,
    model: torch.nn.Module,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    predict_xstart: bool,
    diffusion_steps: int = 1000,
    noise_schedule: str = "linear",
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    t_eps: float = 5e-2,
    noise_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample using the legacy (dev-branch) Euler/Heun ODE formulation.

    This is designed for models trained with the DDPM forward process on the
    dev branch.  It differs from the current ``sample_with_jit_ode`` in two
    ways:

    * Timestep mapping: continuous ODE t is converted to discrete DDPM
      timesteps that the model's sinusoidal embedding was trained on.
    * Velocity formula: ``v = (x_0_pred - x_t) / (1 - t)`` instead of
      ``v = x_1 - x_0_pred``.
    """
    if not predict_xstart:
        raise ValueError(
            f"Legacy {method} sampling requires predict_xstart=True because "
            "velocity is derived from predicted x_0."
        )
    if num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    low, high = cfg_interval
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError(f"cfg_interval must satisfy 0 <= low <= high <= 1, got {cfg_interval}")

    shape = _validate_sampling_shape(model, shape)
    sample = noise_scale * torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    # Integrate from t=0 (noise) to t=1 (clean) — dev-branch convention.
    timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    if method == "euler":
        stepper = _legacy_euler_step
    elif method == "heun":
        stepper = _legacy_heun_step
    else:
        raise ValueError(f"Unknown legacy ODE method {method!r}")

    for step_idx in range(max(0, num_inference_steps - 1)):
        sample = stepper(
            model=model,
            sample=sample,
            t_value=timesteps[step_idx],
            t_next=timesteps[step_idx + 1],
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            diffusion_steps=diffusion_steps,
            t_eps=t_eps,
        )

    # Final step always uses Euler (matching dev-branch behaviour).
    sample = _legacy_euler_step(
        model=model,
        sample=sample,
        t_value=timesteps[-2],
        t_next=timesteps[-1],
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        diffusion_steps=diffusion_steps,
        t_eps=t_eps,
    )
    if was_training:
        model.train()

    return sample


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_model_legacy(
    *,
    sampler: str,
    model: torch.nn.Module,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    solver_order: int = 2,
    algorithm_type: str = "dpmsolver++",
    solver_type: str = "midpoint",
    timestep_spacing: str = "trailing",
    use_karras_sigmas: bool = False,
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    t_eps: float = 5e-2,
    noise_scale: float = 1.0,
    ddim_eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from a DDPM-trained (dev-branch) model.

    Euler / Heun use the corrected legacy ODE with proper discrete-timestep
    mapping.  DDPM / DPM / DDIM delegate to the standard implementations
    which already handle integer timesteps correctly.
    """
    if sampler in {"heun", "euler"}:
        return sample_with_legacy_ode(
            method=sampler,
            model=model,
            shape=shape,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            device=device,
            predict_xstart=predict_xstart,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            t_eps=t_eps,
            noise_scale=noise_scale,
            generator=generator,
        )
    if sampler == "dpm":
        return sample_with_dpm(
            model=model,
            shape=shape,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            device=device,
            predict_xstart=predict_xstart,
            noise_schedule=noise_schedule,
            diffusion_steps=diffusion_steps,
            solver_order=solver_order,
            algorithm_type=algorithm_type,
            solver_type=solver_type,
            timestep_spacing=timestep_spacing,
            use_karras_sigmas=use_karras_sigmas,
            generator=generator,
        )
    if sampler == "ddpm":
        return sample_with_ddpm(
            model=model,
            shape=shape,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            device=device,
            predict_xstart=predict_xstart,
            noise_schedule=noise_schedule,
            diffusion_steps=diffusion_steps,
            generator=generator,
        )
    if sampler == "ddim":
        return sample_with_ddim(
            model=model,
            shape=shape,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            device=device,
            predict_xstart=predict_xstart,
            noise_schedule=noise_schedule,
            diffusion_steps=diffusion_steps,
            eta=ddim_eta,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            generator=generator,
        )
    raise ValueError(f"Unknown sampler {sampler!r}. Available: {', '.join(SAMPLER_CHOICES)}")
