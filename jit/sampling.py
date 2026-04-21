from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from jit.diffusion import create_diffusion
from jit.diffusion.gaussian_diffusion import get_named_beta_schedule

if TYPE_CHECKING:
    from diffusers import DPMSolverMultistepScheduler

SAMPLER_CHOICES = ("heun", "euler", "dpm", "ddpm", "ddim")


def resolve_sampling_shape(
    *,
    model: torch.nn.Module,
    batch_size: int,
    in_channels: int,
) -> tuple[int, int, int, int]:
    spatial_fold_factor = int(getattr(model, "spatial_fold_factor", 1))
    sample_size = int(getattr(model, "sample_size", getattr(model, "input_size", 128)))
    model_in_channels = int(getattr(model, "in_channels", in_channels))

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if spatial_fold_factor != 1:
        raise ValueError(
            f"JiT sampling does not support spatial folding; expected spatial_fold_factor=1, "
            f"got {spatial_fold_factor}"
        )
    if model_in_channels != in_channels:
        raise ValueError(
            f"Model in_channels={model_in_channels} does not match expected input channels={in_channels}"
        )
    if sample_size < 1:
        raise ValueError(f"sample_size must be >= 1, got {sample_size}")

    return (batch_size, in_channels, sample_size, sample_size)


def _validate_sampling_shape(model: torch.nn.Module, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError(f"Expected sampling shape (N, C, H, W), got {shape}")

    batch_size, in_channels, height, width = (int(dim) for dim in shape)
    if height != width:
        raise ValueError(f"Sampling shape must be square, got H={height}, W={width}")

    expected_shape = resolve_sampling_shape(
        model=model,
        batch_size=batch_size,
        in_channels=in_channels,
    )
    if expected_shape != (batch_size, in_channels, height, width):
        raise ValueError(
            f"Sampling shape {shape} does not match model expectation {expected_shape}. "
            "The unfolded shape must match the model sample size and input channel count."
        )
    return expected_shape


def build_dpm_scheduler(
    *,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    solver_order: int = 2,
    algorithm_type: str = "dpmsolver++",
    solver_type: str = "midpoint",
    timestep_spacing: str = "trailing",
    use_karras_sigmas: bool = False,
) -> "DPMSolverMultistepScheduler":
    from diffusers import DPMSolverMultistepScheduler

    prediction_type = "sample" if predict_xstart else "epsilon"
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    return DPMSolverMultistepScheduler(
        num_train_timesteps=diffusion_steps,
        trained_betas=betas,
        solver_order=solver_order,
        prediction_type=prediction_type,
        algorithm_type=algorithm_type,
        solver_type=solver_type,
        lower_order_final=True,
        use_karras_sigmas=use_karras_sigmas,
        timestep_spacing=timestep_spacing,
    )


def build_ddpm_diffusion(
    *,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    num_inference_steps: int = 1000,
):
    if num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    return create_diffusion(
        timestep_respacing=str(num_inference_steps),
        noise_schedule=noise_schedule,
        learn_sigma=False,
        predict_xstart=predict_xstart,
        diffusion_steps=diffusion_steps,
    )


def _jit_velocity_from_xstart(
    *,
    model: torch.nn.Module,
    sample: torch.Tensor,
    t_value: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    t_eps: float,
    diffusion_steps: int = 1000,
) -> torch.Tensor:
    """Evaluate velocity at continuous time t by predicting x₀ and deriving v = (x₀ - x_t) / (1 - t).

    The model outputs a prediction of the clean data x₀.  Velocity is then
    computed as  v = (x₀_pred - x_t) / (1 - t), clamped by t_eps to avoid
    division by zero near t = 1.
    """
    model_dtype = next(model.parameters()).dtype
    # Scale continuous t ∈ [0, 1] to discrete integer range for the model's timestep embedding.
    t_discrete = (t_value * (diffusion_steps - 1)).round().clamp(0, diffusion_steps - 1).long()
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


def _jit_euler_step(
    *,
    model: torch.nn.Module,
    sample: torch.Tensor,
    t_value: torch.Tensor,
    t_next: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    t_eps: float,
    diffusion_steps: int = 1000,
) -> torch.Tensor:
    velocity = _jit_velocity_from_xstart(
        model=model,
        sample=sample,
        t_value=t_value,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        t_eps=t_eps,
        diffusion_steps=diffusion_steps,
    )
    step = (t_next - t_value).to(dtype=sample.dtype)
    return sample + step * velocity


def _jit_heun_step(
    *,
    model: torch.nn.Module,
    sample: torch.Tensor,
    t_value: torch.Tensor,
    t_next: torch.Tensor,
    class_labels: torch.Tensor,
    cfg_scale: float,
    cfg_interval: tuple[float, float],
    t_eps: float,
    diffusion_steps: int = 1000,
) -> torch.Tensor:
    velocity_t = _jit_velocity_from_xstart(
        model=model,
        sample=sample,
        t_value=t_value,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        t_eps=t_eps,
        diffusion_steps=diffusion_steps,
    )
    step = (t_next - t_value).to(dtype=sample.dtype)
    sample_euler = sample + step * velocity_t
    velocity_t_next = _jit_velocity_from_xstart(
        model=model,
        sample=sample_euler,
        t_value=t_next,
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        t_eps=t_eps,
        diffusion_steps=diffusion_steps,
    )
    return sample + step * (0.5 * (velocity_t + velocity_t_next))


@torch.no_grad()
def sample_with_jit_ode(
    *,
    method: str,
    model: torch.nn.Module,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    predict_xstart: bool,
    diffusion_steps: int = 1000,
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    t_eps: float = 5e-2,
    noise_scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not predict_xstart:
        raise ValueError(
            f"JiT {method} sampling requires predict_xstart=True because the sampler integrates "
            "velocity derived from predicted x0, matching the upstream JiT formulation."
        )
    if num_inference_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
    low, high = cfg_interval
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError(f"cfg_interval must satisfy 0 <= low <= high <= 1, got {cfg_interval}")
    if t_eps <= 0.0:
        raise ValueError(f"t_eps must be > 0, got {t_eps}")

    shape = _validate_sampling_shape(model, shape)
    sample = noise_scale * torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    if method == "euler":
        stepper = _jit_euler_step
    elif method == "heun":
        stepper = _jit_heun_step
    else:
        raise ValueError(f"Unknown JiT ODE method {method!r}")

    for step_idx in range(max(0, num_inference_steps - 1)):
        sample = stepper(
            model=model,
            sample=sample,
            t_value=timesteps[step_idx],
            t_next=timesteps[step_idx + 1],
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            t_eps=t_eps,
            diffusion_steps=diffusion_steps,
        )

    sample = _jit_euler_step(
        model=model,
        sample=sample,
        t_value=timesteps[-2],
        t_next=timesteps[-1],
        class_labels=class_labels,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        t_eps=t_eps,
        diffusion_steps=diffusion_steps,
    )
    if was_training:
        model.train()

    return sample


@torch.no_grad()
def sample_with_dpm(
    *,
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
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    shape = _validate_sampling_shape(model, shape)
    scheduler = build_dpm_scheduler(
        predict_xstart=predict_xstart,
        noise_schedule=noise_schedule,
        diffusion_steps=diffusion_steps,
        solver_order=solver_order,
        algorithm_type=algorithm_type,
        solver_type=solver_type,
        timestep_spacing=timestep_spacing,
        use_karras_sigmas=use_karras_sigmas,
    )
    scheduler.set_timesteps(num_inference_steps, device=device)

    sample_dtype = next(model.parameters()).dtype
    sample = torch.randn(shape, device=device, dtype=sample_dtype, generator=generator)
    sample = sample * scheduler.init_noise_sigma

    was_training = model.training
    model.eval()
    for timestep in scheduler.timesteps:
        timestep_batch = torch.full(
            (shape[0],),
            int(timestep.item()),
            device=device,
            dtype=torch.long,
        )
        model_input = scheduler.scale_model_input(sample, timestep)
        model_output = model(model_input, timestep_batch, class_labels)
        sample = scheduler.step(
            model_output,
            timestep,
            sample,
            generator=generator,
            return_dict=False,
        )[0]
    if was_training:
        model.train()

    return sample


@torch.no_grad()
def sample_with_ddim(
    *,
    model: torch.nn.Module,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    eta: float = 0.0,
    cfg_scale: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 1.0),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    shape = _validate_sampling_shape(model, shape)
    diffusion = build_ddpm_diffusion(
        predict_xstart=predict_xstart,
        noise_schedule=noise_schedule,
        diffusion_steps=diffusion_steps,
        num_inference_steps=num_inference_steps,
    )

    sample_dtype = next(model.parameters()).dtype
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)

    was_training = model.training
    model.eval()
    model_kwargs = {"y": class_labels}

    if cfg_scale != 1.0:
        y_embedder = getattr(model, "y_embedder", None)
        num_classes = getattr(y_embedder, "num_classes", None)
        if num_classes is None:
            raise ValueError("DDIM CFG requires model.y_embedder.num_classes to be available")
        null_labels = torch.full_like(class_labels, num_classes)
        low, high = cfg_interval

        def model_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # t here is the remapped original timestep (in [0, diffusion_steps-1])
            t_frac = float(t[0].item()) / max(1, diffusion_steps - 1)
            # Match the CFG interval logic used in _jit_velocity_from_xstart
            apply_cfg = t_frac < high and (low == 0.0 or t_frac > low)
            if not apply_cfg:
                return model(x.to(dtype=sample_dtype), t, y)
            out_cond = model(x.to(dtype=sample_dtype), t, y)
            out_uncond = model(x.to(dtype=sample_dtype), t, null_labels)
            return out_uncond + cfg_scale * (out_cond - out_uncond)
    else:
        def model_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return model(x.to(dtype=sample_dtype), t, y)

    sample = diffusion.ddim_sample_loop(
        model_fn,
        shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        eta=eta,
    )
    if was_training:
        model.train()

    return sample


@torch.no_grad()
def sample_with_ddpm(
    *,
    model: torch.nn.Module,
    shape: tuple[int, ...],
    class_labels: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    predict_xstart: bool,
    noise_schedule: str = "linear",
    diffusion_steps: int = 1000,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    shape = _validate_sampling_shape(model, shape)
    diffusion = build_ddpm_diffusion(
        predict_xstart=predict_xstart,
        noise_schedule=noise_schedule,
        diffusion_steps=diffusion_steps,
        num_inference_steps=num_inference_steps,
    )

    sample_dtype = next(model.parameters()).dtype
    sample = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)

    was_training = model.training
    model.eval()
    model_kwargs = {"y": class_labels}

    def model_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return model(x.to(dtype=sample_dtype), t, y)

    for timestep in reversed(range(diffusion.num_timesteps)):
        timestep_batch = torch.full(
            (shape[0],),
            timestep,
            device=device,
            dtype=torch.long,
        )
        out = diffusion.p_mean_variance(
            model_fn,
            sample,
            timestep_batch,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn(sample.shape, device=device, dtype=sample.dtype, generator=generator)
        nonzero_mask = (timestep_batch != 0).to(dtype=sample.dtype).view(-1, *([1] * (sample.ndim - 1)))
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
    if was_training:
        model.train()

    return sample


@torch.no_grad()
def sample_model(
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
    if sampler in {"heun", "euler"}:
        return sample_with_jit_ode(
            method=sampler,
            model=model,
            shape=shape,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            device=device,
            predict_xstart=predict_xstart,
            diffusion_steps=diffusion_steps,
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
    raise ValueError(f"Unknown sampler {sampler!r}. Available samplers: {', '.join(SAMPLER_CHOICES)}")
