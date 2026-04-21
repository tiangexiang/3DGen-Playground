"""
Inference script for JiT 3DGS models trained with the DDPM path (dev branch).

Uses the legacy Euler/Heun ODE samplers from ``sampling_legacy.py`` which fix
the timestep-mapping bug present in the original dev-branch code (float t
values were passed to a model trained with discrete integer timesteps).

DDPM / DPM / DDIM samplers are also available and work correctly with these
models since they use integer timesteps internally.

Usage:
    python jit/infer_gsplat_legacy.py \
        --checkpoint path/to/checkpoint.pt \
        --ref_camera_tar path/to/ref_camera.tar.gz \
        --mean_file path/to/mean.pt \
        --std_file path/to/std.pt \
        --sphere2plane_path data/sphere2plane.npy \
        --class_map object_labels/object_to_class.json \
        [--sampler euler] [--num_steps 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
import torch
from PIL import Image

# ── Repo / submodule path setup ──────────────────────────────────────────────
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GS_ROOT = os.path.join(REPO_ROOT, "submodules", "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES, FULL_3DGS_FEATURE_DIM
from jit.models import JiT_3DGS_models
from jit.sampling_legacy import SAMPLER_CHOICES, resolve_sampling_shape, sample_model_legacy
from utils.plane_utils import load_sphere2plane
from utils.gsplat_render_util import (
    _denormalize_point_cloud,
    _load_reference_cameras,
    _plane_to_point_cloud_batch,
    _point_clouds_to_gsplat_inputs,
    _prepare_train_cameras,
    _render_gsplat_batch,
    _try_import_renderer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Checkpoint loading ────────────────────────────────────────────────────────

def _load_checkpoint(path: str, device: torch.device) -> dict:
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def _ckpt_arg(ckpt: dict, key: str):
    """Return a stored training arg from the checkpoint, or None."""
    return (ckpt.get("args") or {}).get(key)


def _resolve_arg(cli_value, ckpt: dict, key: str, fallback):
    """Prefer explicit CLI value; fall back to checkpoint-stored arg, then fallback default."""
    if cli_value is not None:
        return cli_value
    stored = _ckpt_arg(ckpt, key)
    if stored is not None:
        return stored
    return fallback


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = _load_checkpoint(args.checkpoint, device)

    # Resolve model architecture: CLI wins, then checkpoint stored args, then default
    model_name   = _resolve_arg(args.model,        ckpt, "model",        "JiT-B/8")
    predict_x0   = _resolve_arg(args.predict_xstart, ckpt, "predict_xstart", True)
    noise_sched  = _resolve_arg(args.noise_schedule, ckpt, "noise_schedule", "linear")
    sh_degree0   = _resolve_arg(args.sh_degree0_only, ckpt, "sh_degree0_only", False)
    cls_dropout  = _resolve_arg(args.class_dropout_prob, ckpt, "class_dropout_prob", 0.1)

    logger.info(
        "Model: %s | predict_xstart=%s | noise_schedule=%s | sh_degree0_only=%s",
        model_name, predict_x0, noise_sched, sh_degree0,
    )
    logger.info("Using LEGACY sampling (DDPM-trained model, corrected timestep mapping)")

    # ── Feature config ────────────────────────────────────────────────────────
    if sh_degree0:
        feature_indices = torch.tensor(DC_ONLY_FEATURE_INDICES, dtype=torch.long)
        in_channels = len(DC_ONLY_FEATURE_INDICES)
        logger.info("sh_degree0_only: %d channels", in_channels)
    else:
        feature_indices = None
        in_channels = FULL_3DGS_FEATURE_DIM

    # ── Class map ─────────────────────────────────────────────────────────────
    with open(args.class_map, "r") as f:
        class_map = json.load(f)
    num_classes = max(v for v in class_map.values() if v >= 0) + 1
    logger.info("Number of classes: %d", num_classes)

    # Validate explicit class label
    if args.class_label is not None and not (0 <= args.class_label < num_classes):
        raise ValueError(
            f"--class_label {args.class_label} out of range [0, {num_classes - 1}]"
        )

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Building model: %s", model_name)
    model = JiT_3DGS_models[model_name](
        input_size=128,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=cls_dropout,
        learn_sigma=False,
    ).to(device)

    # Load weights — prefer EMA unless user opts out
    if args.use_ema and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        logger.info("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded base model weights (no EMA)")
    model.eval()

    step = ckpt.get("step", "?")
    logger.info("Checkpoint step: %s", step)

    # ── Sphere2plane permutation ──────────────────────────────────────────────
    num_points = 128 * 128
    plane_to_sphere = load_sphere2plane(args.sphere2plane_path, num_points)
    logger.info("Loaded sphere2plane permutation")

    # ── Normalization stats ───────────────────────────────────────────────────
    norm_mean: Optional[torch.Tensor] = None
    norm_std:  Optional[torch.Tensor] = None
    if args.mean_file and args.std_file:
        norm_mean_full = torch.load(args.mean_file, weights_only=True).float().cpu()
        norm_std_full  = torch.load(args.std_file,  weights_only=True).float().cpu()
        if feature_indices is not None:
            norm_mean = norm_mean_full[feature_indices]
            norm_std  = norm_std_full[feature_indices]
        else:
            norm_mean = norm_mean_full
            norm_std  = norm_std_full
        logger.info("Loaded normalization stats")

    # ── Renderer setup ────────────────────────────────────────────────────────
    if device.type != "cuda":
        raise RuntimeError("gsplat rendering requires CUDA. Run on a GPU.")

    ref_cameras = _load_reference_cameras(args.ref_camera_tar)
    logger.info("Loaded %d reference cameras", len(ref_cameras))

    renderer = _try_import_renderer()
    if isinstance(renderer, Exception):
        raise RuntimeError(f"gsplat import failed: {renderer}") from renderer

    train_cameras = _prepare_train_cameras(ref_cameras, args.render_size, device)
    num_cams_available = int(train_cameras["viewmats"].shape[0])
    num_render_cams = min(args.num_cameras, num_cams_available)
    logger.info(
        "Renderer ready | resolution=%d | rendering %d/%d cameras per sample",
        args.render_size, num_render_cams, num_cams_available,
    )

    # ── Output dir ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    # ── Sampling loop ─────────────────────────────────────────────────────────
    logger.info(
        "Generating %d sample(s) | sampler=%s | steps=%d | cfg_scale=%.2f | "
        "cfg_interval=(%.2f, %.2f) | t_eps=%.4f | noise_scale=%.3f | predict_xstart=%s",
        args.num_samples, args.sampler, args.num_steps, args.cfg_scale,
        args.cfg_interval[0], args.cfg_interval[1], args.t_eps, args.noise_scale, predict_x0,
    )

    for i in range(args.num_samples):
        y_label = args.class_label if args.class_label is not None else random.randrange(num_classes)
        y = torch.tensor([y_label], dtype=torch.long, device=device)

        sample_shape = resolve_sampling_shape(model=model, batch_size=1, in_channels=in_channels)

        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=device).manual_seed(args.seed + i)

        with torch.no_grad():
            sample = sample_model_legacy(
                sampler=args.sampler,
                model=model,
                shape=sample_shape,
                class_labels=y,
                num_inference_steps=args.num_steps,
                device=device,
                predict_xstart=predict_x0,
                noise_schedule=noise_sched,
                diffusion_steps=args.diffusion_steps,
                solver_order=args.dpm_solver_order,
                algorithm_type=args.dpm_algorithm_type,
                solver_type=args.dpm_solver_type,
                timestep_spacing=args.dpm_timestep_spacing,
                use_karras_sigmas=args.dpm_use_karras_sigmas,
                cfg_scale=args.cfg_scale,
                cfg_interval=tuple(args.cfg_interval),
                t_eps=args.t_eps,
                noise_scale=args.noise_scale,
                ddim_eta=args.ddim_eta,
                generator=generator,
            )

        # Convert atlas sample -> point cloud -> gsplat inputs
        pred_pc = _plane_to_point_cloud_batch(sample.float(), plane_to_sphere)
        pred_pc_raw = _denormalize_point_cloud(pred_pc, norm_mean, norm_std)
        pred_gaussians = _point_clouds_to_gsplat_inputs(
            pred_pc_raw.to(device),
            dc_only=sh_degree0,
            detach_input=True,
        )

        # Pick camera indices: sequential or random
        if args.random_cameras:
            cam_indices = random.sample(range(num_cams_available), num_render_cams)
        else:
            step_size = max(1, num_cams_available // num_render_cams)
            cam_indices = list(range(0, num_cams_available, step_size))[:num_render_cams]

        with torch.no_grad():
            rendered = _render_gsplat_batch(renderer, pred_gaussians, train_cameras, cam_indices, device)
            if rendered.ndim == 5:
                rendered = rendered[0]

        # Save each camera view
        tag = f"sample{i:04d}_class{y_label:03d}_step{step}"
        for cam_idx_pos, cam_idx in enumerate(cam_indices):
            img_tensor = rendered[cam_idx_pos]
            img_np = img_tensor.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
            img_uint8 = (img_np * 255.0).astype(np.uint8)
            fname = f"{tag}_cam{cam_idx:03d}.png"
            out_path = os.path.join(args.output_dir, fname)
            Image.fromarray(img_uint8).save(out_path)

        logger.info(
            "[%d/%d] class=%d | saved %d view(s) -> %s/%s_cam*.png",
            i + 1, args.num_samples, y_label, num_render_cams, args.output_dir, tag,
        )

    logger.info("Done. All outputs in: %s", args.output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "JiT 3DGS inference for DDPM-trained (dev-branch) models. "
            "Uses legacy Euler/Heun ODE samplers with corrected timestep mapping."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",       required=True, help="Path to .pt checkpoint file")
    p.add_argument("--ref_camera_tar",   required=True, help="Path to ref_camera.tar.gz")
    p.add_argument("--mean_file",        required=True, help="Path to normalization mean .pt")
    p.add_argument("--std_file",         required=True, help="Path to normalization std .pt")
    p.add_argument("--sphere2plane_path",required=True, help="Path to sphere2plane.npy")
    p.add_argument("--class_map",        required=True, help="Path to object_to_class.json")

    # ── Model (auto-read from checkpoint if omitted) ──────────────────────────
    g_model = p.add_argument_group("Model (auto-detected from checkpoint if omitted)")
    g_model.add_argument("--model",        type=str, default=None,
                         choices=list(JiT_3DGS_models.keys()),
                         help="JiT model variant")
    g_model.add_argument("--predict_xstart", action=argparse.BooleanOptionalAction, default=None,
                         help="Predict x0 directly instead of epsilon")
    g_model.add_argument("--noise_schedule", type=str, default=None,
                         choices=["linear", "squaredcos_cap_v2"],
                         help="Beta schedule used during training")
    g_model.add_argument("--sh_degree0_only", action=argparse.BooleanOptionalAction, default=None,
                         help="Use only DC SH coefficients (14 channels instead of 59)")
    g_model.add_argument("--class_dropout_prob", type=float, default=None,
                         help="Label dropout prob (must match training)")
    g_model.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True,
                         help="Use EMA weights from checkpoint (recommended)")

    # ── Sampling ─────────────────────────────────────────────────────────────
    g_sample = p.add_argument_group("Sampling")
    g_sample.add_argument("--sampler", type=str, default="euler", choices=SAMPLER_CHOICES,
                          help="Sampling algorithm (euler/heun use legacy ODE)")
    g_sample.add_argument("--num_steps", type=int, default=50,
                          help="Number of denoising steps")
    g_sample.add_argument("--diffusion_steps", type=int, default=1000,
                          help="Total diffusion timesteps (must match training)")

    # CFG
    g_sample.add_argument("--cfg_scale", type=float, default=1.0,
                          help="Classifier-free guidance scale (1.0 = disabled)")
    g_sample.add_argument("--cfg_interval", type=float, nargs=2, default=[0.0, 1.0],
                          metavar=("LOW", "HIGH"),
                          help="CFG is applied only when t in (LOW, HIGH)")

    # Legacy ODE knobs (euler/heun only)
    g_sample.add_argument("--t_eps", type=float, default=5e-2,
                          help="Min t to avoid division by zero in velocity (euler/heun)")
    g_sample.add_argument("--noise_scale", type=float, default=1.0,
                          help="Scale factor on the initial noise (euler/heun)")
    g_sample.add_argument("--ddim_eta", type=float, default=0.0,
                          help="DDIM eta: 0.0 = deterministic, 1.0 ~ DDPM (ddim only)")

    # DPM-Solver knobs (dpm only)
    g_dpm = p.add_argument_group("DPM-Solver (only used when --sampler dpm)")
    g_dpm.add_argument("--dpm_solver_order", type=int, default=2, choices=[1, 2, 3])
    g_dpm.add_argument("--dpm_algorithm_type", type=str, default="dpmsolver++",
                       choices=["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"])
    g_dpm.add_argument("--dpm_solver_type", type=str, default="midpoint",
                       choices=["midpoint", "heun"])
    g_dpm.add_argument("--dpm_timestep_spacing", type=str, default="trailing",
                       choices=["linspace", "leading", "trailing"])
    g_dpm.add_argument("--dpm_use_karras_sigmas", action=argparse.BooleanOptionalAction, default=False)

    # ── Generation ────────────────────────────────────────────────────────────
    g_gen = p.add_argument_group("Generation")
    g_gen.add_argument("--num_samples", type=int, default=4,
                       help="Total number of 3DGS objects to generate")
    g_gen.add_argument("--class_label", type=int, default=None,
                       help="Fixed class index to generate (omit for random per sample)")

    # ── Rendering ─────────────────────────────────────────────────────────────
    g_render = p.add_argument_group("Rendering")
    g_render.add_argument("--render_size", type=int, default=128,
                          help="Rendering resolution (pixels)")
    g_render.add_argument("--num_cameras", type=int, default=4,
                          help="Number of camera views to render per sample")
    g_render.add_argument("--random_cameras", action=argparse.BooleanOptionalAction, default=False,
                          help="Pick camera views randomly (default: evenly spaced)")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, default="output/infer_legacy_results",
                   help="Directory to save rendered PNG images")
    p.add_argument("--seed", type=int, default=None,
                   help="Global RNG seed (None = non-deterministic)")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file; merged before CLI (CLI overrides YAML)")

    return p


def _merge_yaml_into_parser_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    path = Path(config_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        raise ValueError("--config must contain a YAML mapping at the top level")
    allowed = {a.dest for a in parser._actions if a.dest not in ("help", "config")}
    merged: dict[str, Any] = {}
    for k, v in cfg.items():
        if k not in allowed:
            logger.warning("Ignoring unknown config key: %s", k)
            continue
        if v is None:
            continue
        merged[k] = v
    parser.set_defaults(**merged)
    for action in parser._actions:
        if action.dest in merged:
            action.required = False


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    parser = _build_parser()
    if pre_args.config:
        _merge_yaml_into_parser_defaults(parser, pre_args.config)
    args = parser.parse_args()
    main(args)
