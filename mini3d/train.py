"""Training loop for the Gaussian Splat Diffusion baseline.

Uses Muon/AdamW, constant learning rates after warmup, short EMA,
auxiliary matching/render losses, checkpoint resume, and held-out CMMD."""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import training_paths as settings

from . import aux_losses, diffusion
from .config import TrainConfig
from .data import build_dataset, check_training_data, make_loader
from .model import build_model
from .optim import build_optimizer, set_lr
from .render import build_render_context, render_containers, save_grid
from .text import T5TokenStore
from .utils import (
    amp_dtype, atomic_torch_save, is_main, lr_at, rank, seed_all, unwrap,
    update_ema, world_size,
)

logger = logging.getLogger(__name__)


def init_dist() -> tuple[torch.device, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), local_rank
    return torch.device("cpu"), 0


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------

def save_checkpoint(cfg: TrainConfig, model, ema_model, opt, step: int) -> None:
    if not is_main():
        return
    ckpt_dir = Path(cfg.output_dir) / "checkpoints"
    path = ckpt_dir / f"checkpoint_{step:07d}.pt"
    atomic_torch_save({
        "step": step,
        # unwrap() strips DDP's "module." and compile's "_orig_mod." so the
        # state_dict keys do not depend on how the run happened to be launched.
        "model": unwrap(model).state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg.to_dict(),
        "provenance": settings.repository_revision(),
    }, path)
    ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    for old in ckpts[: -cfg.save_keep]:
        old.unlink(missing_ok=True)


def latest_checkpoint(cfg: TrainConfig) -> Path | None:
    ckpts = sorted((Path(cfg.output_dir) / "checkpoints").glob("checkpoint_*.pt"))
    return ckpts[-1] if ckpts else None


def load_checkpoint_if_available(cfg: TrainConfig, model, ema_model, device):
    path = None
    if cfg.resume_from:
        path = latest_checkpoint(cfg) if cfg.resume_from == "latest" else Path(cfg.resume_from)
    elif cfg.auto_resume:
        path = latest_checkpoint(cfg)
    if path is None:
        return 0, None
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    # Strip DDP/compile prefixes and load strictly to catch checkpoint mismatches.
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    ema_model.load_state_dict(
        {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck["ema"].items()},
        strict=True,
    )
    step = int(ck["step"])
    if is_main():
        prov = ck.get("provenance", {})
        logger.info("Resumed %s at step=%d (git %s%s)", path, step,
                    str(prov.get("git_sha", prov.get("playground_sha")))[:8],
                    " DIRTY" if prov.get("git_dirty", prov.get("playground_dirty")) else "")
    return step, ck.get("optimizer")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@torch.no_grad()
def sample_and_render(cfg: TrainConfig, model, ctx, batch, device, n: int, steps: int,
                      cfg_scale: float) -> torch.Tensor:
    net = unwrap(model)
    was_training = net.training
    net.eval()
    tokens = batch["tokens"][:n].to(device).float()
    mask = batch["mask"][:n].to(device)
    x = diffusion.euler_sample(
        net, tokens, mask,
        (cfg.in_channels, cfg.container_size, cfg.container_size),
        steps=steps, cfg_scale=cfg_scale, noise_scale=cfg.noise_scale,
        v_clamp_min=cfg.sampler_v_clamp_min,
            exact_final=cfg.sampler_exact_final, device=device,
    )
    imgs = render_containers(ctx, x, cam_indices=[0])
    if was_training:
        net.train()
    return imgs


@torch.no_grad()
def run_cmmd(cfg: TrainConfig, model, ctx, loader, device, feat_extractor,
             num_samples: int) -> float | None:
    """CMMD between renders of generated objects and of their held-out ground truth."""
    from .metrics import cmmd
    net = unwrap(model)
    was_training = net.training
    net.eval()
    gen_feats, gt_feats, seen = [], [], 0
    for batch in loader:
        if seen >= num_samples:
            break
        take = min(cfg.cmmd_batch_size, num_samples - seen)
        tokens = batch["tokens"][:take].to(device).float()
        mask = batch["mask"][:take].to(device)
        x = diffusion.euler_sample(
            net, tokens, mask,
            (cfg.in_channels, cfg.container_size, cfg.container_size),
            steps=cfg.val_steps, cfg_scale=cfg.cfg_scale, noise_scale=cfg.noise_scale,
            v_clamp_min=cfg.sampler_v_clamp_min,
            exact_final=cfg.sampler_exact_final, device=device,
        )
        gen = render_containers(ctx, x, cam_indices=[0])
        gt = render_containers(ctx, batch["pc"][:take].to(device), cam_indices=[0])
        gen_feats.append(feat_extractor(gen.reshape(-1, *gen.shape[-3:])))
        gt_feats.append(feat_extractor(gt.reshape(-1, *gt.shape[-3:])))
        seen += take
    if was_training:
        net.train()
    if not gen_feats:
        return None
    return cmmd(torch.cat(gen_feats), torch.cat(gt_feats))


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    cfg.resolve()
    check_training_data(cfg)
    device, local_rank = init_dist()
    cfg.resolve_batch(world_size())
    seed_all(cfg.seed)
    dtype = amp_dtype(cfg.amp_dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main():
        prov = settings.repository_revision()
        logger.info("git %s (%s)%s", str(prov["git_sha"])[:8],
                    prov["git_branch"], " DIRTY" if prov["git_dirty"] else "")
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.output_dir) / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    t5 = T5TokenStore(cfg.t5_token_dir, expect_len=cfg.prompt_length,
                      expect_dim=cfg.txt_input_size)
    if is_main():
        logger.info(t5.describe())

    # Rendering uses the same 14-channel DC targets, so full SH payloads are unnecessary.
    train_ds = build_dataset(cfg, "train", t5_store=t5, return_full_for_render=False)
    loader = make_loader(cfg, train_ds, shuffle=True, distributed=world_size() > 1)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main():
        logger.info("MMJiT3D: %.1fM params, %d image tokens + %d text tokens, time_cond=%s",
                    n_params / 1e6, model.num_tokens, cfg.prompt_length, cfg.time_cond)
        # Log the resolved global batch and sample budget.
        eff = cfg.micro_batch_size * cfg.grad_accum_steps * world_size()
        seen = eff * cfg.num_steps
        logger.info("effective batch = %d x %d x %d ranks = %d | %d steps = %.0fM samples "
                    "= %.0f epochs over %d objects",
                    cfg.micro_batch_size, cfg.grad_accum_steps, world_size(), eff,
                    cfg.num_steps, seen / 1e6, seen / max(len(train_ds), 1), len(train_ds))
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    start_step, opt_state = load_checkpoint_if_available(cfg, model, ema_model, device)

    if cfg.compile_model:
        model = torch.compile(model)
    if world_size() > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=cfg.ddp_find_unused_parameters)

    opt = build_optimizer(cfg, model)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    ctx = None
    if (aux_losses.needs_render(cfg) or cfg.sample_every > 0
            or cfg.render_metrics_every > 0):
        # Metrics require LPIPS even when its training loss is disabled.
        ctx = build_render_context(
            cfg, device,
            want_lpips=(cfg.lpips_loss_weight > 0 or cfg.render_metrics_every > 0),
        )

    val_batch = None
    feat_extractor = None
    held_loader = None
    if is_main() and cfg.sample_every > 0:
        held_ds = build_dataset(cfg, "heldout", t5_store=t5)
        held_loader = make_loader(cfg, held_ds, shuffle=False,
                                  batch_size=cfg.cmmd_batch_size, num_workers=2)
        val_batch = next(iter(held_loader))

    wb = _maybe_wandb(cfg, n_params)

    accum_loss = torch.zeros((), device=device)
    accum_x0 = torch.zeros((), device=device)
    accum_n = 0
    accum_aux: dict[str, float] = {}
    opt.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    t_start = time.time()

    for step in range(start_step, cfg.num_steps):
        for micro in range(cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            x0 = batch["pc"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True)

            sync = (model.no_sync() if hasattr(model, "no_sync")
                    and micro < cfg.grad_accum_steps - 1 else contextlib.nullcontext())
            with sync:
                with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
                    loss, aux = diffusion.training_loss(
                        model, x0, tokens, mask,
                        label_drop_rate=cfg.label_drop_rate,
                        t_schedule=cfg.t_sample_schedule,
                        t_lognorm_mu=cfg.t_lognorm_mu,
                        t_lognorm_sigma=cfg.t_lognorm_sigma,
                        noise_scale=cfg.noise_scale,
                        v_clamp_min=cfg.v_clamp_min,
                        recon_loss=cfg.recon_loss,
                    )
                    if aux_losses.any_aux_enabled(cfg):
                        extra, adiag = aux_losses.compute(cfg, ctx, aux, x0, step,
                                                          keys=batch["key"])
                        loss = loss + extra
                        for k, v in adiag.items():
                            accum_aux[k] = accum_aux.get(k, 0.0) + v
                    loss = loss / cfg.grad_accum_steps
                loss.backward()
            # Zero-weight render METRICS, so an arm training without the render
            # loss still reports numbers comparable to one that does. Outside the
            # autocast/no_sync block and after backward, since it takes no
            # gradient and must not join the DDP reduction.
            if (ctx is not None and cfg.render_metrics_every > 0
                    and micro == 0 and step % cfg.render_metrics_every == 0):
                for k, v in aux_losses.render_metrics(
                        cfg, ctx, aux["pred_x0"], x0, keys=batch["key"]).items():
                    accum_aux[k] = accum_aux.get(k, 0.0) + v
            accum_loss += aux["loss"].detach()
            accum_x0 += aux["x0_mse"].detach()
            accum_n += 1

        lr = lr_at(cfg.lr_schedule, step, cfg.num_steps, cfg.warmup_steps,
                   cfg.learning_rate, cfg.lr_min)
        set_lr(opt, cfg, lr)
        params = [p for p in model.parameters() if p.grad is not None]
        if cfg.max_grad_norm and cfg.max_grad_norm > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        else:
            gnorm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2) for p in params]), 2)
        opt.step()
        opt.zero_grad(set_to_none=True)
        update_ema(ema_model, model, cfg.ema_decay)
        step1 = step + 1

        if is_main() and step1 % cfg.log_every == 0:
            elapsed = time.time() - t_start
            log = {
                "step": step1,
                "loss": float(accum_loss) / max(accum_n, 1),
                "x0_mse": float(accum_x0) / max(accum_n, 1),
                "lr": lr, "grad_norm": float(gnorm),
                "steps_per_sec": (step1 - start_step) / max(elapsed, 1e-6),
                "elapsed_s": elapsed,
            }
            if device.type == "cuda":
                # Peak, not current: the config that OOMs is the one whose PEAK
                # crosses the card, and that peak is reached inside the backward
                # of the largest micro-batch, not at the point we happen to log.
                log["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
            for k, v in accum_aux.items():
                log[k] = v / max(accum_n, 1)
            print(json.dumps(log), flush=True)
            if wb:
                wb.log(log, step=step1)
            accum_loss = torch.zeros((), device=device)
            accum_x0 = torch.zeros((), device=device)
            accum_n = 0
            accum_aux = {}

        if step1 % cfg.ckpt_every == 0:
            save_checkpoint(cfg, model, ema_model, opt, step1)

        if is_main() and cfg.sample_every > 0 and step1 % cfg.sample_every == 0 and val_batch:
            for tag, net in (("sample", model), ("sample_ema", ema_model)):
                imgs = sample_and_render(cfg, net, ctx, val_batch, device,
                                         cfg.sample_num, cfg.val_steps, cfg.cfg_scale)
                out = Path(cfg.output_dir) / "samples" / f"{tag}_{step1:07d}.png"
                save_grid(imgs.reshape(-1, *imgs.shape[-3:]), str(out))
                _wandb_image(wb, tag, out, step1)
            logger.info("wrote validation grids at step %d", step1)

        if (is_main() and cfg.cmmd_every > 0 and step1 % cfg.cmmd_every == 0
                and held_loader is not None):
            if feat_extractor is None:
                from .metrics import ClipFeatureExtractor
                feat_extractor = ClipFeatureExtractor(device)
            score = run_cmmd(cfg, ema_model, ctx, held_loader, device, feat_extractor,
                             cfg.cmmd_num_samples)
            if score is not None:
                logger.info("step %d CMMD(ema) = %.4f", step1, score)
                if wb:
                    wb.log({"cmmd/ema": score}, step=step1)

    save_checkpoint(cfg, model, ema_model, opt, cfg.num_steps)
    if wb:
        wb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


def _maybe_wandb(cfg: TrainConfig, n_params: int):
    if not (cfg.wandb and is_main()) or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
        conf = cfg.to_dict()
        conf["n_params"] = n_params
        conf.update(settings.repository_revision())
        return wandb.init(project=cfg.project, name=cfg.run_name, config=conf)
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)
        return None


def _wandb_image(wb, key: str, path: Path, step: int) -> None:
    if wb is None or not path.exists():
        return
    try:
        import wandb
        wb.log({key: wandb.Image(str(path))}, step=step)
    except Exception as e:
        print(f"[wandb] image log failed: {e}", flush=True)
