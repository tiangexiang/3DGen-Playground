"""Sampling and offline evaluation entry points."""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from . import diffusion
from .config import TrainConfig
from .model import build_model
from .render import build_render_context, render_containers, save_grid

logger = logging.getLogger(__name__)


def load_for_inference(cfg: TrainConfig, checkpoint: str, target: str, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    key = "ema" if target == "ema" else "model"
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ck[key].items()}
    # strict=True on purpose: a prefix mismatch here loads NOTHING and produces
    # confident noise that looks like an undertrained model.
    model.load_state_dict(sd, strict=True)
    prov = ck.get("provenance", {})
    logger.info("loaded %s[%s] step=%s (git %s%s)", checkpoint, key, ck.get("step"),
                str(prov.get("git_sha", prov.get("playground_sha")))[:8],
                " DIRTY" if prov.get("git_dirty", prov.get("playground_dirty")) else "")
    return model.to(device).eval(), ck


def encode_prompts(cfg: TrainConfig, prompts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode free-text prompts with the same frozen T5 used for training."""
    from transformers import AutoTokenizer, T5EncoderModel
    tok = AutoTokenizer.from_pretrained(cfg.t5_name, model_max_length=cfg.prompt_length)
    enc = T5EncoderModel.from_pretrained(cfg.t5_name).to(device).eval()
    with torch.no_grad():
        batch = tok(prompts, padding="max_length", max_length=cfg.prompt_length,
                    truncation=True, return_tensors="pt").to(device)
        out = enc(**batch).last_hidden_state
        out = out * batch["attention_mask"][:, :, None].to(out.dtype)
    del enc
    return out.float(), batch["attention_mask"].bool()


def sample_cli(cfg: TrainConfig, args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_for_inference(cfg, args.checkpoint, args.checkpoint_target, device)
    prompts = args.prompt or ["a blue phone booth"]
    ctx = build_render_context(cfg, device)
    tokens, mask = encode_prompts(cfg, prompts, device)
    x = diffusion.euler_sample(
        model, tokens, mask, (cfg.in_channels, cfg.container_size, cfg.container_size),
        steps=args.steps or cfg.n_T, cfg_scale=args.cfg_scale or cfg.cfg_scale,
        noise_scale=cfg.noise_scale, v_clamp_min=cfg.sampler_v_clamp_min,
            exact_final=cfg.sampler_exact_final,
        device=device, progress=True,
    )
    imgs = render_containers(ctx, x, cam_indices=[0])
    out = args.out or str(Path(cfg.output_dir) / "samples" / "cli_sample.png")
    save_grid(imgs.reshape(-1, *imgs.shape[-3:]), out)
    torch.save(x.cpu(), Path(out).with_suffix(".containers.pt"))
    print(f"wrote {out} ({len(prompts)} prompt(s))", flush=True)


def eval_cli(cfg: TrainConfig, args) -> None:
    """CMMD over the held-out split."""
    from .data import build_dataset, make_loader
    from .metrics import ClipFeatureExtractor
    from .text import T5TokenStore
    from .train import run_cmmd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_for_inference(cfg, args.checkpoint, args.checkpoint_target, device)
    ctx = build_render_context(cfg, device)
    t5 = T5TokenStore(cfg.t5_token_dir, expect_len=cfg.prompt_length,
                      expect_dim=cfg.txt_input_size)
    held = build_dataset(cfg, "heldout", t5_store=t5)
    loader = make_loader(cfg, held, shuffle=False, batch_size=cfg.cmmd_batch_size,
                         num_workers=2)
    n = args.num_samples or cfg.cmmd_num_samples
    score = run_cmmd(cfg, model, ctx, loader, device, ClipFeatureExtractor(device), n)
    print(f"CMMD ({n} samples, cfg_scale={cfg.cfg_scale}): {score:.4f}", flush=True)
