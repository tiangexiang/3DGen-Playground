"""Typed training configuration with YAML inheritance and CLI overrides."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import os
from pathlib import Path
from typing import Any

import yaml

from utils import training_paths as settings


@dataclass
class TrainConfig:
    # --- run identity ------------------------------------------------------
    project: str = settings.WANDB_PROJECT
    run_name: str = "gaussian-splat-diffusion"
    output_dir: str = str(settings.OUTPUT_ROOT / "gaussian_splat_diffusion")
    seed: int = 42
    wandb: bool = True

    # --- data --------------------------------------------------------------
    asset_dir: str = str(settings.ASSET_ROOT)
    obj_list: str = str(settings.OBJ_LIST)
    gs_path: str = str(settings.GS_PATH)
    caption_path: str = str(settings.CAPTIONS_PATH)
    mean_file: str = "stats/all_mean_postfix.pt"
    std_file: str = "stats/all_std_postfix.pt"
    rank_transform_file: str = "stats/rank_quantiles_8ch_clipped.pt"
    clip_thresholds_file: str = "stats/clip_thresholds_opacity_scales.pt"
    sphere2plane_path: str = "sphere2plane.npy"
    ref_camera_tar: str = "ref_cameras.tar.gz"
    exclude_keys_file: str | None = "data/outlier_keys_8sigma.json"
    per_sample_zoom_file: str | None = "data/stats/per_sample_render_fov.pt"
    heldout_size: int = 5000

    sh_degree0_only: bool = True     # -> 14 of 59 channels (DC_ONLY_FEATURE_INDICES)
    in_channels: int = 14            # kept explicit; cross-checked in validate()
    token_order: str = "hilbert1d"   # hilbert1d | plane2d
    hilbert_bits: int = 8
    num_workers: int = 32            # workers per training process
    dataloader_prefetch: int = 2
    lazy_cache_to_cpu: bool = True
    cache_dir: str = str(settings.CACHE_ROOT)

    # --- text --------------------------------------------------------------
    t5_token_dir: str = str(settings.T5_TOKEN_ROOT)
    t5_name: str = settings.T5_NAME
    prompt_length: int = 96
    txt_input_size: int = 1024       # FLAN-T5-Large hidden

    # --- model -------------------------------------------------------------
    container_size: int = 128        # 128x128 = 16384 splats
    group_size: int = 64             # -> 16384/64 = 256 tokens
    patch_size: int = 16             # matching group size; also plane2d patch width
    hidden_size: int = 768
    txt_hidden_size: int = 768
    cond_vec_size: int = 768
    depth_double: int = 17
    text_preamble_depth: int = 2
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: float = 2.6667
    pca_channels: int = 128          # bottleneck tokenizer rank
    final_layer_zero: bool = True
    attention_impl: str = "sdpa"     # sdpa | einsum
    txt_refine_depth: int | None = None
    time_cond: str = "adaln_img"     # adaln_img | adaln_both | adaln_shared | none
    adaln_gate_init: str = "one"     # one | zero
    pooled_in_vec: bool = True       # vec = t_emb + pooled_embed(masked-mean context)
    gradient_checkpointing: bool = False

    # --- diffusion ---------------------------------------------------------
    prediction: str = "x"            # x0-prediction (the JiT parameterization)
    noise_scale: float = 2.0
    t_lognorm_mu: float = -0.8
    t_lognorm_sigma: float = 0.8
    t_sample_schedule: str = "lognorm"   # lognorm | uniform
    label_drop_rate: float = 0.1     # CFG dropout: zero the attention mask
    v_clamp_min: float = 0.05        # TRAINING clamp on (1-t); caps the v-weight at 400x
    sampler_v_clamp_min: float = 0.05
    sampler_exact_final: bool = True
    n_T: int = 100
    cfg_scale: float = 2.5

    # --- loss --------------------------------------------------------------
    recon_loss: str = "v_mse"        # v_mse | x0_mse
    sinkhorn_weight: float = 0.0
    sinkhorn_mode: str = "sinkhorn_patch_hard"
    sinkhorn_iters: int = 50
    sinkhorn_epsilon: float = 0.05
    huber_delta: float = 2.0
    chamfer_loss_weight: float = 0.07
    chamfer_rev_weight: float = 12.0
    chamfer_subsample: int = 4096
    mse_hybrid_weight: float = 0.0
    mse_hybrid_lownoise_mult: float = 1.0
    render_loss_weight: float = 0.0
    lpips_loss_weight: float = 0.0
    alpha_mask_loss_weight: float = 0.0
    lpips_net: str = "vgg"
    render_loss_noise_cutoff: float = 0.6
    render_fg_phi: float = 0.0
    use_per_sample_zoom: bool = False
    render_metrics_every: int = 0
    render_loss_num_cam: int = 1
    train_render_size: int = 224
    enable_render_loss_after: int = 0

    # --- optimizer ---------------------------------------------------------
    optimizer: str = "adamw"         # adamw | muon
    learning_rate: float = 4e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0       # 0 disables clipping
    lr_schedule: str = "warmup_constant"   # warmup_constant | cosine
    warmup_steps: int = 5000
    lr_min: float = 1e-6
    ema_decay: float = 0.999
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # --- batch / horizon ---------------------------------------------------
    batch_size: int = 512            # global samples per optimizer step
    micro_batch_size: int = 64
    grad_accum_steps: int | None = None
    num_steps: int = 250_000
    amp_dtype: str = "bf16"
    compile_model: bool = False
    ddp_find_unused_parameters: bool = False
    auto_resume: bool = True
    resume_from: str | None = None

    # --- logging / validation ---------------------------------------------
    log_every: int = 100
    defer_loss_sync: bool = True
    sample_every: int = 5_000
    sample_num: int = 8
    ckpt_every: int = 2_000
    save_keep: int = 3
    cmmd_every: int = 20_000
    cmmd_num_samples: int = 2_000
    cmmd_batch_size: int = 32
    val_steps: int = 50              # euler steps for in-loop validation samples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # --- asset resolution --------------------------------------------------
    _REFERENCE_ASSET_KEYS = (
        "mean_file", "std_file", "rank_transform_file", "clip_thresholds_file",
        "sphere2plane_path", "ref_camera_tar",
    )
    _ASSET_KEYS = (
        "mean_file", "std_file", "rank_transform_file", "clip_thresholds_file",
        "sphere2plane_path", "ref_camera_tar", "exclude_keys_file",
        "per_sample_zoom_file",
    )

    _PATH_KEYS = _ASSET_KEYS + (
        "asset_dir", "obj_list", "gs_path", "caption_path", "t5_token_dir", "output_dir", "cache_dir",
    )

    def resolved(self, key: str) -> str | None:
        """Resolve reference files under asset_dir and other paths under the checkout."""
        val = getattr(self, key)
        if val is None:
            return None
        if key in self._REFERENCE_ASSET_KEYS:
            path = Path(os.path.expandvars(val)).expanduser()
            return str(path if path.is_absolute() else Path(settings.asset(self.asset_dir)) / path)
        return settings.asset(val)

    def update_from_mapping(self, values: dict[str, Any]) -> "TrainConfig":
        for key, value in values.items():
            if key not in {item.name for item in fields(self)}:
                raise KeyError(
                    f"unknown TrainConfig key in config file: {key!r}. "
                    "Use a field listed in configs/train.yml or TrainConfig."
                )
            setattr(self, key, value)
        return self

    def update_from_yaml(self, path: str | Path, _seen: tuple[Path, ...] = ()) -> "TrainConfig":
        """Merge a YAML file, following an optional ``_base`` chain first."""
        path = Path(path).expanduser().resolve()
        if path in _seen:
            raise ValueError(f"Cyclic YAML _base inheritance: {path}")
        with path.open("r", encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
        if not isinstance(values, dict):
            raise TypeError(f"config file must contain a YAML mapping: {path}")
        base = values.pop("_base", None)
        if base is not None:
            base_path = Path(base)
            if not base_path.is_absolute():
                base_path = (path.parent / base_path).resolve()
            self.update_from_yaml(base_path, (*_seen, path))
        return self.update_from_mapping(values)

    def validate(self) -> "TrainConfig":
        n_points = self.container_size ** 2
        if self.token_order == "hilbert1d":
            if n_points % self.group_size != 0:
                raise ValueError(
                    f"container_size^2={n_points} not divisible by group_size={self.group_size}"
                )
        elif self.token_order == "plane2d":
            if self.container_size % self.patch_size != 0:
                raise ValueError(
                    f"container_size={self.container_size} not divisible by patch_size={self.patch_size}"
                )
        else:
            raise ValueError(f"token_order must be hilbert1d|plane2d, got {self.token_order!r}")
        if self.adaln_gate_init not in ("one", "zero"):
            raise ValueError(f"adaln_gate_init must be one|zero, got {self.adaln_gate_init!r}")
        if self.txt_refine_depth is not None and not 0 <= self.txt_refine_depth <= self.depth_double:
            raise ValueError(f"txt_refine_depth must be in [0, {self.depth_double}], "
                             f"got {self.txt_refine_depth!r}")
        if self.time_cond not in ("adaln_img", "adaln_both", "adaln_shared", "none"):
            raise ValueError(
                f"time_cond must be adaln_img|adaln_both|adaln_shared|none, got {self.time_cond!r}")
        if self.recon_loss not in ("v_mse", "x0_mse"):
            raise ValueError(f"recon_loss must be v_mse|x0_mse, got {self.recon_loss!r}")
        if self.render_fg_phi < 0 or self.render_fg_phi >= 1:
            raise ValueError(
                f"render_fg_phi must be in [0, 1), got {self.render_fg_phi}. It is a SHARE of "
                "each sample's render weight given to foreground pixels."
            )
        if self.use_per_sample_zoom and not self.per_sample_zoom_file:
            raise ValueError("use_per_sample_zoom=True requires per_sample_zoom_file")
        if self.recon_loss == "v_mse" and self.mse_hybrid_lownoise_mult != 1.0:
            raise ValueError(
                "mse_hybrid_lownoise_mult must be 1.0 under recon_loss=v_mse: the v-space "
                "loss already applies a 1/(1-t)^2 low-noise weight (capped at "
                f"{1.0 / self.v_clamp_min ** 2:.0f}x) and the two stack multiplicatively."
            )
        expect_c = 14 if self.sh_degree0_only else 59
        if self.in_channels != expect_c:
            raise ValueError(
                f"in_channels={self.in_channels} contradicts sh_degree0_only="
                f"{self.sh_degree0_only} (expected {expect_c}). The container channel\n"
                "count drives the tokenizer and the final projection, so a mismatch "
                "here silently trains on the wrong feature layout."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for 1D RoPE, got {self.head_dim}")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch_size and micro_batch_size must be positive")
        if self.grad_accum_steps is not None and self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive or null (automatic)")
        return self

    def resolve_batch(self, world_size: int) -> "TrainConfig":
        per_micro = self.micro_batch_size * world_size
        if self.grad_accum_steps is None:
            if self.batch_size % per_micro:
                raise ValueError(f"batch_size={self.batch_size} must be divisible by "
                                 f"micro_batch_size * world_size={per_micro}")
            self.grad_accum_steps = self.batch_size // per_micro
        actual = per_micro * self.grad_accum_steps
        if actual != self.batch_size:
            raise ValueError(f"Configured global batch {self.batch_size} differs from actual "
                             f"{actual}; set batch_size explicitly to change the recipe")
        return self

    def resolve(self) -> "TrainConfig":
        self.validate()
        for key in self._PATH_KEYS:
            setattr(self, key, self.resolved(key))
        if self.resume_from and self.resume_from != "latest":
            self.resume_from = settings.asset(self.resume_from)
        return self


def num_tokens(cfg: TrainConfig) -> int:
    if cfg.token_order == "hilbert1d":
        return (cfg.container_size ** 2) // cfg.group_size
    return (cfg.container_size // cfg.patch_size) ** 2
