"""Generate frozen text tokens and camera framing from the training configuration."""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys

from typing import TYPE_CHECKING

from utils.training_paths import REPO_ROOT

if TYPE_CHECKING:
    from mini3d.config import TrainConfig

logger = logging.getLogger(__name__)


def prepare(cfg: TrainConfig, *, only: str | None = None,
            text_batch_size: int = 256, device: str | None = None) -> None:
    """Run the preparation tools using the same paths and dimensions as training."""
    cfg.resolve()
    if only not in (None, "text", "framing"):
        raise ValueError(f"Unknown preparation stage: {only}")
    if text_batch_size <= 0:
        raise ValueError("text_batch_size must be positive")
    required = ["obj_list"]
    if only != "framing":
        required.append("caption_path")
    if only != "text":
        required.extend(["gs_path", "sphere2plane_path", "ref_camera_tar"])
        if cfg.exclude_keys_file:
            required.append("exclude_keys_file")
        if not cfg.per_sample_zoom_file:
            raise ValueError("Set per_sample_zoom_file to the framing table output path")
    for name in required:
        path = getattr(cfg, name)
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    if only != "framing":
        command = [
            sys.executable, str(REPO_ROOT / "tools/encode_t5_tokens.py"),
            "--t5-name", cfg.t5_name, "--output", cfg.t5_token_dir,
            "--obj-list", cfg.obj_list, "--captions", cfg.caption_path,
            "--prompt-length", str(cfg.prompt_length),
            "--batch-size", str(text_batch_size),
        ]
        if device:
            command.extend(["--device", device])
        logger.info("Preparing frozen text tokens in %s", cfg.t5_token_dir)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if only != "text":
        command = [
            sys.executable, str(REPO_ROOT / "data/build_per_sample_render_fov.py"),
            "--obj_list", cfg.obj_list, "--gs_path", cfg.gs_path,
            "--sphere2plane_path", cfg.sphere2plane_path,
            "--ref_camera_tar", cfg.ref_camera_tar,
            "--exclude_keys_file", cfg.exclude_keys_file or "",
            "--output", cfg.per_sample_zoom_file,
            "--train_render_size", str(cfg.train_render_size),
            "--num_workers", str(max(1, cfg.num_workers)),
        ]
        logger.info("Preparing camera framing in %s", cfg.per_sample_zoom_file)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
