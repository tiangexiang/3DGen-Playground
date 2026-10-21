"""Repository-relative assets and environment-configured data/output paths."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def asset(value: str | os.PathLike[str]) -> str:
    """Expand environment/home variables and anchor relative paths to this checkout."""
    path = Path(os.path.expandvars(os.fspath(value))).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _env_path(name: str, default: str | Path) -> Path:
    return Path(asset(os.environ.get(name, default)))


DATA_ROOT = _env_path("DGEN_DATA_ROOT", REPO_ROOT / "datasets")
ASSET_ROOT = _env_path("DGEN_ASSET_ROOT", DATA_ROOT / "baseline_assets")
WORK_ROOT = _env_path("DGEN_WORK_ROOT", REPO_ROOT / "output")
OUTPUT_ROOT = _env_path("DGEN_OUTPUT_ROOT", WORK_ROOT / "runs")
T5_TOKEN_ROOT = _env_path("DGEN_T5_TOKEN_ROOT", WORK_ROOT / "t5_tokens")
T5_NAME = os.environ.get("DGEN_T5_NAME", "google/flan-t5-large")
GS_PATH = _env_path("GS_PATH", DATA_ROOT / "3dgs")
OBJ_LIST = _env_path("DGEN_OBJ_LIST", os.environ.get("ALL_OBJ_JSON", GS_PATH / "all_obj_list.json"))
CAPTIONS_PATH = _env_path("CAPTIONS_PATH", DATA_ROOT / "captions.json")
WANDB_PROJECT = os.environ.get("DGEN_WANDB_PROJECT", "3dgen")
CACHE_ROOT = _env_path("DGEN_CACHE_ROOT", WORK_ROOT / "cache" / "preload_text")


def repository_revision() -> dict:
    """Record Git provenance when available, including source archives without Git."""
    def git(*args):
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
