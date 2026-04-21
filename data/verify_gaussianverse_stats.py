#!/usr/bin/env python3
"""Verify GaussianVerse normalization statistics against the actual dataset.

This script mirrors the standard loader preprocessing path used by
``dataloaders/standard_3dgen_loader.py``:

1. Load ``point_cloud.ply`` for each object.
2. Invert ``gs2sphere.npy`` so the point cloud is in sphere order.
3. Accumulate raw per-channel mean/std over all points across all objects.
4. Optionally apply the provided normalization stats and measure the residual
   mean/std after normalization.

The goal is to confirm that the precomputed ``gaussianverse_mean.pt`` and
``gaussianverse_std.pt`` files match the current data and preprocessing path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloaders.standard_3dgen_loader import extract_directory_info, load_ply  # noqa: E402


_WORKER_STATE: dict[str, object] = {}
MAX_SKIP_EXAMPLES = 10


def _load_stats_tensor(path: str) -> np.ndarray:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")

    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(tensor, dtype=np.float64)


def _load_obj_lists(paths: Sequence[str]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for obj_list_path in paths:
        with open(obj_list_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected object list JSON dict at {obj_list_path}, got {type(data)!r}")
        merged.update(data)
    return merged


def _iter_chunks(items: Sequence[str], chunk_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def _load_and_reorder_point_cloud(gs_path: str, tar_gz_path: str) -> np.ndarray:
    directory_number, filename = extract_directory_info(tar_gz_path)
    data_dir = Path(gs_path) / directory_number / filename

    gs2sphere = np.load(str(data_dir / "gs2sphere.npy"))
    point_cloud = load_ply(str(data_dir / "point_cloud.ply"))

    if gs2sphere.ndim != 1:
        raise ValueError(f"Expected 1D gs2sphere, got shape {gs2sphere.shape} for {tar_gz_path}")
    if gs2sphere.shape[0] != point_cloud.shape[0]:
        raise ValueError(
            "Point count mismatch for "
            f"{tar_gz_path}: point_cloud={point_cloud.shape[0]} vs gs2sphere={gs2sphere.shape[0]}"
        )

    sphere_to_gs = np.empty_like(gs2sphere)
    sphere_to_gs[gs2sphere] = np.arange(gs2sphere.shape[0], dtype=gs2sphere.dtype)
    return point_cloud[sphere_to_gs]


def _accumulate_stats(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    pc64 = np.asarray(point_cloud, dtype=np.float64)
    return pc64.sum(axis=0), np.square(pc64).sum(axis=0), int(pc64.shape[0])


def _accumulate_normalized_stats(
    point_cloud: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    pc64 = np.asarray(point_cloud, dtype=np.float64)
    norm = (pc64 - mean[None, :]) / (std[None, :] + 1e-8)
    return norm.sum(axis=0), np.square(norm).sum(axis=0), int(norm.shape[0])


def _process_chunk(
    tar_paths: list[str],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    int,
    np.ndarray | None,
    np.ndarray | None,
    int,
    int,
    list[str],
]:
    state = _WORKER_STATE
    gs_path = state["gs_path"]
    mean = state["mean"]
    std = state["std"]
    compute_normalized = state["compute_normalized"]
    skip_missing = state["skip_missing"]

    raw_sum = None
    raw_sumsq = None
    raw_count = 0
    norm_sum = None
    norm_sumsq = None
    norm_count = 0
    skipped_objects = 0
    skipped_examples: list[str] = []

    for tar_gz_path in tar_paths:
        try:
            point_cloud = _load_and_reorder_point_cloud(gs_path, tar_gz_path)
        except FileNotFoundError as exc:
            if not skip_missing:
                raise
            skipped_objects += 1
            if len(skipped_examples) < MAX_SKIP_EXAMPLES:
                skipped_examples.append(f"{tar_gz_path}: {exc.filename}")
            continue

        sample_sum, sample_sumsq, sample_count = _accumulate_stats(point_cloud)
        if raw_sum is None:
            raw_sum = np.zeros_like(sample_sum, dtype=np.float64)
            raw_sumsq = np.zeros_like(sample_sumsq, dtype=np.float64)
        raw_sum += sample_sum
        raw_sumsq += sample_sumsq
        raw_count += sample_count

        if compute_normalized:
            sample_norm_sum, sample_norm_sumsq, sample_norm_count = _accumulate_normalized_stats(
                point_cloud, mean, std
            )
            if norm_sum is None:
                norm_sum = np.zeros_like(sample_norm_sum, dtype=np.float64)
                norm_sumsq = np.zeros_like(sample_norm_sumsq, dtype=np.float64)
            norm_sum += sample_norm_sum
            norm_sumsq += sample_norm_sumsq
            norm_count += sample_norm_count

    if raw_sum is None or raw_sumsq is None:
        if skipped_objects > 0 and skip_missing:
            return None, None, 0, None, None, 0, skipped_objects, skipped_examples
        raise ValueError("Empty chunk encountered unexpectedly")

    return raw_sum, raw_sumsq, raw_count, norm_sum, norm_sumsq, norm_count, skipped_objects, skipped_examples


def _worker_init(
    gs_path: str,
    mean: np.ndarray,
    std: np.ndarray,
    compute_normalized: bool,
    skip_missing: bool,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "gs_path": gs_path,
        "mean": mean,
        "std": std,
        "compute_normalized": compute_normalized,
        "skip_missing": skip_missing,
    }


def _finalize_stats(total_sum: np.ndarray, total_sumsq: np.ndarray, total_count: int) -> tuple[np.ndarray, np.ndarray]:
    if total_count <= 0:
        raise ValueError("No points were accumulated")
    mean = total_sum / total_count
    var = total_sumsq / total_count - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std


def _print_diff_summary(name: str, computed: np.ndarray, reference: np.ndarray) -> None:
    diff = np.abs(computed - reference)
    print(
        f"{name}: mean_abs_diff={diff.mean():.6e} "
        f"max_abs_diff={diff.max():.6e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify GaussianVerse normalization statistics")
    parser.add_argument("--obj_list", type=str, nargs="+", required=True, help="One or more obj list JSON files")
    parser.add_argument("--gs_path", type=str, required=True, help="GaussianVerse root directory")
    parser.add_argument("--mean_file", type=str, required=True, help="Precomputed mean .pt file")
    parser.add_argument("--std_file", type=str, required=True, help="Precomputed std .pt file")
    parser.add_argument("--limit", type=int, default=0, help="Limit verification to N objects (after optional shuffle)")
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="If set, shuffle objects with this seed before applying --limit (for unbiased sampling)",
    )
    parser.add_argument("--workers", type=int, default=0, help="Worker processes to use; 0 selects automatically")
    parser.add_argument("--chunk_size", type=int, default=32, help="Number of objects per worker task")
    parser.add_argument(
        "--skip_normalized",
        action="store_true",
        help="Skip the normalized mean/std residual check",
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip objects whose required local files are missing and report how many were skipped",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        default=None,
        help="If set, dump per-channel computed mean/std (raw and normalized) as .npy files into this dir",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    obj_data = _load_obj_lists(args.obj_list)
    tar_paths = list(obj_data.values())
    if args.shuffle_seed is not None:
        rng = np.random.default_rng(args.shuffle_seed)
        perm = rng.permutation(len(tar_paths))
        tar_paths = [tar_paths[i] for i in perm]
    if args.limit > 0:
        tar_paths = tar_paths[: args.limit]
    if not tar_paths:
        raise ValueError("No objects available after applying --limit")

    precomputed_mean = _load_stats_tensor(args.mean_file)
    precomputed_std = _load_stats_tensor(args.std_file)
    if precomputed_mean.shape != precomputed_std.shape:
        raise ValueError(
            f"Mean/std shape mismatch: mean={precomputed_mean.shape}, std={precomputed_std.shape}"
        )

    channel_count = int(precomputed_mean.shape[0])
    compute_normalized = not args.skip_normalized
    skip_missing = args.skip_missing

    if args.workers <= 0:
        workers = min(len(tar_paths), os.cpu_count() or 1)
    else:
        workers = min(len(tar_paths), args.workers)
    workers = max(1, workers)

    print(
        f"Loaded {len(tar_paths)} objects from {len(args.obj_list)} obj list file(s); "
        f"channels={channel_count}; workers={workers}; chunk_size={args.chunk_size}"
    )
    if args.limit > 0:
        print(f"Limit enabled: first {args.limit} objects")

    raw_sum = np.zeros(channel_count, dtype=np.float64)
    raw_sumsq = np.zeros(channel_count, dtype=np.float64)
    raw_count = 0
    norm_sum = np.zeros(channel_count, dtype=np.float64) if compute_normalized else None
    norm_sumsq = np.zeros(channel_count, dtype=np.float64) if compute_normalized else None
    norm_count = 0
    skipped_objects = 0
    skipped_examples: list[str] = []

    chunks = list(_iter_chunks(tar_paths, args.chunk_size))
    if workers == 1:
        _worker_init(args.gs_path, precomputed_mean, precomputed_std, compute_normalized, skip_missing)
        for chunk in chunks:
            chunk_result = _process_chunk(chunk)
            if chunk_result[0] is not None and chunk_result[1] is not None:
                raw_sum += chunk_result[0]
                raw_sumsq += chunk_result[1]
                raw_count += chunk_result[2]
            if compute_normalized and chunk_result[3] is not None and chunk_result[4] is not None:
                norm_sum += chunk_result[3]
                norm_sumsq += chunk_result[4]
                norm_count += chunk_result[5]
            skipped_objects += chunk_result[6]
            for example in chunk_result[7]:
                if len(skipped_examples) >= MAX_SKIP_EXAMPLES:
                    break
                skipped_examples.append(example)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(args.gs_path, precomputed_mean, precomputed_std, compute_normalized, skip_missing),
        ) as executor:
            future_map = {executor.submit(_process_chunk, chunk): len(chunk) for chunk in chunks}
            for future in as_completed(future_map):
                chunk_result = future.result()
                if chunk_result[0] is not None and chunk_result[1] is not None:
                    raw_sum += chunk_result[0]
                    raw_sumsq += chunk_result[1]
                    raw_count += chunk_result[2]
                if compute_normalized and chunk_result[3] is not None and chunk_result[4] is not None:
                    norm_sum += chunk_result[3]
                    norm_sumsq += chunk_result[4]
                    norm_count += chunk_result[5]
                skipped_objects += chunk_result[6]
                for example in chunk_result[7]:
                    if len(skipped_examples) >= MAX_SKIP_EXAMPLES:
                        break
                    skipped_examples.append(example)

    processed_objects = len(tar_paths) - skipped_objects
    print(f"Processed objects: {processed_objects:,}")
    if skipped_objects > 0:
        print(f"Skipped objects with missing local files: {skipped_objects:,}")
        for example in skipped_examples:
            print(f"  skipped: {example}")
    if processed_objects == 0:
        print("No complete local objects were available to accumulate statistics.")
        return 0

    raw_mean, raw_std = _finalize_stats(raw_sum, raw_sumsq, raw_count)
    print(f"Raw accumulation: points={raw_count:,}")
    _print_diff_summary("Raw mean vs precomputed mean", raw_mean, precomputed_mean)
    _print_diff_summary("Raw std vs precomputed std", raw_std, precomputed_std)

    norm_mean = None
    norm_std = None
    if compute_normalized:
        norm_mean, norm_std = _finalize_stats(norm_sum, norm_sumsq, norm_count)
        print(f"Normalized accumulation: points={norm_count:,}")
        _print_diff_summary("Normalized mean vs 0", norm_mean, np.zeros_like(norm_mean))
        _print_diff_summary("Normalized std vs 1", norm_std, np.ones_like(norm_std))

    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(dump_dir / "raw_mean.npy", raw_mean)
        np.save(dump_dir / "raw_std.npy", raw_std)
        np.save(dump_dir / "precomputed_mean.npy", precomputed_mean)
        np.save(dump_dir / "precomputed_std.npy", precomputed_std)
        np.save(dump_dir / "raw_count.npy", np.asarray([raw_count], dtype=np.int64))
        if norm_mean is not None and norm_std is not None:
            np.save(dump_dir / "norm_mean.npy", norm_mean)
            np.save(dump_dir / "norm_std.npy", norm_std)
        print(f"Dumped per-channel arrays to {dump_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
