#!/usr/bin/env python3
"""Compute GaussianVerse per-channel mean/std over an object list and save as .pt.

Mirrors the Standard3DGenDataset loader path:

1. Load ``point_cloud.ply`` for each object.
2. Invert ``gs2sphere.npy`` so the point cloud is in sphere order (matches the
   order in which the training dataloader produces features).
3. Accumulate per-channel sum / sum-of-squares in float64 across all points.
4. Finalize mean / std and save as ``torch.float64`` tensors of shape ``(C,)``.

The output format matches the existing ``gaussianverse_mean.pt`` /
``gaussianverse_std.pt`` files distributed with the dataset so the training
scripts can consume them unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.verify_gaussianverse_stats import (  # noqa: E402
    _accumulate_stats,
    _iter_chunks,
    _load_and_reorder_point_cloud,
    _load_obj_lists,
)


_WORKER_STATE: dict[str, object] = {}
MAX_SKIP_EXAMPLES = 10


def _process_chunk(
    tar_paths: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, int, int, list[str]]:
    state = _WORKER_STATE
    gs_path: str = state["gs_path"]  # type: ignore[assignment]
    skip_missing: bool = state["skip_missing"]  # type: ignore[assignment]

    raw_sum: np.ndarray | None = None
    raw_sumsq: np.ndarray | None = None
    count = 0
    skipped = 0
    examples: list[str] = []

    for tar_gz_path in tar_paths:
        try:
            point_cloud = _load_and_reorder_point_cloud(gs_path, tar_gz_path)
        except FileNotFoundError as exc:
            if not skip_missing:
                raise
            skipped += 1
            if len(examples) < MAX_SKIP_EXAMPLES:
                examples.append(f"{tar_gz_path}: {exc.filename}")
            continue

        sample_sum, sample_sumsq, sample_count = _accumulate_stats(point_cloud)
        if raw_sum is None:
            raw_sum = np.zeros_like(sample_sum, dtype=np.float64)
            raw_sumsq = np.zeros_like(sample_sumsq, dtype=np.float64)
        raw_sum += sample_sum
        raw_sumsq += sample_sumsq
        count += sample_count

    return raw_sum, raw_sumsq, count, skipped, examples


def _worker_init(gs_path: str, skip_missing: bool) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {"gs_path": gs_path, "skip_missing": skip_missing}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute GaussianVerse mean/std and save as .pt")
    parser.add_argument("--obj_list", type=str, nargs="+", required=True)
    parser.add_argument("--gs_path", type=str, required=True)
    parser.add_argument("--mean_out", type=str, required=True)
    parser.add_argument("--std_out", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (os.cpu_count())")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 = process all objects")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")

    obj_data = _load_obj_lists(args.obj_list)
    tar_paths = list(obj_data.values())
    if args.limit > 0:
        tar_paths = tar_paths[: args.limit]
    if not tar_paths:
        raise ValueError("No objects available after applying --limit")

    workers = args.workers if args.workers > 0 else min(len(tar_paths), os.cpu_count() or 1)
    workers = max(1, workers)
    chunks = list(_iter_chunks(tar_paths, args.chunk_size))

    print(
        f"Computing stats over {len(tar_paths):,} objects "
        f"(workers={workers}, chunk_size={args.chunk_size})"
    )

    total_sum: np.ndarray | None = None
    total_sumsq: np.ndarray | None = None
    total_count = 0
    skipped_total = 0
    example_log: list[str] = []

    def _merge(rs, rsq, cnt, sk, ex):
        nonlocal total_sum, total_sumsq, total_count, skipped_total
        if rs is not None and rsq is not None:
            if total_sum is None:
                total_sum = np.zeros_like(rs)
                total_sumsq = np.zeros_like(rsq)
            total_sum += rs
            total_sumsq += rsq
            total_count += cnt
        skipped_total += sk
        for e in ex:
            if len(example_log) < MAX_SKIP_EXAMPLES:
                example_log.append(e)

    if workers == 1:
        _worker_init(args.gs_path, args.skip_missing)
        for chunk in chunks:
            _merge(*_process_chunk(chunk))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(args.gs_path, args.skip_missing),
        ) as executor:
            future_map = {executor.submit(_process_chunk, chunk): None for chunk in chunks}
            for future in as_completed(future_map):
                _merge(*future.result())

    if total_count <= 0 or total_sum is None or total_sumsq is None:
        raise RuntimeError("No points were accumulated")

    processed = len(tar_paths) - skipped_total
    print(f"Processed objects: {processed:,} ({total_count:,} points)")
    if skipped_total:
        print(f"Skipped (missing local files): {skipped_total:,}")
        for e in example_log:
            print(f"  {e}")

    mean = total_sum / total_count
    var = total_sumsq / total_count - np.square(mean)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    mean_t = torch.from_numpy(mean).to(torch.float64).contiguous()
    std_t = torch.from_numpy(std).to(torch.float64).contiguous()

    Path(args.mean_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.std_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_t, args.mean_out)
    torch.save(std_t, args.std_out)
    print(f"Saved mean ({tuple(mean_t.shape)}, {mean_t.dtype}) to {args.mean_out}")
    print(f"Saved std  ({tuple(std_t.shape)}, {std_t.dtype}) to {args.std_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
