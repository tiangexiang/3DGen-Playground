#!/usr/bin/env python3
"""Audit per-channel normalization stats on a representative subset.

Questions:
  1. Do the saved mean/std files match global (over all points, all objects)
     first- and second-moment stats?
  2. After normalizing with the saved stats, is per-channel spatial std ~1.0
     when averaged across objects? Or are some channels systematically smaller
     (implying loss is dominated by a handful of channels)?
  3. Difference between (global std) and (mean over objects of per-object
     spatial std) reveals whether inter-object variance or intra-object
     variance dominates each channel.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.verify_gaussianverse_stats import (
    _load_and_reorder_point_cloud,
    _load_obj_lists,
)
from dataloaders.class_3dgen_loader import DC_ONLY_FEATURE_INDICES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_list", required=True)
    parser.add_argument("--gs_path", required=True)
    parser.add_argument("--mean_file", required=True)
    parser.add_argument("--std_file", required=True)
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dc_only", action="store_true", default=True)
    args = parser.parse_args()

    saved_mean = torch.load(args.mean_file, weights_only=True).cpu().float().numpy()
    saved_std = torch.load(args.std_file, weights_only=True).cpu().float().numpy()
    print(f"Saved mean: shape={saved_mean.shape}, dtype={saved_mean.dtype}")
    print(f"Saved std : shape={saved_std.shape}, dtype={saved_std.dtype}")

    obj_data = _load_obj_lists([args.obj_list])
    tar_paths = list(obj_data.values())
    print(f"Total objects in list: {len(tar_paths):,}")

    rng = random.Random(args.seed)
    rng.shuffle(tar_paths)
    subset = tar_paths[: args.n_samples]
    print(f"Sampling {len(subset)} objects\n")

    # Accumulators (float64).
    C = saved_mean.shape[0]
    global_sum = np.zeros(C, dtype=np.float64)
    global_sumsq = np.zeros(C, dtype=np.float64)
    global_count = 0

    per_obj_spatial_std = []    # list of (C,) arrays
    per_obj_spatial_mean = []   # list of (C,) arrays
    per_obj_norm_spatial_std = []  # normalized by saved_std

    skipped = 0
    for tar_gz in tqdm(subset, desc="objects"):
        try:
            pc = _load_and_reorder_point_cloud(args.gs_path, tar_gz)  # (N, C)
        except FileNotFoundError:
            skipped += 1
            continue

        if pc.ndim != 2 or pc.shape[1] != C:
            raise ValueError(f"Unexpected point cloud shape {pc.shape} vs C={C}")

        pc64 = pc.astype(np.float64, copy=False)
        global_sum += pc64.sum(axis=0)
        global_sumsq += (pc64 ** 2).sum(axis=0)
        global_count += pc64.shape[0]

        per_obj_spatial_std.append(pc64.std(axis=0))
        per_obj_spatial_mean.append(pc64.mean(axis=0))

        normed = (pc.astype(np.float32) - saved_mean.astype(np.float32)) / (
            saved_std.astype(np.float32) + 1e-12
        )
        per_obj_norm_spatial_std.append(normed.std(axis=0))

    if skipped:
        print(f"Skipped (FileNotFoundError): {skipped}")

    n_obj = len(per_obj_spatial_std)
    print(f"\nObjects successfully loaded: {n_obj}\n")

    computed_mean = global_sum / global_count
    computed_var = global_sumsq / global_count - computed_mean ** 2
    computed_var = np.maximum(computed_var, 0.0)
    computed_std = np.sqrt(computed_var)

    per_obj_spatial_std_arr = np.stack(per_obj_spatial_std, axis=0)       # (N_obj, C)
    per_obj_spatial_mean_arr = np.stack(per_obj_spatial_mean, axis=0)     # (N_obj, C)
    per_obj_norm_spatial_std_arr = np.stack(per_obj_norm_spatial_std, axis=0)

    avg_per_obj_std = per_obj_spatial_std_arr.mean(axis=0)
    mean_of_per_obj_means = per_obj_spatial_mean_arr.mean(axis=0)
    std_of_per_obj_means = per_obj_spatial_mean_arr.std(axis=0)

    avg_per_obj_normed_std = per_obj_norm_spatial_std_arr.mean(axis=0)

    print("=" * 140)
    print(f"{'idx':<4} {'DC?':<4} {'saved_mean':>12} {'saved_std':>12} "
          f"{'subset_mean':>13} {'subset_std':>13} {'avg_obj_std':>13} "
          f"{'std_obj_mean':>13} {'normed_obj_std':>16}")
    print("=" * 140)

    dc_set = set(DC_ONLY_FEATURE_INDICES)
    for c in range(C):
        dc_flag = "*" if c in dc_set else " "
        print(
            f"{c:<4} {dc_flag:<4} "
            f"{saved_mean[c]:>+12.4f} {saved_std[c]:>12.4f} "
            f"{computed_mean[c]:>+13.4f} {computed_std[c]:>13.4f} "
            f"{avg_per_obj_std[c]:>13.4f} {std_of_per_obj_means[c]:>13.4f} "
            f"{avg_per_obj_normed_std[c]:>16.4f}"
        )

    print("\n--- Legend ---")
    print("saved_{mean,std}   : values in the .pt files currently used for training")
    print("subset_{mean,std}  : recomputed on the representative subset (global pooling)")
    print("avg_obj_std        : mean over objects of per-object spatial std")
    print("std_obj_mean       : std over objects of per-object spatial mean (inter-object variance)")
    print("normed_obj_std     : mean over objects of std((x - saved_mean) / saved_std)")
    print("                     Should be ≈1 if saved stats make the input unit-variance in spatial dim.")

    # DC channel summary
    dc_idx = list(DC_ONLY_FEATURE_INDICES)
    print("\n--- DC channel summary (14 channels fed to JiT with sh_degree0_only) ---")
    print(f"DC indices: {dc_idx}")
    print(f"normed_obj_std on DC channels:")
    for c in dc_idx:
        print(f"  ch {c:>2}: {avg_per_obj_normed_std[c]:.4f}")
    vals = avg_per_obj_normed_std[dc_idx]
    print(f"\nDC normed_obj_std  min={vals.min():.4f}  max={vals.max():.4f}  "
          f"ratio max/min={vals.max()/max(vals.min(), 1e-8):.2f}")

    # Proposed loss-weighting vector (inverse variance post-normalization).
    # If channel c has spatial std < 1, its MSE contribution is squashed; weight by
    # 1 / normed_obj_std[c]^2 to compensate.
    weights = 1.0 / np.maximum(avg_per_obj_normed_std[dc_idx] ** 2, 1e-8)
    # Normalize so average weight is 1 (doesn't change overall loss scale).
    weights = weights / weights.mean()
    print("\nProposed per-channel loss weight (DC channels, normalized to mean=1):")
    for c, w in zip(dc_idx, weights):
        print(f"  ch {c:>2}: weight = {w:.4f}")

    # Also emit as a Python list for easy copy-paste.
    print("\nCOPY-PASTE:")
    print("DC_CHANNEL_LOSS_WEIGHTS = [")
    for w in weights:
        print(f"    {w:.6f},")
    print("]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
