"""
gs2atlas: Gaussian splat point cloud to atlas (2D plane) conversion.

This is not the original script used in the project; it is an implementation
snippet for reference only. It maps 3D Gaussian splat point clouds onto a
canonical 2D plane via sphere parameterization and optimal transport (OT)
matching, used for building atlas representations from 3D
Gaussian splatting outputs.
"""

import os
import sys
import argparse
import time
import concurrent.futures

import numpy as np
import torch
import ot
from lapjv import lapjv
from plyfile import PlyData

# ---------------------------------------------------------------------------
# TODO (user): Set these placeholders for your environment before running.
# ---------------------------------------------------------------------------
PATH_SPHERE_TO_PLANE_IDX = "sphere2plane.npy"  # TODO: download from https://downloads.cs.stanford.edu/vision/gaussianverse/sphere2plane.npy
PLY_ITERATION = 30000  # TODO: iteration folder name under point_cloud/ (e.g. point_cloud/iteration_30000/point_cloud.ply)
ATLAS_RESOLUTION = 128 * 128    # TODO: atlas grid size (e.g. 128*128)
SEGMENT_BATCH_SIZE = 250        # TODO: number of items to process sequentially in one program run
NUM_WORKERS_DEFAULT = 2         # TODO: default parallel workers in main
# ---------------------------------------------------------------------------

# Default grid: single segment (atlas resolution)
num_segments = 1 # FULL OT
segment_size = ATLAS_RESOLUTION

def process_single_wrapper(line, source_root, save_root, max_sh_degree, bound, visuzalize_mapping, sphere_points, sphere_to_plane):
    """Wrapper for process_single: builds full source path and delegates."""
    process_single(
        os.path.join(source_root, line), save_root, sphere_points, sphere_to_plane,
        max_sh_degree, bound, visuzalize_mapping
    )


def compute_lap(p1_segment, p2_segment, scaling_factor=1000, index=0):
    """Solve linear assignment (LAP) between two point sets; returns cost and permutation indices."""
    cost_matrix = ot.dist(p1_segment, p2_segment, metric='sqeuclidean')
    scaled_cost_matrix = np.rint(cost_matrix * scaling_factor).astype(int)
    x, y, cost = lapjv(scaled_cost_matrix)
    return cost, x, y, index


def load_ply(path, max_sh_degree=3):
    """
    Load a Gaussian splat PLY (xyz, color, opacity, scale, rotation).
    Returns a single tensor of shape (N, C) with columns: xyz(3), color(3), opacity(1), scale(3), rotation(…).
    """
    plydata = PlyData.read(path)
    vert = plydata.elements[0]

    xyz = np.stack((
        np.asarray(vert["x"]), np.asarray(vert["y"]), np.asarray(vert["z"])
    ), axis=1)
    opacities = np.asarray(vert["opacity"])[..., np.newaxis]

    def sorted_attrs(prefix):
        names = [p.name for p in vert.properties if p.name.startswith(prefix)]
        return sorted(names, key=lambda x: int(x.split("_")[-1]))

    color_names = sorted_attrs("color_")
    color = np.column_stack([np.asarray(vert[n]) for n in color_names])

    scale_names = sorted_attrs("scale_")
    scales = np.column_stack([np.asarray(vert[n]) for n in scale_names])

    rot_names = sorted_attrs("rot")
    rots = np.column_stack([np.asarray(vert[n]) for n in rot_names])

    xyz = torch.tensor(xyz, dtype=torch.float, device="cpu")
    features_dc = torch.tensor(color, dtype=torch.float, device="cpu").contiguous()
    opacity = torch.tensor(opacities, dtype=torch.float, device="cpu")
    scaling = torch.tensor(scales, dtype=torch.float, device="cpu")
    rotation = torch.tensor(rots, dtype=torch.float, device="cpu")
    print("xyz shape: {} \t features_dc shape: {} \t opacity shape: {} \t scaling shape: {} \t rotation shape: {}".format(
        xyz.shape, features_dc.shape, opacity.shape, scaling.shape, rotation.shape))
    return torch.cat((xyz, features_dc, opacity, scaling, rotation), dim=1)


def generate_fibonacci_sphere(num_points, radius):
    """Sample points on a sphere using Fibonacci (golden spiral) distribution; returns (N, 3)."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_points):
        y = -1.0 * (1.0 - (i / float(num_points - 1)) * 2.0)
        radius_at_y = np.sqrt(max(0, 1.0 - y * y))
        theta = phi * i
        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        points.append((x * radius, y * radius, z * radius))
    return np.array(points)


def lag_segment_matching(p1, p2):
    """Compute optimal assignment between two point sets via LAP; returns (corrs_2_to_1, corrs_1_to_2)."""
    num_segments = 1
    segment_size = p1.shape[0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_segments) as executor:
        futures = {executor.submit(compute_lap, p1, p2, scaling_factor=1000, index=0): 0}
        results = [None] * num_segments
        for future in concurrent.futures.as_completed(futures):
            cost, x, y, index = future.result()
            results[index] = (cost, x, y)
    corrs_2_to_1 = np.concatenate([y + i * segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    corrs_1_to_2 = np.concatenate([x + i * segment_size for i, (cost, x, y) in enumerate(results)], axis=0)
    return corrs_2_to_1, corrs_1_to_2


def process_single(source_root, save_root, sphere_points, sphere_to_plane, max_sh_degree=0, bound=0.95, visuzalize_mapping=False):
    """
    Convert one 3D Gaussian splat scene to atlas (2D plane) and save as .pt.

    Pipeline: load PLY -> filter by opacity -> pad/prune to 128*128 -> sort ->
    OT match to sphere -> reorder by sphere_to_plane -> save (128, 128, C) tensor.
    """
    all_start = time.time()

    source_name_segments = source_root.strip().split("/")
    source_name = source_name_segments[-2] + "-" + source_name_segments[-1]
    # TODO (user): adjust subpaths if your PLY layout differs
    source_file = os.path.join(source_root, f"point_cloud/iteration_{PLY_ITERATION}/point_cloud.ply")

    if os.path.exists(os.path.join(save_root, source_name + ".pt")):
        print(source_name + ".pt", "SKIP!!!!")
        return

    if not os.path.exists(source_file):
        print("File {} does not exist. Skip.".format(source_file))
        return
    gs = load_ply(source_file, max_sh_degree)

    gs = gs.cpu().numpy()

    # Keep only visible Gaussians (opacity > 0)
    opacity_mask = gs[:, 6] > 0.0
    gs = gs[opacity_mask]
    total_vis_num = int(np.sum(opacity_mask))
    print("visble num:", total_vis_num, "total_num:", ATLAS_RESOLUTION, total_vis_num - ATLAS_RESOLUTION)

    # Pad or prune to exactly ATLAS_RESOLUTION
    target_num = ATLAS_RESOLUTION
    if total_vis_num < target_num:
        print("START PADDING!")
        residual = target_num - total_vis_num
        scales = np.mean(gs[:, 7:10], axis=1) # need to double check channel indices
        min_scale = np.argsort(scales)[:residual]
        pad = np.copy(gs[min_scale])
        pad[:, 6] *= 0.0
        gs = np.concatenate((gs, pad), axis=0)
    elif total_vis_num > target_num:
        print("START PRUNNING!")
        residual = total_vis_num - target_num
        scales = np.mean(gs[:, 7:10], axis=1) # need to double check channel indices
        min_scale = np.argsort(scales)[residual:]
        gs = gs[min_scale]

    points_3d = gs[:, :3]
    sorted_indices = np.lexsort((points_3d[:, 2], points_3d[:, 1], points_3d[:, 0]))
    points_3d = points_3d[sorted_indices]
    gs = gs[sorted_indices]

    # Optimal transport: match scene points to canonical sphere
    print("START OT")
    corrs_2_to_1, corrs_1_to_2 = lag_segment_matching(points_3d, sphere_points)
    print("DONE OT")

    gs = gs[corrs_2_to_1]
    offset = gs[:, :3] - sphere_points
    gs = np.concatenate((gs, offset), axis=-1)
    gs = gs[sphere_to_plane]
    grid_side = int(np.sqrt(ATLAS_RESOLUTION))  # e.g. 128
    gs = np.reshape(gs, (grid_side, grid_side, -1))
    gs = torch.from_numpy(gs)

    if visuzalize_mapping:
        # TODO: requires std_volume and xyz in scope; currently disabled
        pass

    print("plane shape", gs.shape)
    torch.save(gs, os.path.join(save_root, source_name + ".pt"))

    print("pre-processing time:", time.time() - all_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 3D Gaussian splat PLYs to atlas (.pt) via sphere + OT.")
    parser.add_argument("--source_root", type=str, help="Root dir containing scene folders")
    parser.add_argument("--save_root", type=str, help="Output dir for .pt atlases")
    parser.add_argument("--max_sh_degree", type=int, default=0)
    parser.add_argument("--txt_file", type=str, help="List of scene paths (one per line)")
    parser.add_argument("--obj", type=str, help="Single scene path (overrides txt_file)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1)
    parser.add_argument("--bound", type=float, default=0.95, help="Spatial bound [-bound, bound]")
    parser.add_argument("--visuzalize_mapping", action="store_true", help="Save mapping visualization (currently no-op)")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--index", type=int, default=0, help="Segment index: process lines [index*segment : (index+1)*segment]")
    args = parser.parse_args()

    # TODO (user): override num_workers / batch size via placeholders or CLI
    num_workers = NUM_WORKERS_DEFAULT
    num_points = ATLAS_RESOLUTION
    radius = 1.0
    sphere_points = generate_fibonacci_sphere(num_points, radius)
    sphere_points_sorted_indices = np.lexsort((sphere_points[:, 2], sphere_points[:, 1], sphere_points[:, 0]))
    sphere_points = sphere_points[sphere_points_sorted_indices]
    sphere_to_plane = np.load(PATH_SPHERE_TO_PLANE_IDX)  # TODO (user): ensure this file exists for your resolution

    segment = SEGMENT_BATCH_SIZE
    # text file contains the list of scene paths, so we can chunk the list into segments and process in parallel
    with open(args.txt_file, "r") as f:
        lines = f.read().splitlines()
    lines = lines[args.index * segment : (args.index + 1) * segment]
    print(len(lines))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_single_wrapper,
                line, args.source_root, args.save_root, args.max_sh_degree, args.bound,
                args.visuzalize_mapping, sphere_points, sphere_to_plane,
            )
            for line in lines
        ]
        for future in futures:
            future.result()