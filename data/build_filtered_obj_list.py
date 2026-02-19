#!/usr/bin/env python3
"""
Build a filtered object list containing only objects whose 3DGS data exists on disk.

Use when you have partial GaussianVerse data (e.g., only some chunks extracted).
Output format matches aesthetic_list.json and can be passed to the standard dataloader.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm


def main(args):
    gs_path = Path(args.gs_path)
    if not gs_path.is_dir():
        print(f"Error: gs_path is not a directory: {gs_path}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input_json)
    if not input_path.is_file():
        print(f"Error: input_json not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        obj_list = json.load(f)
    if not isinstance(obj_list, dict):
        print("Error: input_json must be a JSON object (dict).", file=sys.stderr)
        sys.exit(1)

    print(f"Input list has {len(obj_list)} entries")

    filtered = {}
    missing_count = 0
    for obj_id, tar_gz_path in tqdm(obj_list.items(), desc="Checking 3DGS data"):
        path_part = tar_gz_path.replace('.tar.gz', '')
        data_dir = gs_path / path_part
        gs2sphere_path = data_dir / 'gs2sphere.npy'
        ply_path = data_dir / 'point_cloud.ply'
        if gs2sphere_path.exists() and ply_path.exists():
            filtered[obj_id] = tar_gz_path
        else:
            missing_count += 1

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, indent=2)

    print(f"Kept: {len(filtered)} objects")
    print(f"Excluded: {missing_count} objects")
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter object list to entries that exist on disk (partial GaussianVerse).')
    parser.add_argument('--gs_path', type=str, required=True,
                        help='Path to 3DGS data root (e.g., downloaded/aesthetic_chunk)')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to full object list (e.g., aesthetic_list.json)')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to save filtered object list')
    args = parser.parse_args()
    main(args)
