#!/usr/bin/env python3
"""
Create webdataset shards from GaussianVerse data.

This script reads one or more obj_list JSON files and creates webdataset .tar files
containing the corresponding data from the edgs directory structure.

Each webdataset entry includes:
- gs2sphere.npy: Gaussian sphere data  
- point_cloud.ply: Point cloud data (converted to numpy)
- caption.txt: Text caption for the 3D object

Usage:
    python make_webdataset.py --obj_list file1.json file2.json --captions /path/to/captions.json --output_dir /path/to/output --shard_size 1000
"""

import os
import numpy as np
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import webdataset as wds
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    features = np.concatenate((features_dc, features_extra), axis=-1).reshape(xyz.shape[0], -1)


    return np.concatenate((xyz, opacities, features, scales, rots), axis=1).astype(np.float32)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('webdataset_creation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_obj_list(obj_list_paths: List[str]) -> Dict[str, str]:
    """Load one or more obj_list JSON files and merge them.
    
    Args:
        obj_list_paths: List of paths to obj_list JSON files
        
    Returns:
        Dictionary mapping hash keys to tar.gz paths (merged from all files)
    """
    merged_data = {}
    
    for obj_list_path in obj_list_paths:
        logging.info(f"Loading obj list from {obj_list_path}")
        
        with open(obj_list_path, 'r') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} entries from {obj_list_path}")
        
        # Merge into the combined dictionary
        merged_data.update(data)
    
    logging.info(f"Total merged entries: {len(merged_data)}")
    return merged_data


def load_captions(captions_path: str) -> Dict[str, str]:
    """Load the captions JSON file.
    
    Args:
        captions_path: Path to captions.json
        
    Returns:
        Dictionary mapping foldername/filename to caption text
    """
    logging.info(f"Loading captions from {captions_path}")
    
    with open(captions_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} captions")
    return data


def extract_directory_info(tar_gz_path: str) -> Tuple[str, str]:
    """Extract directory and filename info from tar.gz path.
    
    Args:
        tar_gz_path: Path like "1923/9611649.tar.gz"
        
    Returns:
        Tuple of (directory_number, filename_without_extension)
    """
    # Split path like "1923/9611649.tar.gz" -> ("1923", "9611649")
    parts = tar_gz_path.split('/')
    if len(parts) != 2:
        raise ValueError(f"Unexpected tar.gz path format: {tar_gz_path}")
    
    directory_number = parts[0]
    filename = parts[1].replace('.tar.gz', '')
    
    return directory_number, filename


def find_data_files(gs_path: str, directory_number: str, filename: str) -> Dict[str, Optional[str]]:
    """Find the required data files for a given entry.
    
    Args:
        gs_path: Path to the 3DGS directory
        directory_number: Directory number (e.g., "1923")
        filename: Filename without extension (e.g., "9611649")
        
    Returns:
        Dictionary with paths to gs2sphere.npy, point_cloud.ply
    """
    data_dir = Path(gs_path) / directory_number / filename
    
    required_files = {
        'gs2sphere.npy': data_dir / 'gs2sphere.npy', 
        'point_cloud.ply': data_dir / 'point_cloud.ply'
    }
    
    # Check which files exist
    file_paths = {}
    for file_type, file_path in required_files.items():
        if file_path.exists():
            file_paths[file_type] = str(file_path)
        else:
            file_paths[file_type] = None
            logging.warning(f"Missing file: {file_path}")
    
    return file_paths




def create_webdataset_shards(
    obj_data: Dict[str, str],
    gs_path: str,
    captions: Dict[str, str],
    output_dir: str,
    shard_size: int = 1000,
    max_shards: Optional[int] = None
) -> None:
    """Create webdataset shards from the obj data.
    
    Args:
        obj_data: Dictionary from obj_list JSON files (merged)
        gs_path: Path to the 3DGS directory
        captions: Dictionary mapping foldername/filename to caption text
        output_dir: Output directory for webdataset shards
        shard_size: Number of samples per shard
        max_shards: Maximum number of shards to create (for testing)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to list for easier processing
    items = list(obj_data.items())
    
    if max_shards is not None:
        items = items[:max_shards * shard_size]
    
    total_items = len(items)
    num_shards = (total_items + shard_size - 1) // shard_size
    
    logging.info(f"Creating {num_shards} shards with up to {shard_size} samples each")
    logging.info(f"Total items to process: {total_items}")
    
    successful_entries = 0
    failed_entries = 0
    
    for shard_idx in tqdm(range(num_shards)):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, total_items)
        shard_items = items[start_idx:end_idx]
        
        shard_path = os.path.join(output_dir, f"gaussianverse-{shard_idx:06d}.tar")
        
        logging.info(f"Creating shard {shard_idx + 1}/{num_shards}: {shard_path}")
        
        with wds.TarWriter(shard_path) as writer:
            for hash_key, tar_gz_path in tqdm(shard_items, desc=f"Shard {shard_idx + 1}"):
                try:
                    # Extract directory info
                    directory_number, filename = extract_directory_info(tar_gz_path)
                    
                    # Find data files from 3DGS directory
                    file_paths = find_data_files(gs_path, directory_number, filename)
                    
                    # Check if required files exist
                    missing_files = [k for k, v in file_paths.items() if v is None]
                    if missing_files:
                        logging.warning(f"Skipping {hash_key}: missing {missing_files}")
                        failed_entries += 1
                        continue
                    
                    # Get caption for this entry
                    caption = captions.get(tar_gz_path.split('.tar.gz')[0], None)
                    if caption is None:
                        logging.warning(f"Skipping {hash_key}: no caption found for {tar_gz_path}")
                        failed_entries += 1
                        continue
                    
                    # Create sample dictionary
                    sample = {"__key__": hash_key}
                    
                    # Add files from 3DGS directory
                    print(file_paths.keys())
                    for file_type, file_path in file_paths.items():
                        if file_path is not None:
                            with open(file_path, 'rb') as f:
                                if file_type.endswith('.json'):
                                    # Store JSON as text
                                    sample[file_type] = f.read().decode('utf-8')
                                elif file_type.endswith('.ply'):
                                    # Convert PLY to numpy array
                                    sample[file_type[:-4]+'.npy'] = load_ply(file_path)
                                else:
                                    # Store binary files as bytes
                                    sample[file_type] = f.read()
                    
                    # Add caption
                    sample["caption.txt"] = caption
                    
                    # Add metadata
                    sample["metadata.json"] = json.dumps({
                        "hash_key": hash_key,
                        "tar_gz_path": tar_gz_path,
                        "directory_number": directory_number,
                        "filename": filename,
                        "has_caption": True
                    })
                    
                    writer.write(sample)
                    successful_entries += 1
                    
                except Exception as e:
                    logging.error(f"Error processing {hash_key} ({tar_gz_path}): {str(e)}")
                    failed_entries += 1
                    continue
      
    logging.info(f"Webdataset creation completed!")
    logging.info(f"Successfully processed: {successful_entries} entries")
    logging.info(f"Failed to process: {failed_entries} entries")
    logging.info(f"Created {num_shards} shard files in {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create webdataset shards from GaussianVerse data")
    parser.add_argument(
        "--obj_list", 
        type=str,
        nargs='+',
        required=True,
        help="One or more paths to obj_list JSON files (will be merged)"
    )
    parser.add_argument(
        "--gs_path",
        type=str,
        required=True,
        help="Path to the directory containing 3DGS fittings"
    )
    parser.add_argument(
        "--captions",
        type=str,
        required=True,
        help="Path to preprocessed captions.json file containing foldername/filename to caption mapping"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for webdataset shards"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of samples per shard (default: 1000)"
    )
    parser.add_argument(
        "--max_shards",
        type=int,
        default=None,
        help="Maximum number of shards to create (for testing only)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate input paths
    for obj_list_file in args.obj_list:
        if not os.path.exists(obj_list_file):
            logging.error(f"Object list file not found: {obj_list_file}")
            sys.exit(1)
        
    if not os.path.exists(args.gs_path):
        logging.error(f"3DGS path not found: {args.gs_path}")
        sys.exit(1)
        
    if not os.path.exists(args.captions):
        logging.error(f"Captions file not found: {args.captions}")
        sys.exit(1)
    
    # Load obj data
    obj_data = load_obj_list(args.obj_list)
    
    # Load captions
    captions_data = load_captions(args.captions)
    
    # Create webdataset shards
    create_webdataset_shards(
        obj_data=obj_data,
        gs_path=args.gs_path,
        captions=captions_data,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        max_shards=args.max_shards
    )


if __name__ == "__main__":
    main()
