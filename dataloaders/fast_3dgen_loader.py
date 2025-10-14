#!/usr/bin/env python3
"""
Fast 3D Generation Data Loader for GaussianVerse using WebDataset.

This module provides a high-performance data loading pipeline for 3D Gaussian Splatting
data using the webdataset format, without requiring 2D renderings. The webdataset is
returned directly without PyTorch DataLoader wrapping for maximum efficiency.
This is optimized for fast training of 3D generation models.
"""

import os
import json
import logging
import io
import glob
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import torch

try:
    import webdataset as wds
    WDS_AVAILABLE = True
except ImportError:
    WDS_AVAILABLE = False
    logging.warning("WebDataset not available. Install with: pip install webdataset")


def expand_shard_pattern(shard_pattern: str) -> Union[str, List[str]]:
    """Expand shard pattern to handle glob wildcards.
    
    WebDataset doesn't automatically expand shell glob patterns like '*.tar'.
    This function expands them into a list of actual files, or returns the
    pattern as-is if it uses brace notation or is a URL.
    
    Args:
        shard_pattern: Pattern for shard files (e.g., "shards/*.tar" or "shards/{000..099}.tar")
        
    Returns:
        Either the original pattern (if brace notation/URL) or expanded list of file paths
    """
    # If pattern contains braces, it's already in webdataset format
    if '{' in shard_pattern and '}' in shard_pattern:
        return shard_pattern
    
    # If it's a URL, return as-is
    if shard_pattern.startswith(('http://', 'https://', 's3://', 'gs://')):
        return shard_pattern
    
    # If pattern contains glob wildcards, expand them
    if '*' in shard_pattern or '?' in shard_pattern:
        expanded = sorted(glob.glob(shard_pattern))
        if not expanded:
            raise FileNotFoundError(
                f"No files found matching pattern: {shard_pattern}\n"
                f"Please check if the path is correct and files exist."
            )
        logging.info(f"Expanded glob pattern '{shard_pattern}' to {len(expanded)} files")
        return expanded
    
    # Return single file/pattern as-is
    return shard_pattern


def decode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a webdataset sample into usable format.
    
    Args:
        sample: Raw webdataset sample with encoded data
        
    Returns:
        Decoded sample with processed data
    """
    decoded = {"__key__": sample["__key__"]}
    
    # Decode gs2sphere.npy - stored as binary bytes in webdataset
    if "gs2sphere.npy" in sample:
        try:
            # The gs2sphere.npy is stored as raw bytes, load as numpy array
            if isinstance(sample["gs2sphere.npy"], bytes):
                gs_bytes = sample["gs2sphere.npy"]
                decoded["gs2sphere"] = np.load(io.BytesIO(gs_bytes))
            else:
                # If it's already processed by webdataset decode
                decoded["gs2sphere"] = sample["gs2sphere.npy"]
        except Exception as e:
            logging.warning(f"Failed to decode gs2sphere.npy for {sample['__key__']}: {e}")
            decoded["gs2sphere"] = None
    
    # Handle point_cloud.npy - stored as binary bytes in webdataset
    if "point_cloud.npy" in sample:
        try:
            # The point_cloud.npy is stored as raw bytes, load as numpy array
            if isinstance(sample["point_cloud.npy"], bytes):
                pc_bytes = sample["point_cloud.npy"]
                decoded["point_cloud"] = np.load(io.BytesIO(pc_bytes))
            else:
                # If it's already processed by webdataset decode
                decoded["point_cloud"] = sample["point_cloud.npy"]
        except Exception as e:
            logging.warning(f"Failed to decode point_cloud.npy for {sample['__key__']}: {e}")
            decoded["point_cloud"] = None
    
    # Apply gs2sphere indexing to point cloud
    if decoded.get("point_cloud") is not None and decoded.get("gs2sphere") is not None:
        try:
            decoded["point_cloud"] = decoded["point_cloud"][decoded["gs2sphere"]]
        except Exception as e:
            logging.warning(f"Failed to apply gs2sphere indexing for {sample['__key__']}: {e}")
    
    # Decode metadata.json - stored as UTF-8 text string in webdataset
    if "metadata.json" in sample:
        try:
            # The metadata.json is stored as a UTF-8 string, so we need to parse it
            if isinstance(sample["metadata.json"], str):
                decoded["metadata"] = json.loads(sample["metadata.json"])
            elif isinstance(sample["metadata.json"], bytes):
                decoded["metadata"] = json.loads(sample["metadata.json"].decode('utf-8'))
            else:
                decoded["metadata"] = sample["metadata.json"]
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logging.warning(f"Failed to decode metadata.json for {sample['__key__']}: {e}")
            decoded["metadata"] = {}
    
    return decoded


def process_sample(sample: Dict[str, Any], mean: Optional[np.ndarray] = None, 
                   std: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Process a decoded sample into the final output format.
    
    Args:
        sample: Decoded webdataset sample
        mean: Mean for normalization (optional)
        std: Standard deviation for normalization (optional)
        
    Returns:
        Processed sample ready for training
    """
    # Extract point cloud
    point_cloud = sample.get("point_cloud")
    
    if point_cloud is None:
        logging.error(f"Point cloud is None for sample {sample['__key__']}")
        # Return a dummy sample to avoid breaking the pipeline
        return {
            'point_cloud': torch.zeros((1, 59), dtype=torch.float32),
            'caption': "",
            'hash_key': sample['__key__'],
            'valid': False
        }
    
    # Convert to float32 if needed
    if point_cloud.dtype != np.float32:
        point_cloud = point_cloud.astype(np.float32)
    
    # Apply normalization if provided
    if mean is not None and std is not None:
        point_cloud = (point_cloud - mean[None]) / (std[None] + 1e-8)
    
    # Extract caption from metadata
    metadata = sample.get("metadata", {})
    caption = metadata.get("caption", "")
    
    # Prepare output
    output = {
        'point_cloud': torch.from_numpy(point_cloud),
        'caption': caption,
        'hash_key': sample['__key__'],
        'valid': True
    }
    
    # Add metadata fields if available
    if "tar_gz_path" in metadata:
        output['tar_gz_path'] = metadata['tar_gz_path']
    
    return output


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of samples into a single dictionary.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data dictionary
    """
    # Filter out invalid samples
    valid_batch = [sample for sample in batch if sample.get('valid', True)]
    
    if len(valid_batch) == 0:
        # Return a dummy batch if all samples are invalid
        logging.warning("All samples in batch are invalid")
        return {
            'point_cloud': torch.zeros((1, 1, 59), dtype=torch.float32),
            'caption': [""],
            'hash_key': ["dummy"],
            'valid': torch.tensor([False])
        }
    
    # Stack point clouds (assuming they all have the same number of points after gs2sphere)
    point_clouds = [sample['point_cloud'] for sample in valid_batch]
    
    # Check if all point clouds have the same shape
    shapes = [pc.shape for pc in point_clouds]
    if len(set(shapes)) > 1:
        logging.warning(f"Point clouds have different shapes: {shapes}")
        # Pad or handle differently if needed
        # For now, we'll just take samples with matching shapes
        most_common_shape = max(set(shapes), key=shapes.count)
        valid_batch = [sample for sample in valid_batch 
                      if sample['point_cloud'].shape == most_common_shape]
        point_clouds = [sample['point_cloud'] for sample in valid_batch]
    
    # Stack point clouds
    batched_point_clouds = torch.stack(point_clouds, dim=0)
    
    # Collect other fields
    captions = [sample['caption'] for sample in valid_batch]
    hash_keys = [sample['hash_key'] for sample in valid_batch]
    
    output = {
        'point_cloud': batched_point_clouds,
        'caption': captions,
        'hash_key': hash_keys,
        'valid': torch.tensor([sample.get('valid', True) for sample in valid_batch])
    }
    
    # Add tar_gz_path if available
    if 'tar_gz_path' in valid_batch[0]:
        output['tar_gz_path'] = [sample['tar_gz_path'] for sample in valid_batch]
    
    return output


def create_webdataset(
    shard_pattern: str,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    batch_size: int = 1,
    shuffle_buffer: int = 1000,
    num_workers: int = 0,
    repeat: bool = False,
    **kwargs
):
    """Create a webdataset pipeline with batching.
    
    Args:
        shard_pattern: Pattern for webdataset shard files (e.g., "shards/gaussianverse-*.tar")
                      Supports glob patterns (*.tar), brace notation ({000..099}.tar), or file lists
        mean: Mean for normalization (optional)
        std: Standard deviation for normalization (optional)
        batch_size: Batch size for processing
        shuffle_buffer: Buffer size for shuffling (set to 0 to disable)
        repeat: If True, the dataset will repeat indefinitely (for training)
                If False, the dataset will iterate once and stop (for validation/testing)
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        Configured WebDataset with batching
    """
    if not WDS_AVAILABLE:
        raise ImportError("WebDataset is not available. Install with: pip install webdataset")
    
    # Expand glob patterns if necessary
    expanded_pattern = expand_shard_pattern(shard_pattern)
    
    dataset = wds.WebDataset(expanded_pattern, resampled=True, shardshuffle=True)
    
    # Apply repeat BEFORE shuffle for proper behavior
    if repeat:
        dataset = dataset.repeat()
    
    # Apply shuffling if requested
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    
    # Decode and process samples
    dataset = (
        dataset
        .decode()
        .map(decode_sample)
        .map(lambda sample: process_sample(sample, mean=mean, std=std))
    )
    
    # Apply batching using webdataset's batched method
    if batch_size > 1:
        dataset = dataset.batched(batch_size)

    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    loader = loader.unbatched().shuffle(shuffle_buffer).batched(batch_size)
    
    return dataset, loader


def create_dataloader(
    shard_pattern: str,
    mean_file: Optional[str] = None,
    std_file: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
    repeat: bool = False,
    **kwargs
):
    """Create a WebDataset for the Fast3DGen dataset.
    
    This function creates a webdataset pipeline that can be directly iterated over.
    It does NOT wrap the webdataset in a PyTorch DataLoader for maximum efficiency.
    
    Args:
        shard_pattern: Pattern for webdataset shard files. Supports:
                      - Glob patterns: "shards/gaussianverse-*.tar"
                      - Brace notation: "shards/gaussianverse-{000000..000099}.tar"
                      - File lists: ["shard1.tar", "shard2.tar"]
        mean_file: Path to GS mean file for normalization (optional)
        std_file: Path to GS std file for normalization (optional)
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading (Note: WebDataset handles this internally)
        shuffle: Whether to shuffle the data
        shuffle_buffer: Buffer size for shuffling (only used if shuffle=True)
        repeat: If True, dataset repeats indefinitely (for training)
                If False, iterates once then stops (for validation/testing)
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        WebDataset instance that can be directly iterated
            
    Usage:
        For training (infinite loop):
        >>> dataset = create_dataloader(..., repeat=True)
        >>> for batch in dataset:
        >>>     # process batch (will never end, use manual stopping)
        
        For validation (single epoch):
        >>> dataset = create_dataloader(..., repeat=False)
        >>> for batch in dataset:
        >>>     # process batch (stops after one full pass)
        
        To manually restart after one pass (if repeat=False):
        >>> dataset = create_dataloader(..., repeat=False)
        >>> # First pass
        >>> for batch in dataset:
        >>>     pass
        >>> # Need to recreate for second pass
        >>> dataset = create_dataloader(..., repeat=False)
    """
    # Load normalization statistics if provided
    mean = None
    std = None
    if mean_file is not None and std_file is not None:
        logging.info(f"Loading normalization statistics from {mean_file} and {std_file}")
        mean = torch.load(mean_file).cpu().numpy().astype(np.float32)
        std = torch.load(std_file).cpu().numpy().astype(np.float32)
        logging.info(f"Normalization enabled: mean shape {mean.shape}, std shape {std.shape}")
    else:
        logging.warning("Normalization is NOT enabled, mean or std file not provided.")
    
    # Create webdataset with batching
    dataset, loader = create_webdataset(
        shard_pattern=shard_pattern,
        mean=mean,
        std=std,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer if shuffle else 0,
        num_workers=num_workers,
        repeat=repeat,
    )
    
    # Return the webdataset directly (no PyTorch DataLoader wrapper)
    return dataset, loader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Fast3DGen DataLoader with WebDataset")
    parser.add_argument("--shard_pattern", type=str, required=True,
                       help="Pattern for webdataset shard files (e.g., 'shards/gaussianverse-*.tar')")
    parser.add_argument("--mean_file", type=str, default=None,
                       help="Path to mean file for normalization")
    parser.add_argument("--std_file", type=str, default=None,
                       help="Path to std file for normalization")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers")
    parser.add_argument("--shuffle", action="store_true",
                       help="Shuffle the dataset")
    parser.add_argument("--shuffle_buffer", type=int, default=1000,
                       help="Shuffle buffer size")
    parser.add_argument("--repeat", action="store_true",
                       help="Repeat dataset infinitely (for training)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create dataloader (returns webdataset directly)
    logging.info("Creating Fast3DGen WebDataset...")
    dataset, loader = create_dataloader(
        shard_pattern=args.shard_pattern,
        mean_file=args.mean_file,
        std_file=args.std_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        shuffle_buffer=args.shuffle_buffer,
        repeat=args.repeat
    )
    
    # Test loading a few batches
    print(f"\nTesting Fast3DGen WebDataset (Direct Iteration)")
    print("=" * 60)
    print(f"Glob pattern expansion enabled for *.tar patterns")
    print(f"Repeat mode: {'ON (infinite)' if args.repeat else 'OFF (single pass)'}")
    
    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 30:  # Only test first 30 batches
                break
            
            print(f"\n--- Batch {batch_idx + 1} ---")
            print(f"Keys: {batch.keys()}")
            print(f"Point cloud shape: {batch['point_cloud'].shape}")
            print(f"Point cloud dtype: {batch['point_cloud'].dtype}")
            print(f"Number of captions: {len(batch['caption'])}")
            print(f"First caption: {batch['caption'][0][:100]}...")  # Show first 100 chars
            print(f"Hash keys: {batch['hash_key']}")
            
            # Print statistics
            pc = batch['point_cloud']
            print(f"Point cloud stats:", pc.shape)
        
        print("\n" + "=" * 60)
        print("DataLoader test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

