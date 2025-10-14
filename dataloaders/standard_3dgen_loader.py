#!/usr/bin/env python3
"""
Standard 3D Generation Data Loader for GaussianVerse.

This module provides a PyTorch Dataset and DataLoader for loading 3D Gaussian Splatting data
along with optional 2D renderings and text captions for 3D generation tasks.
"""

import os
import json
import logging
import tarfile
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from plyfile import PlyData
from PIL import Image


def fov2focal(fov: float, pixels: int) -> float:
    """Convert field of view to focal length.
    
    Args:
        fov: Field of view in radians
        pixels: Image dimension in pixels
        
    Returns:
        Focal length in pixels
    """
    return pixels / (2 * math.tan(fov / 2))


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


def load_captions(caption_path: str) -> Dict[str, str]:
    """Load the captions JSON file.
    
    Args:
        caption_path: Path to captions.json
        
    Returns:
        Dictionary mapping foldername/filename to caption text
    """
    logging.info(f"Loading captions from {caption_path}")
    
    with open(caption_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} captions")
    return data


def load_ply(path: str) -> np.ndarray:
    """Load PLY file and convert to numpy array format.
    
    Args:
        path: Path to PLY file
        
    Returns:
        Numpy array of shape (N, D) containing [xyz, opacity, features, scales, rotations]
    """
    max_sh_degree = 3
    plydata = PlyData.read(path)
    
    # Load positions
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # Load DC features
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # Load extra features
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    # Load scales
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Load rotations
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Concatenate features
    features = np.concatenate((features_dc, features_extra), axis=-1).reshape(xyz.shape[0], -1)

    return np.concatenate((xyz, opacities, features, scales, rots), axis=1).astype(np.float32)


def extract_directory_info(tar_gz_path: str) -> Tuple[str, str]:
    """Extract directory and filename info from tar.gz path.
    
    Args:
        tar_gz_path: Path like "1923/9611649.tar.gz"
        
    Returns:
        Tuple of (directory_number, filename_without_extension)
    """
    parts = tar_gz_path.split('/')
    if len(parts) != 2:
        raise ValueError(f"Unexpected tar.gz path format: {tar_gz_path}")
    
    directory_number = parts[0]
    filename = parts[1].replace('.tar.gz', '')
    
    return directory_number, filename


class Standard3DGenDataset(Dataset):
    """PyTorch Dataset for 3D object generation.
    
    This dataset loads 3DGS data, captions, and optionally 2D renderings for
    training 3D generative models.
    """
    
    def __init__(
        self,
        obj_list: List[str],
        gs_path: str,
        caption_path: str,
        rendering_path: Optional[str] = None,
        num_images: int = 1,
        mean_file: Optional[str] = None,
        std_file: Optional[str] = None,
    ):
        """Initialize the dataset.
        
        Args:
            obj_list: List of paths to obj_list JSON files
            gs_path: Root path for downloaded 3DGS data
            caption_path: Path to preprocessed caption file
            rendering_path: Root path for downloaded 2D renderings (optional)
            num_images: Number of images to fetch at each step (only when rendering_path is not None)
            mean_file: Path to downloaded GS mean file
            std_file: Path to downloaded GS std file
        """
        super().__init__()
        
        self.gs_path = Path(gs_path)
        self.rendering_path = Path(rendering_path) if rendering_path is not None else None
        self.num_images = num_images
        
        # Load object list
        self.obj_data = load_obj_list(obj_list)
        self.keys = list(self.obj_data.keys())
        
        # Load captions
        self.captions = load_captions(caption_path)
        
        # Load normalization statistics if provided
        self.mean = None
        self.std = None
        if mean_file is not None and std_file is not None:
            logging.info(f"Loading normalization statistics from {mean_file} and {std_file}")
            self.mean = torch.load(mean_file).cpu().numpy().astype(np.float32)
            self.std = torch.load(std_file).cpu().numpy().astype(np.float32)
        else:
            logging.warning("Normalization is NOT enabled, mean or std file not provided.")
        
        logging.info(f"Initialized dataset with {len(self.keys)} samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.keys)
    
    def _load_3dgs_data(self, directory_number: str, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load 3DGS data (point cloud and gs2sphere).
        
        Args:
            directory_number: Directory number (e.g., "1923")
            filename: Filename without extension (e.g., "9611649")
            
        Returns:
            Tuple of (point_cloud, gs2sphere) as numpy arrays
        """
        data_dir = self.gs_path / directory_number / filename
        
        # Load gs2sphere indices
        gs2sphere_path = data_dir / 'gs2sphere.npy'
        gs2sphere = np.load(str(gs2sphere_path))
        
        # Load point cloud
        ply_path = data_dir / 'point_cloud.ply'
        point_cloud = load_ply(str(ply_path))
        
        # Apply gs2sphere indexing
        point_cloud = point_cloud[gs2sphere]
        
        return point_cloud, gs2sphere
    
    def _load_renderings(self, directory_number: str, filename: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """Load 2D renderings and camera parameters from tar.gz file.
        
        Args:
            directory_number: Directory number (e.g., "1923")
            filename: Filename without extension (e.g., "9611649")
            
        Returns:
            Tuple of (images, cameras):
                - images: Numpy array of shape (num_images, H, W, C) or None
                - cameras: Dictionary containing camera parameters (K, R, t) or None
        """
        if self.rendering_path is None:
            return None, None
        
        tar_path = self.rendering_path / directory_number / f"{filename}.tar.gz"
        
        if not tar_path.exists():
            logging.warning(f"Rendering tar file not found: {tar_path}")
            return None, None
        
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                # Find all available frame indices
                frame_indices = []
                for member in tar.getmembers():
                    if '.png' in member.name and 'campos_512_v1' in member.name:
                        # Extract frame index from path like 'campos_512_v1/00000/00000.png'
                        parts = member.name.split('/')
                        if len(parts) >= 3:
                            try:
                                frame_idx = int(parts[-2])
                                if frame_idx not in frame_indices:
                                    frame_indices.append(frame_idx)
                            except ValueError:
                                continue
                
                if len(frame_indices) == 0:
                    logging.warning(f"No valid frames found in {tar_path}")
                    return None, None
                
                frame_indices.sort()
                
                # Randomly select num_images frames
                if len(frame_indices) < self.num_images:
                    selected_indices = frame_indices
                else:
                    selected_indices = sorted(np.random.choice(frame_indices, self.num_images, replace=False).tolist())
                
                # Load selected frames and camera parameters
                images = []
                cameras = {
                    'K': [],      # Intrinsic matrices
                    'R': [],      # Rotation matrices
                    't': [],      # Translation vectors
                    'c2w': [],    # Camera-to-world matrices
                    'fov_x': [],  # Field of view x
                    'fov_y': [],  # Field of view y
                }
                
                for frame_idx in selected_indices:
                    # Load PNG image
                    png_member_name = f'campos_512_v1/{frame_idx:05d}/{frame_idx:05d}.png'
                    try:
                        png_member = tar.getmember(png_member_name)
                        png_data = tar.extractfile(png_member).read()
                        image = Image.open(BytesIO(png_data))
                        image_array = np.array(image)[:, :, :4]  # RGBA
                        # Convert to RGB and normalize to [0, 1]
                        image_rgb = image_array[:, :, :3].astype(np.float32) / 255.0
                        images.append(image_rgb)
                    except Exception as e:
                        logging.warning(f"Failed to load image for frame {frame_idx}: {e}")
                        continue
                    
                    # Load JSON camera parameters
                    json_member_name = f'campos_512_v1/{frame_idx:05d}/{frame_idx:05d}.json'
                    try:
                        json_member = tar.getmember(json_member_name)
                        json_data = tar.extractfile(json_member).read().decode('utf-8')
                        meta = json.loads(json_data)
                        
                        # Build camera-to-world matrix
                        c2w = np.eye(4, dtype=np.float32)
                        c2w[:3, 0] = np.array(meta['x'], dtype=np.float32)
                        c2w[:3, 1] = np.array(meta['y'], dtype=np.float32)
                        c2w[:3, 2] = np.array(meta['z'], dtype=np.float32)
                        c2w[:3, 3] = np.array(meta['origin'], dtype=np.float32)
                        
                        # Get field of view
                        fov_x = meta['x_fov']
                        fov_y = meta['y_fov']
                        
                        # Compute focal length and intrinsic matrix
                        focal = fov2focal(fov_x, 512)
                        K = np.eye(3, dtype=np.float32)
                        K[0, 0] = K[1, 1] = focal
                        K[0, 2] = K[1, 2] = 256.0
                        
                        # Get world-to-camera transform
                        w2c = np.linalg.inv(c2w)
                        R = w2c[:3, :3].astype(np.float32)
                        t = w2c[:3, 3].astype(np.float32)
                        
                        cameras['K'].append(K)
                        cameras['R'].append(R)
                        cameras['t'].append(t)
                        cameras['c2w'].append(c2w)
                        cameras['fov_x'].append(fov_x)
                        cameras['fov_y'].append(fov_y)
                        
                    except Exception as e:
                        logging.warning(f"Failed to load camera for frame {frame_idx}: {e}")
                        continue
                
                if len(images) == 0:
                    logging.warning(f"No images successfully loaded from {tar_path}")
                    return None, None
                
                # Convert lists to numpy arrays
                images_array = np.stack(images, axis=0)
                cameras_array = {
                    'K': np.stack(cameras['K'], axis=0),
                    'R': np.stack(cameras['R'], axis=0),
                    't': np.stack(cameras['t'], axis=0),
                    'c2w': np.stack(cameras['c2w'], axis=0),
                    'fov_x': np.array(cameras['fov_x'], dtype=np.float32),
                    'fov_y': np.array(cameras['fov_y'], dtype=np.float32),
                }
                
                return images_array, cameras_array
                
        except Exception as e:
            logging.error(f"Failed to load renderings from {tar_path}: {e}")
            return None, None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - point_cloud: 3DGS point cloud data (N, D)
                - caption: Text caption
                - hash_key: Unique identifier
                - images: 2D renderings (optional, if rendering_path is not None)
                - cameras: Camera parameters (optional, if rendering_path is not None)
                    - K: Intrinsic matrices (num_images, 3, 3)
                    - R: Rotation matrices (num_images, 3, 3)
                    - t: Translation vectors (num_images, 3)
                    - c2w: Camera-to-world matrices (num_images, 4, 4)
                    - fov_x: Field of view x (num_images,)
                    - fov_y: Field of view y (num_images,)
        """
        hash_key = self.keys[idx]
        tar_gz_path = self.obj_data[hash_key]
        
        # Extract directory info
        directory_number, filename = extract_directory_info(tar_gz_path)
        
        # Load 3DGS data
        point_cloud, gs2sphere = self._load_3dgs_data(directory_number, filename)
        
        # Normalize if enabled
        if self.mean is not None and self.std is not None:
            point_cloud = (point_cloud - self.mean[None]) / (self.std[None] + 1e-8)
        
        # Get caption (key is directory/filename without .tar.gz)
        caption_key = tar_gz_path.split('.tar.gz')[0]
        caption = self.captions.get(caption_key, "")
        
        # Prepare output
        sample = {
            'point_cloud': torch.from_numpy(point_cloud),
            'caption': caption,
            'hash_key': hash_key,
            'tar_gz_path': tar_gz_path,
        }
        
        # Load renderings and cameras if path is provided
        if self.rendering_path is not None:
            images, cameras = self._load_renderings(directory_number, filename)
            if images is not None:
                sample['images'] = torch.from_numpy(images)
            if cameras is not None:
                # Convert camera parameters to torch tensors
                sample['cameras'] = {
                    'K': torch.from_numpy(cameras['K']),
                    'R': torch.from_numpy(cameras['R']),
                    't': torch.from_numpy(cameras['t']),
                    'c2w': torch.from_numpy(cameras['c2w']),
                    'fov_x': torch.from_numpy(cameras['fov_x']),
                    'fov_y': torch.from_numpy(cameras['fov_y']),
                }
        
        return sample


def create_dataloader(
    obj_list: List[str],
    gs_path: str,
    caption_path: str,
    rendering_path: Optional[str] = None,
    num_images: int = 1,
    mean_file: Optional[str] = None,
    std_file: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """Create a PyTorch DataLoader for the Standard3DGen dataset.
    
    Args:
        obj_list: List of paths to obj_list JSON files
        gs_path: Root path for downloaded 3DGS data
        caption_path: Path to preprocessed caption file
        rendering_path: Root path for downloaded 2D renderings (optional)
        num_images: Number of images to fetch at each step
        mean_file: Path to downloaded GS mean file (normalization will be applied if provided)
        std_file: Path to downloaded GS std file (normalization will be applied if provided)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments to pass to DataLoader
        
    Returns:
        PyTorch DataLoader instance
    """
    dataset = Standard3DGenDataset(
        obj_list=obj_list,
        gs_path=gs_path,
        caption_path=caption_path,
        rendering_path=rendering_path,
        num_images=num_images,
        mean_file=mean_file,
        std_file=std_file,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    
    return dataset, dataloader


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Standard3DGen DataLoader")
    parser.add_argument("--obj_list", type=str, nargs='+', required=True,
                       help="Paths to obj_list JSON files")
    parser.add_argument("--gs_path", type=str, required=True,
                       help="Path to 3DGS data directory")
    parser.add_argument("--caption_path", type=str, required=True,
                       help="Path to captions JSON file")
    parser.add_argument("--rendering_path", type=str, default=None,
                       help="Path to renderings directory")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images per sample")
    parser.add_argument("--mean_file", type=str, default=None,
                       help="Path to mean file")
    parser.add_argument("--std_file", type=str, default=None,
                       help="Path to std file")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of workers")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create dataloader
    dataset,dataloader = create_dataloader(
        obj_list=args.obj_list,
        gs_path=args.gs_path,
        caption_path=args.caption_path,
        rendering_path=args.rendering_path,
        num_images=args.num_images,
        mean_file=args.mean_file,
        std_file=args.std_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Test loading a few batches
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Only test first 3 batches
            break
        
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Keys: {batch.keys()}")
        print(f"Point cloud shape: {batch['point_cloud'].shape}")
        print(f"Captions: {batch['caption']}")
        
        if 'images' in batch:
            print(f"Images shape: {batch['images'].shape}")
        
        if 'cameras' in batch:
            print(f"Cameras available:")
            print(f"  K shape: {batch['cameras']['K'].shape}")
            print(f"  R shape: {batch['cameras']['R'].shape}")
            print(f"  t shape: {batch['cameras']['t'].shape}")
            print(f"  c2w shape: {batch['cameras']['c2w'].shape}")
    
    print("\nDataLoader test completed successfully!")

