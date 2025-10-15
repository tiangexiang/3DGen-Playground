
# Data Loaders

> [!IMPORTANT]
> The loaders are demonstrated for single-GPU fetching. Please follow [PyTorch's distributed training guideline](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) or [third-party distributed training techniques](https://github.com/huggingface/accelerate) to make the loaders distributed.

We provide two types of data loaders for text-to-3D object generation tasks, each optimized for different use cases:

## **Standard Setting**
   
   This loader supports both 3DGS data and optional 2D renderings. See `dataloaders/standard_3dgen_loader.py` for implementation details. To test the loader, configure the paths in `dataloaders/test_dataloader.sh` and run the script to perform a quick sanity check.

   A minimal plug-and-play code snippet:

   ```python
   from dataloader.standard_3dgen_loader import create_dataloader

   dataset, dataloader = create_dataloader(
       obj_list,        # Paths to aesthetic_list.json and/or non_aesthetic_list.json
       gs_path,         # Root path to the directory containing downloaded GaussianVerse fittings
       caption_path,    # Path to the preprocessed captions.json file
       rendering_path,  # Root path to the directory containing downloaded 2D renderings (tar.gz files, optional), default: None
       num_images,      # Number of images to randomly sample per object (only when rendering_path is provided)
       mean_file,       # Path to the GS mean file for normalization
       std_file,        # Path to the GS std file for normalization
       batch_size,      # Number of samples per batch
       num_workers,     # Number of worker processes for data loading
       shuffle,         # Whether to shuffle the dataset, default: True
   )

   # Iterate through data
   for batch in dataloader:
       # batch is a dictionary containing:
       point_clouds = batch['point_cloud']  # (batch_size, num_points, feature_dim)
       captions = batch['caption']          # List of text strings
       
       # Optional fields (when rendering_path is provided):
       if 'images' in batch:
           images = batch['images']         # (batch_size, num_images, H, W, 3)
           cameras = batch['cameras']       # Dictionary with keys: K, R, t, c2w, fov_x, fov_y
       # ... do your stuff ... 
   ```

## **Fast Setting**
> [!WARNING]
> You will need preprocessed .tar files as instructed in the [WebDataset Preprocessing section](data/README.md#webdataset-preprocessing-optional).

   This loader uses WebDataset format for optimized loading of 3DGS data and captions without any image renderings, ideal for fast model verification but not for the best quality.


   A minimal plug-and-play code snippet:

   ```python
   from dataloader.fast_3dgen_loader import create_dataloader

   dataset, dataloader = create_dataloader(
       shard_pattern,        # URL to the preprocessed .tar files. E.g. "/THE/PATH/TO/YOUR/SHARDS/gaussianverse-*.tar"
       mean_file,       # Path to the GS mean file for normalization
       std_file,        # Path to the GS std file for normalization
       batch_size,      # Number of samples per batch
       num_workers,     # Number of worker processes for data loading
       shuffle,         # Whether to shuffle the dataset, default: True
   )

   # Iterate through data
   for batch in dataloader:
       # batch is a dictionary containing:
       point_clouds = batch['point_cloud']  # (batch_size, num_points, feature_dim)
       captions = batch['caption']          # List of text strings
       # ... do your stuff ... 
   ```
