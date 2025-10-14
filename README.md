<div align="center">

# The 3D Gen Playground

![GaussianVerse Teaser](assets/gaussianverse_teaser.gif)

</div>

The 3D Gen Playground is a user-friendly codebase designed to accelerate 3D generation research and development. We provide an **open data platform** with standardized protocols and curated community datasets, enabling reproducible and fair model comparisons. Built on efficient data loaders, visualizers, template model baselines, and utility functions, our **plug-and-play components** seamlessly integrates into your existing workflows, allowing you to focus on innovation rather than infrastructure. 

## Development Plan

- [x] Two types of dataloaders
- [ ] Spark visualizer @Ryan-Rong-24
- [ ] Baseline tokenizers and generative models
- [ ] Evaluation pipeline

## Installation

0. Create a virtual environment:
```bash
conda create -n 3dgen python=3.10
```

1. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) based on your system configurations. The code base was tested on torch==2.4.0.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Before proceeding, configure your environment variables & download paths at `.env`:

All shell scripts will automatically read from `.env`.

## Dataset Downloading & Preprocessing

The core feature of this project is its **Open Data** approach. We provide curated access to community datasets with standardized formats for consistent training across different models.

Please see the **[data](data/)** folder for detailed downloading instructions.

## Data Loaders

> [!IMPORTANT]
> The loaders are demonstrated for single-GPU fetching. Please follow [PyTorch's distributed training guideline](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) or [third-party distributed training techniques](https://github.com/huggingface/accelerate) to make the loaders distributed.

We provide two types of data loaders for text-to-3D object generation tasks, each optimized for different use cases:

1. **Standard Setting**
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



2. **Fast Setting**
This loader uses WebDataset format for optimized loading of 3DGS data and captions only, ideal for large-scale training.

> [!WARNING]
> You will need preprocessed .tar files as instructed in the [WebDataset Preprocessing section](data/README.md#webdataset-preprocessing-optional).

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


## Models

To be updated soon!

## Q&A

<details>
<summary><strong>Why does this project exist?</strong></summary>
<br>

While many excellent 3D generation codebases exist (e.g., <a href="https://github.com/microsoft/TRELLIS">TRELLIS</a> and <a href="https://github.com/chenguolin/DiffSplat">DiffSplat</a>), the field lacks a standardized training data protocol. This inconsistency makes fair model comparisons nearly impossible—similar to how ImageNet standardized class-to-image generation benchmarking.

The challenge is two-fold:
<ul>
<li><strong>Inconsistent training data:</strong> Different models use different datasets, making performance comparisons unreliable.</li>
<li><strong>Accessibility barriers:</strong> Many projects rely heavily on photometric supervision, requiring multi-TB 2D rendering datasets that are difficult to store and process efficiently.</li>
</ul>

The 3D Gen Playground addresses these issues by leveraging the publicly available <a href="https://cs.stanford.edu/~xtiange/projects/gaussianverse/">GaussianVerse</a> 3DGS dataset. This enables native 3D generation without photometric dependencies, while establishing a consistent training protocol for the community.
</details>

<details>
<summary><strong>Why use 3DGS?</strong></summary>
<br>

3D Gaussian Splatting (3DGS) has become increasingly popular in 3D generation due to its explicit parameterization and exceptional rendering quality. Recent projects like GaussianAtlas, DreamGaussian, LGM, DiffSplat, and <a href="https://supergaussian.github.io/">SuperGaussian</a> have demonstrated its effectiveness.

While 3D content can be represented in various formats (meshes, voxels, point clouds, etc.), these representations are intrinsically related. As noted by <a href="https://github.com/microsoft/TRELLIS">TRELLIS</a>, successfully generating one representation (such as 3DGS) provides a strong foundation that can be easily extended to other 3D formats. This makes 3DGS an ideal starting point for general 3D generation research. Moreover, 3DGS can be easily converted into <a href="https://github.com/Anttwo/SuGaR">meshes</a> and other formats.
</details>

<details>
<summary><strong>Isn't 250K samples insufficient for production-scale training?</strong></summary>
<br>

You're correct that 250K samples may not be enough for training production-ready models at scale. However, the 3D Gen Playground is specifically designed for rapid prototyping and research validation.

Key points:
<ul>
<li><strong>Research-focused:</strong> This dataset enables quick verification of new ideas and model architectures.</li>
<li><strong>Competitive scale:</strong> 250K samples exceeds the training data used in many published 3D generation projects.</li>
<li><strong>Proof of concept:</strong> The dataset is sufficient for demonstrating novel concepts and validating approaches before scaling up.</li>
<li><strong>Growing ecosystem:</strong> We will continue to release additional high-quality data in future updates—stay tuned!</li>
</ul>

For researchers seeking to validate their ideas before investing in larger-scale infrastructure, this dataset provides an accessible and standardized starting point.
</details> 

## Citations
Please cite the 3D Gen Playground if you find it useful in your research!

```bibtex
@misc{xiang2025_3dgen_playground,
  author       = {Tiange Xiang},
  title        = {3DGen-Playground},
  year         = {2025},
  howpublished = {\url{https://github.com/tiangexiang/3DGen-Playground}},
  note         = {GitHub repository}
}
```

## Contributions

We are always open to suggestions & contributions from the community, feel free to open an issue or submit a PR.

## Acknowledgements

This project builds upon the excellent work of the open-source community:

* We utilize 3D parameterizations from [GaussianVerse](https://cs.stanford.edu/~xtiange/projects/gaussianverse/), 2D renderings from [GObjaverse](https://aigc3d.github.io/gobjaverse/), and captions from [3DTopia](https://github.com/3DTopia/3DTopia) and [Cap3D](https://cap3d-um.github.io/).
* Parts of the codebase were inspired by [GaussianAtlas](https://cs.stanford.edu/~xtiange/projects/gaussianatlas/), [GaussianCube](https://gaussiancube.github.io/), and [LGM](https://me.kiui.moe/lgm/).
* Our web rendering is powered by [Spark](https://github.com/sparkjsdev/spark) @ [World Labs](https://www.worldlabs.ai/).
* We thank [Ryan Zhijie Rong](https://ryan-rong-24.github.io/) for his contributions to part of this project.

We are grateful to all contributors and maintainers of these projects for advancing the 3D generation community.