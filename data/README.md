# Data Downloading

Every model training requires data. At The 3D Gen Playground, we are dedicated to making 3D data open and consistent. We provide public access to a fixed set of data to make 3D Gen model training consistent. All you need to do is download and run a few preprocessing steps.

This page provides detailed instructions on how to (very easily) set up consistent training data for the 3D object generation task.

## Overview

For (text/image to) 3D object generation tasks, there are three different data types usually required:

1. **3D parameterizations**
2. **Text captions**
3. **2D renderings**

Thanks to the open-source community, the task of 3D object generation has been made easier by utilizing high-quality data that has been precomputed and publicly released by researchers in the community.

Please see the Q&A section for explanations & clarifications.

---

## 1. 3D Parameterizations

> [!NOTE]
> **💾 Disk Storage Requirements**
> - Aesthetic list: 285 GB
> - Non-aesthetic list: 600 GB
> - **Total: 885 GB**

Unlike images and videos where pixels are the common standard representation, 3D objects can be represented in many different (but related) forms: such as meshes, point clouds, implicit functions, and more recently neural fields and Gaussian splats.

In this project, we use [**3D Gaussian Splats**](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) as the parameterization, due to its explicit parameterization and amazing rendering quality.

We use the high-quality 3DGS fittings from [**Gaussian Verse**](https://gaussianverse.stanford.edu).

To start automatic downloading, update the paths in `download_3dgs.sh` and run:

```bash
source download_3dgs.sh
```

---

## 2. Text Captions

> [!NOTE]
> **💾 Disk Storage Requirements**
> - **Total: 371 MB**

We use the captions from [**Cap3D**](https://huggingface.co/datasets/tiange/Cap3D/tree/main) and [**3DTopia** ](https://github.com/3DTopia/3DTopia/releases).

To start automatic downloading, update the paths in `download_captions.sh` and run:

```bash
source download_captions.sh
```

---

## 3. 2D Renderings (Optional for Text-to-3D Task)

> [!WARNING]
> **Important Note:** Downloading 2D renderings is slow and requires A LOT of disk space!

> [!NOTE]
> **💾 Disk Storage Requirements**
> - Aesthetic list: 3.14 TB
> - Non-aesthetic list: 5.33 TB
> - **Total: 8.47 TB**

2D renderings provide additional photometric guidance during training. However, we can see that the 2D renderings require significantly more disk storage compared to 3D parameterizations (~10x). Therefore, we suggest optional use of 2D renderings in training.

We use the 2D renderings from [**GObjaverse**](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse).

**Note:** It is highly recommended to use multiple CPUs for the download.

To start automatic downloading, update the paths in `download_renderings.sh` and run:

```bash
source download_renderings.sh
```



# WebDataset Preprocessing (Optional)

> [!NOTE]
> **Important Note:** Making WebDatasets is only for text-to-3D generation tasks, after the 3DGS data and captions are downloaded. No images will be needed or encoded in the .tar files due to their scale. You will need ~900GB additional disk storage for storing the processed .tar files.


We provide a preprocessing script that converts 3DGS fittings and text captions into WebDataset-formatted .tar files for optimized loading. To start processing, update the paths in `make_webdataset.sh` and run:

```bash
source make_webdataset.sh
```