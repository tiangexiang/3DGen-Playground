# Gaussian Splat Diffusion baseline

The [default configuration](../configs/train.yml) trains a text-conditioned
transformer over Hilbert-ordered Gaussian splats: a 128-channel bottleneck,
768-wide, 17-block model, and frozen FLAN-T5-Large tokens. EMA decay is **0.999
from the first optimizer step**.

## Setup

Install CUDA-compatible PyTorch/torchvision, then:

```bash
pip install -r requirements-training.txt
cp configs/local.example.yml configs/local.yml
```

Rendering requires a CUDA GPU and a compatible CUDA compiler for gsplat's
first-use compilation, with `ninja` on `PATH`. PyTorch3D and KeOps are optional
for matching modes other than the default.

[Download and unpack the data](../data/README.md). Object-list entries map hashes
to `directory/file.tar.gz`; each unpacked directory must contain
`point_cloud.ply` and `gs2sphere.npy`.

Edit the Git-ignored `configs/local.yml`: set `obj_list`, `gs_path`,
`caption_path`, and the output locations. Allow roughly 106 GiB for `cache_dir`
and 47 GB for `t5_token_dir` on the reference dataset, plus checkpoint storage.
`num_workers` is per GPU process. Enable W&B with `wandb: true` and `project`.

Path and configuration rules:

- Data/output paths are relative to the repository root; reference files are
  relative to `asset_dir`. Absolute paths, `~`, and `$ENV_VAR` are supported.
- `_base` is relative to its YAML file. Precedence is defaults/environment,
  YAML, then CLI overrides (`--set KEY=VALUE`). Unknown keys are rejected.
- `train.sh` loads `.env`; direct Python commands use exported variables.
  The local YAML works with either entry point.

## Local reference assets

Set `asset_dir` (or its default, `DGEN_ASSET_ROOT`) to a directory containing:

```text
baseline_assets/
├── ref_cameras.tar.gz
├── sphere2plane.npy
└── stats/
    ├── all_mean_postfix.pt
    ├── all_std_postfix.pt
    ├── clip_thresholds_opacity_scales.pt
    └── rank_quantiles_8ch_clipped.pt
```

Supply these files from a compatible training setup; preparation does not
download them. The [manifest](../data/baseline_assets.json) records their sizes
and SHA-256 hashes. The normalization, clipping, and rank tables are specific
to this preprocessing; original GaussianVerse mean/std files are not substitutes.

## Prepare and train

```bash
python prepare.py --config-file configs/local.yml
bash train.sh --config-file configs/local.yml
```

Preparation encodes text and generates per-object camera framing. Use
`--only text` or `--only framing` to run one stage. Text encoding resumes partial
outputs, uses CUDA when available, and accepts `--device cpu` or
`--text-batch-size 32` to reduce memory use. Framing runs on CPU and recomputes
its table. See `python tools/encode_t5_tokens.py --help` for sharding options.

For offline nodes, pre-cache `google/flan-t5-large`,
`openai/clip-vit-large-patch14`, and LPIPS's torchvision VGG16 weights; preserve
the Hugging Face and torch caches.

The launcher uses all visible GPUs unless `NPROC_PER_NODE` is set. Automatic
accumulation preserves a **512-sample global batch**: with microbatch 32, it
uses 16/8/4 accumulation steps on 1/2/4 GPUs. For example:

```bash
NPROC_PER_NODE=2 bash train.sh --config-file configs/local.yml --set micro_batch_size=64
```

Set `batch_size` explicitly to change the global batch; inconsistent explicit
accumulation settings are rejected. Custom launchers can use
`torchrun --standalone --nproc_per_node=4 train.py --config-file configs/local.yml`.
Inspect defaults with `python train.py --print-config`.

Training automatically resumes the latest checkpoint in `output_dir`.
Use a fresh `--output-dir` for a new run, `--set auto_resume=false` to disable
resume, or `--set resume_from=/path/to/checkpoint.pt` to select a checkpoint.
Checkpoints include raw/EMA weights, optimizer state, step, config, and Git
provenance. `--no-wandb` disables logging to W&B.

## Sample and evaluate

```bash
python -m mini3d sample --config-file configs/local.yml --checkpoint /path/to/checkpoint.pt \
  --prompt "a blue phone booth" --out output/phone_booth.png
python -m mini3d eval --config-file configs/local.yml --checkpoint /path/to/checkpoint.pt
```

Both use EMA weights unless `--checkpoint-target model` is selected. Sampling
uses CFG **4.0**, 100 Euler steps, and exact final-step prediction. Evaluation
uses 2,000 samples and 50 steps. The held-out split is the last 5,000 objects
sorted by caption stem, after dropping objects without cached text tokens.

## Recipe and reference result

The loss combines v-space MSE, hard patch Sinkhorn, and RGB/alpha/LPIPS render
losses weighted 8/4/8. Render losses start at step 5,000 for `t >= 0.31`, with
half the pixel weight assigned to foreground. Muon/AdamW learning rates warm
up for 5,000 steps, then stay constant; gradient norm is capped at 50.

The step-134,000 reference checkpoint scored **CMMD 0.1882318047** using the
evaluation settings above. That run shortened EMA after 100k steps; this
configuration uses short EMA throughout. A fresh full run has not been evaluated.

Keep lazy caching enabled when comparing with the reference: the uncached
loader explicitly clips targets before rank transformation, while the lazy
path skips that step. Disabling it changes the preprocessing.

## Verification

```bash
pip install -r requirements-test.txt
python -m pytest tests -q
# Create configs/local-smoke.yml with _base: smoke.yml and your local paths.
bash train.sh --config-file configs/local-smoke.yml --output-dir output/smoke --no-wandb
```

The four-step smoke exercises data, losses, optimization, checkpoints, and
sampling with a small model; it does not measure generation quality.

Model and flow-matching code are adapted from
[MiniT2I](https://github.com/Hope7Happiness/minit2i-torch).
