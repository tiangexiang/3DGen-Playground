# JiT Inference

Generate and render 3DGS objects from a trained JiT checkpoint.

## Quick start

```bash
# 1. Copy and fill in the required paths
cp jit/configs/infer_gsplat.yaml jit/configs/my_infer.yaml
# Edit my_infer.yaml: set checkpoint, ref_camera_tar, mean_file, std_file

# 2. Run
python jit/infer_gsplat.py --config jit/configs/my_infer.yaml
```

Output PNGs land in `output_dir` (default: `output/infer_results/`), one file per camera view per sample:

```
sample0000_class012_step0100000_cam000.png
sample0000_class012_step0100000_cam045.png
...
```

CLI args always override the YAML:

```bash
python jit/infer_gsplat.py --config jit/configs/my_infer.yaml \
    --num_steps 200 --cfg_scale 6.0 --num_samples 16
```

---

## Required paths

These must be set — either in the YAML or on the CLI.

| Argument | Description |
|---|---|
| `--checkpoint` | Path to `.pt` checkpoint file |
| `--ref_camera_tar` | Path to `ref_camera.tar.gz` (same as training) |
| `--mean_file` | Path to normalization mean `.pt` (e.g. `gaussianverse_mean.pt`) |
| `--std_file` | Path to normalization std `.pt` (e.g. `gaussianverse_std.pt`) |
| `--sphere2plane_path` | Path to `data/sphere2plane.npy` |
| `--class_map` | Path to `object_labels/object_to_class.json` |

---

## Model options

These are **auto-detected from the checkpoint** (`ckpt["args"]`) and should not need to be set manually unless loading a checkpoint that predates stored args.

| Argument | Default | Description |
|---|---|---|
| `--model` | from ckpt | Architecture variant: `JiT-S/8`, `JiT-B/8`, `JiT-L/8`, `JiT-XL/8` |
| `--predict_xstart` | from ckpt | Whether the model predicts x0 directly (vs. epsilon) |
| `--noise_schedule` | from ckpt | Beta schedule: `linear` or `squaredcos_cap_v2` |
| `--sh_degree0_only` | from ckpt | Use 14-channel DC-only features instead of full 59 |
| `--class_dropout_prob` | from ckpt | Label dropout (affects CFG null embedding) |
| `--use_ema` | `true` | Use EMA weights from checkpoint (recommended) |

---

## Sampling knobs

### Sampler

```yaml
sampler: heun   # heun | euler | dpm | ddpm
num_steps: 100  # denoising steps — more = slower, often better up to ~200
```

`heun` and `euler` use the JiT flow-matching ODE. `dpm` uses the diffusers DPM-Solver. `ddpm` is DDPM ancestral sampling.

### Classifier-free guidance — heun/euler only

```yaml
cfg_scale: 4.0          # 1.0 = no guidance; typically 2.0–7.0
cfg_interval: [0.0, 1.0] # restrict CFG to t in (low, high); [0.0, 1.0] = always on
```

`cfg_scale` trades diversity for sharpness/class-fidelity. `cfg_interval` lets you apply guidance only in a sub-range of the trajectory (e.g. `[0.0, 0.8]` skips guidance at the very end where the model is mostly refining fine details).

### JiT ODE knobs — heun/euler only

```yaml
t_eps: 0.05       # minimum t to avoid div-by-zero near t=1; try 0.01–0.1
noise_scale: 1.0  # scale on the initial noise draw; <1 = less noise, >1 = more
```

### DPM-Solver knobs — dpm only

```yaml
dpm_solver_order: 2          # 1 | 2 | 3 — higher = fewer steps needed
dpm_algorithm_type: "dpmsolver++"  # dpmsolver | dpmsolver++ | sde-dpmsolver | sde-dpmsolver++
dpm_solver_type: midpoint    # midpoint | heun
dpm_timestep_spacing: trailing  # linspace | leading | trailing
dpm_use_karras_sigmas: false
```

---

## Generation options

| Argument | Default | Description |
|---|---|---|
| `--num_samples` | `4` | Number of 3DGS objects to generate |
| `--class_label` | random | Fix to a specific class index; omit for a random class each sample |
| `--seed` | none | Set for reproducible outputs |

---

## Rendering options

| Argument | Default | Description |
|---|---|---|
| `--render_size` | `256` | Output image resolution in pixels (square) |
| `--num_cameras` | `4` | Camera views to render per generated object |
| `--random_cameras` | `false` | Random camera selection vs. evenly spaced (default) |

---

## Tuning tips

**The model isn't generating recognisable shapes:**
- Confirm `--predict_xstart` and `--noise_schedule` match training (auto-detected from the checkpoint).
- Try increasing `--num_steps` to 200.
- Try `--cfg_scale 3.0`–`6.0` if training used class conditioning.

**Samples look blurry / low-frequency:**
- Increase `--cfg_scale`.
- Try `--noise_scale 0.9`–`1.1` to shift the noise level.
- Decrease `--t_eps` (e.g. `0.01`) to integrate closer to t=1.

**Samples look oversmoothed or saturated:**
- Reduce `--cfg_scale`.
- Narrow `--cfg_interval`, e.g. `[0.1, 0.8]`.

**Trying DPM-Solver for faster generation:**
```yaml
sampler: dpm
num_steps: 20
dpm_solver_order: 3
dpm_algorithm_type: "dpmsolver++"
dpm_use_karras_sigmas: true
```
