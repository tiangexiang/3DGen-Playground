"""End-to-end sampler checks against the real MMJiT3D (CPU, tiny config).

Note the deliberate ``final_layer_zero=False``: at init the final layer is zero,
so the model predicts x0=0 everywhere, the sampler correctly converges to all
zeros, and CFG has nothing to amplify. Testing the sampler on an init model would
therefore pass trivially while proving nothing.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini3d import diffusion  # noqa: E402
from mini3d.config import TrainConfig  # noqa: E402
from mini3d.model import build_model  # noqa: E402


def tiny(**kw):
    c = TrainConfig()
    c.hidden_size = c.txt_hidden_size = c.cond_vec_size = 128
    c.depth_double, c.text_preamble_depth = 2, 1
    c.num_heads, c.head_dim, c.pca_channels = 2, 64, 16
    c.prompt_length, c.txt_input_size = 32, 512
    for k, v in kw.items():
        setattr(c, k, v)
    return c.validate()


def _sample(m, c, tok, msk, seed=0, **kw):
    torch.manual_seed(seed)
    return diffusion.euler_sample(m, tok, msk, (c.in_channels, c.container_size, c.container_size),
                                  noise_scale=c.noise_scale, **kw)


def test_untrained_model_collapses_to_zero():
    c = tiny(); m = build_model(c).eval()
    tok, msk = torch.randn(2, 32, 512), torch.ones(2, 32, dtype=torch.bool)
    x = _sample(m, c, tok, msk, steps=8, cfg_scale=2.5)
    assert x.abs().max() == 0, "zero-init final layer must give an all-zero sample"
    print("  zero-init model -> all-zero sample   OK")


def test_sampler_cfg_and_caption_sensitivity():
    c = tiny(final_layer_zero=False)
    torch.manual_seed(0); m = build_model(c).eval()
    tok, msk = torch.randn(2, 32, 512), torch.ones(2, 32, dtype=torch.bool)

    x = _sample(m, c, tok, msk, steps=8, cfg_scale=2.5, v_clamp_min=c.sampler_v_clamp_min)
    assert torch.isfinite(x).all() and x.std() > 0
    print(f"  sample finite & non-degenerate       OK  (std {x.std():.3f})")

    a = _sample(m, c, tok, msk, steps=4, cfg_scale=1.0)
    b = _sample(m, c, tok, msk, steps=4, cfg_scale=5.0)
    assert not torch.allclose(a, b), "CFG scale had no effect"
    print(f"  cfg 1.0 != 5.0                       OK  (max|d| {(a - b).abs().max():.2e})")

    d = _sample(m, c, torch.randn(2, 32, 512), msk, steps=4, cfg_scale=2.5)
    e = _sample(m, c, tok, msk, steps=4, cfg_scale=2.5)
    assert not torch.allclose(d, e), "caption had no effect"
    print(f"  different captions differ            OK  (max|d| {(d - e).abs().max():.2e})")


def test_sampler_clamp_changes_trajectory():
    """sampler_v_clamp_min must be a real knob, not decoration."""
    c = tiny(final_layer_zero=False)
    torch.manual_seed(0); m = build_model(c).eval()
    tok, msk = torch.randn(2, 32, 512), torch.ones(2, 32, dtype=torch.bool)
    exact = _sample(m, c, tok, msk, steps=100, cfg_scale=1.0, v_clamp_min=0.0)
    clamped = _sample(m, c, tok, msk, steps=100, cfg_scale=1.0, v_clamp_min=0.05)
    assert not torch.allclose(exact, clamped)
    print(f"  clamp 0 vs 0.05 differ               OK  (max|d| {(exact - clamped).abs().max():.2e})")


if __name__ == "__main__":
    for fn in [test_untrained_model_collapses_to_zero,
               test_sampler_cfg_and_caption_sensitivity,
               test_sampler_clamp_changes_trajectory]:
        print(f"{fn.__name__}:")
        fn()
    print("\nAll sampler tests passed.")
