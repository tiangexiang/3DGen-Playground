"""Pin the promoted recipe, launch behavior, and matching invariants."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mini3d.cli import build_cfg
from mini3d.config import TrainConfig
from mini3d.matching import _chamfer_recon_loss
from mini3d.model import build_model
from mini3d.utils import update_ema

ROOT = Path(__file__).resolve().parents[1]


def default_cfg(**kwargs):
    args = argparse.Namespace(config_file=str(ROOT / 'configs/train.yml'),
        output_dir=None, run_name=None, set=[], no_wandb=False, print_config=True)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return build_cfg(args)


def test_promoted_recipe():
    cfg = default_cfg()
    expected = dict(pca_channels=128, group_size=64, hidden_size=768,
        depth_double=17, txt_refine_depth=None, time_cond='adaln_img',
        optimizer='muon', ema_decay=.999, batch_size=512, learning_rate=.0003,
        sinkhorn_weight=1., sinkhorn_iters=50, huber_delta=2.,
        render_loss_weight=8., alpha_mask_loss_weight=4., lpips_loss_weight=8.,
        render_loss_noise_cutoff=.31, render_fg_phi=.5, max_grad_norm=50.,
        lazy_cache_to_cpu=True, cfg_scale=4., sampler_exact_final=True,
        heldout_size=5000, cmmd_num_samples=2000, val_steps=50)
    for name, value in expected.items():
        assert getattr(cfg, name) == value, name
    assert not any('ema_reset' in k or 'ema_initial' in k for k in cfg.to_dict())
    with torch.device('meta'):
        model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == 314349312


@pytest.mark.parametrize('ranks,accum', [(1,16),(2,8),(4,4),(8,2)])
def test_global_batch_independent_of_gpu_count(ranks, accum):
    cfg = default_cfg().resolve_batch(ranks)
    assert cfg.grad_accum_steps == accum
    assert ranks * cfg.micro_batch_size * cfg.grad_accum_steps == 512


def test_explicit_batch_validation_and_optional_numeric_overrides():
    cfg = default_cfg(set=['micro_batch_size=256', 'grad_accum_steps=1'])
    cfg.resolve_batch(2)
    assert cfg.batch_size == 512 and cfg.grad_accum_steps == 1
    with pytest.raises(ValueError, match='actual'):
        cfg.resolve_batch(1)
    with pytest.raises(ValueError, match='divisible'):
        default_cfg(set=['micro_batch_size=100']).resolve_batch(2)
    with pytest.raises(KeyError, match='unknown'):
        default_cfg(set=['ema_decya=.999'])


def test_entrypoint_default_config_works_outside_repo(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT/'train.py'), '--print-config'],
        cwd=tmp_path, check=True, capture_output=True, text=True)
    cfg = json.loads(result.stdout)
    assert cfg['optimizer'] == 'muon' and cfg['ema_decay'] == .999
    assert cfg['cfg_scale'] == 4. and cfg['batch_size'] == 512
    assert not (tmp_path/'output').exists()


def test_short_ema_applies_from_first_update():
    model = torch.nn.Linear(1, 1, bias=False)
    ema = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(10.)
        ema.weight.fill_(0.)
    update_ema(ema, model, default_cfg().ema_decay)
    torch.testing.assert_close(ema.weight, torch.full_like(ema.weight, .01))


def test_patch_matching_permutation_invariance_and_gradients():
    torch.manual_seed(1)
    pred = torch.randn(2, 14, 4, 4, requires_grad=True)
    target = torch.randn_like(pred)
    kwargs = dict(mode='sinkhorn_patch_hard', patch_size=4, chunk_1d=True,
                  sinkhorn_iters=50, sinkhorn_eps=.05, huber_delta=2., weight=.07)
    a = _chamfer_recon_loss(pred, target, **kwargs)
    # Permute within each contiguous group, preserving the Hilbert chunk layout.
    alternate = target.flatten(2).reshape(2,14,4,4).flip(-1).reshape_as(target)
    b = _chamfer_recon_loss(pred, alternate, **kwargs)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    a.mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0
