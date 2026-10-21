"""Public checkout portability and shared preparation/training configuration."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from utils import training_paths as settings
from mini3d.cli import _coerce
from mini3d.config import TrainConfig
from tools.prepare_training_data import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_fresh_checkout_without_git_or_original_paths(tmp_path):
    checkout = tmp_path / 'checkout'
    checkout.mkdir()
    for directory in ('mini3d', 'configs', 'utils'):
        shutil.copytree(ROOT / directory, checkout / directory,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT / 'train.py', checkout / 'train.py')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('DGEN_',))
           and k not in ('GS_PATH', 'ALL_OBJ_JSON', 'CAPTIONS_PATH')}
    env['PYTHONPATH'] = str(checkout)
    result = subprocess.run(
        [sys.executable, str(checkout / 'train.py'), '--print-config'],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    cfg = json.loads(result.stdout)
    assert cfg['gs_path'] == str(checkout / 'datasets/3dgs')
    assert cfg['cache_dir'] == str(checkout / 'output/cache/preload_text')
    assert cfg['mean_file'] == str(checkout / 'datasets/baseline_assets/stats/all_mean_postfix.pt')
    assert str(ROOT) not in result.stdout
    assert not (checkout / 'output').exists()
    result = subprocess.run(
        [sys.executable, '-c', 'from utils.training_paths import repository_revision; '
         'import json; print(json.dumps(repository_revision()))'],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == dict(git_sha=None, git_branch=None, git_dirty=None)


def test_paths_expand_without_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv('TRAIN_DATA', str(tmp_path / 'data'))
    cfg = TrainConfig(gs_path='$TRAIN_DATA/splats', output_dir=str(tmp_path / 'run'),
                      cache_dir='output/test-cache', resume_from='latest').resolve()
    assert cfg.gs_path == str(tmp_path / 'data/splats')
    assert cfg.cache_dir == str(ROOT / 'output/test-cache')
    assert cfg.resume_from == 'latest'
    assert settings.asset('~/data') == str(Path.home() / 'data')
    assert not (tmp_path / 'run').exists()


def test_invalid_override_and_cyclic_yaml(tmp_path):
    with pytest.raises(ValueError, match='boolean'):
        _coerce(TrainConfig(), 'wandb', 'treu')
    with pytest.raises(KeyError, match='unknown'):
        _coerce(TrainConfig(), 'resolve', 'bad')
    with pytest.raises(KeyError, match='unknown'):
        TrainConfig().update_from_mapping({'resolve': 'bad'})
    a, b = tmp_path / 'a.yml', tmp_path / 'b.yml'
    a.write_text('_base: b.yml\n')
    b.write_text('_base: a.yml\n')
    with pytest.raises(ValueError, match='Cyclic'):
        TrainConfig().update_from_yaml(a)


def test_prepare_forwards_config_to_both_stages(tmp_path, monkeypatch):
    for name in ('objects.json', 'captions.json', 'sphere2plane.npy', 'ref_cameras.tar.gz'):
        (tmp_path / name).write_text('{}')
    cfg = TrainConfig(
        obj_list=str(tmp_path / 'objects.json'), caption_path=str(tmp_path / 'captions.json'),
        asset_dir=str(tmp_path), gs_path=str(tmp_path), t5_token_dir=str(tmp_path / 'tokens'),
        per_sample_zoom_file=str(tmp_path / 'framing.pt'), exclude_keys_file=None,
        prompt_length=48, train_render_size=112, num_workers=0,
    )
    calls = []
    monkeypatch.setattr('tools.prepare_training_data.subprocess.run', lambda cmd, **kw: calls.append((cmd, kw)))
    prepare(cfg, text_batch_size=8, device='cpu')
    text, framing = [call[0] for call in calls]
    def value(cmd, flag):
        return cmd[cmd.index(flag) + 1]
    assert value(text, '--obj-list') == value(framing, '--obj_list') == cfg.obj_list
    assert value(text, '--output') == cfg.t5_token_dir
    assert value(text, '--captions') == cfg.caption_path
    assert value(text, '--prompt-length') == '48'
    assert value(text, '--batch-size') == '8'
    assert value(text, '--device') == 'cpu'
    assert value(framing, '--output') == cfg.per_sample_zoom_file
    assert value(framing, '--exclude_keys_file') == ''
    assert value(framing, '--train_render_size') == '112'
    assert value(framing, '--num_workers') == '1'
    assert all(kw == dict(cwd=ROOT, check=True) for _, kw in calls)
    calls.clear()
    cfg.caption_path = str(tmp_path / 'missing-captions.json')
    prepare(cfg, only='framing')
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError, match='caption_path'):
        prepare(cfg, only='text')


def test_repository_provenance_without_git_executable(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError('git')
    monkeypatch.setattr(settings.subprocess, 'run', missing_git)
    assert settings.repository_revision() == dict(git_sha=None, git_branch=None, git_dirty=None)
