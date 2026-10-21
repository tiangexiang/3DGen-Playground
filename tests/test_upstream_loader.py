"""Keep upstream plane ordering and class conditioning usable with the baseline loader."""
import json

import numpy as np
import pytest
import torch

from dataloaders.class_3dgen_loader import Class3DGenDataset
from dataloaders.standard_3dgen_loader import Standard3DGenDataset


@pytest.mark.parametrize('lazy_cache', [False, True])
def test_upstream_plane_order_normalization_and_class_wrapper(tmp_path, monkeypatch, lazy_cache):
    source = tmp_path / 'objects/1000/1234'
    source.mkdir(parents=True)
    np.save(source / 'gs2sphere.npy', np.array([2, 0, 3, 1]))
    np.save(tmp_path / 'sphere2plane.npy', np.array([2, 0, 1, 3]))
    obj_list = tmp_path / 'objects.json'
    obj_list.write_text(json.dumps({'object-hash': '1000/1234.tar.gz'}))
    points = np.zeros((4, 59), dtype=np.float32)
    points[:, :3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    points[:, 3] = np.arange(4)
    monkeypatch.setattr('dataloaders.standard_3dgen_loader.load_ply', lambda path: points.copy())
    monkeypatch.setenv('DGEN_CACHE_ROOT', str(tmp_path / 'cache'))
    torch.save(torch.ones(59), tmp_path / 'mean.pt')
    torch.save(torch.full((59,), 2.), tmp_path / 'std.pt')
    base = Standard3DGenDataset(
        obj_list=[str(obj_list)], gs_path=str(tmp_path / 'objects'),
        sphere2plane_path=str(tmp_path / 'sphere2plane.npy'),
        mean_file=str(tmp_path / 'mean.pt'), std_file=str(tmp_path / 'std.pt'),
    )
    # Lexicographic xyz order [3,2,1,0], then the two stored permutations.
    expected = ((points[[0, 1, 3, 2]] - 1) / 2).T.reshape(59, 2, 2)
    np.testing.assert_allclose(base[0]['point_cloud'], expected)
    wrapped = Class3DGenDataset(base, class_map={'1000/1234': 7}, lazy_cache_to_cpu=lazy_cache)
    cloud, label, key = wrapped[0]
    torch.testing.assert_close(cloud, torch.from_numpy(expected))
    assert int(label) == 7 and key == 'object-hash'
