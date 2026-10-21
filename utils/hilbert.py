"""3D Hilbert-curve ordering for point sets.

Provides a locality-preserving 1D ordering of 3D points: any contiguous run of
the returned permutation maps to a spatially compact region. This is the 1D
analog of a 2D image patch — it lets us tokenize a set of 3D Gaussians into 1D
chunks whose members are spatial neighbours, so a within-chunk set loss matches
nearby points.

The encoder is Skilling's (2004) axes<->transpose Hilbert construction,
vectorised over points with numpy; only the bit/dimension loops (small constants)
are Python.
"""
from __future__ import annotations

import numpy as np


def _axes_to_transpose(X: np.ndarray, nbits: int) -> None:
    """Skilling's AxesToTranspose, in place.

    ``X``: (N, D) int64 array, each coordinate in ``[0, 2**nbits)``. On return the
    rows hold the Hilbert *transpose* form (D integers per point); interleave their
    bits (``_transpose_to_index``) to get the scalar Hilbert distance.
    """
    N, D = X.shape
    M = 1 << (nbits - 1)
    # Inverse undo excess work. Full-array np.where (not boolean fancy-index
    # assignment) — ~6x faster: masked in-place XOR materialises index arrays each step.
    q = M
    while q > 1:
        p = q - 1
        for i in range(D):
            bit = (X[:, i] & q) > 0
            x0 = X[:, 0]
            xi = X[:, i]
            t = (x0 ^ xi) & p
            # bit set -> invert low bits of X[:,0]; else exchange low bits of X[:,0]/X[:,i].
            new0 = np.where(bit, x0 ^ p, x0 ^ t)
            if i != 0:
                X[:, i] = np.where(bit, xi, xi ^ t)  # for i==0 this column IS X[:,0]; skip
            X[:, 0] = new0
        q >>= 1
    # Gray encode.
    for i in range(1, D):
        X[:, i] ^= X[:, i - 1]
    t = np.zeros(N, dtype=X.dtype)
    q = M
    while q > 1:
        mask = (X[:, D - 1] & q) > 0
        t[mask] ^= q - 1
        q >>= 1
    for i in range(D):
        X[:, i] ^= t


def _transpose_to_index(X: np.ndarray, nbits: int) -> np.ndarray:
    """Interleave the D transpose integers (MSB-first, dim-major) into one index."""
    N, D = X.shape
    H = np.zeros(N, dtype=np.int64)
    Xi = X.astype(np.int64)
    for b in range(nbits - 1, -1, -1):
        for i in range(D):
            H = (H << 1) | ((Xi[:, i] >> b) & 1)
    return H


def hilbert_distance(coords_int: np.ndarray, nbits: int) -> np.ndarray:
    """Hilbert distance of integer grid coords ``(N, D)`` in ``[0, 2**nbits)``."""
    # int32 for the transform (per-axis values are < 2**nbits): halves memory
    # bandwidth of the np.where sweep vs int64. Interleave accumulates in int64.
    X = np.ascontiguousarray(coords_int, dtype=np.int32).copy()
    if X.ndim != 2:
        raise ValueError(f"coords_int must be (N, D), got {X.shape}")
    if nbits >= 31:
        raise ValueError(f"nbits={nbits} too large for int32 transform")
    if X.min(initial=0) < 0 or X.max(initial=0) >= (1 << nbits):
        raise ValueError(f"coords out of range [0, 2**{nbits})")
    _axes_to_transpose(X, nbits)
    return _transpose_to_index(X, nbits)


def hilbert_sort_indices(xyz: np.ndarray, bits: int = 8) -> np.ndarray:
    """Permutation ordering 3D points along a Hilbert curve.

    ``xyz``: (N, 3) float coordinates (any units). Returns ``order`` (N,) such that
    ``xyz[order]`` walks the Hilbert curve — contiguous slices are spatially compact.

    Points are quantised into a ``2**bits`` cube that preserves aspect ratio (a
    single isotropic scale, not per-axis min-max), so compactness is measured in the
    true metric, not a stretched one. Cell ties break by original index (stable).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    lo = xyz.min(axis=0)
    ext = float((xyz - lo).max())
    if ext <= 0.0:
        ext = 1.0
    scale = (1 << bits) - 1
    q = np.floor((xyz - lo) / ext * scale).astype(np.int64)
    q = np.clip(q, 0, (1 << bits) - 1)
    H = hilbert_distance(q, bits)
    # Stable sort keeps ties in original-index order (deterministic) in one key pass.
    return np.argsort(H, kind="stable")


def _self_test() -> None:
    # Bijection + adjacency on a fully-populated small grid.
    for bits in (2, 3, 4):
        side = 1 << bits
        g = np.arange(side)
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        H = hilbert_distance(coords, bits)
        assert np.array_equal(np.sort(H), np.arange(side**3)), f"not a bijection at bits={bits}"
        order = np.argsort(H)
        steps = np.abs(np.diff(coords[order].astype(np.int64), axis=0)).sum(axis=1)
        assert np.all(steps == 1), f"non-adjacent step at bits={bits}: max L1 step={steps.max()}"
    # Locality: contiguous Hilbert chunks are far tighter than random chunks.
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((16384, 3))
    order = hilbert_sort_indices(pts, bits=8)
    G = 64
    xs = pts[order].reshape(-1, G, 3)
    hil_diam = np.linalg.norm(xs.max(1) - xs.min(1), axis=1).mean()
    xr = pts.reshape(-1, G, 3)
    rnd_diam = np.linalg.norm(xr.max(1) - xr.min(1), axis=1).mean()
    assert hil_diam < 0.25 * rnd_diam, f"weak locality: hilbert={hil_diam:.3f} random={rnd_diam:.3f}"
    print(f"[hilbert] self-test OK  (chunk diam: hilbert={hil_diam:.3f} vs random={rnd_diam:.3f})")


if __name__ == "__main__":
    _self_test()
