"""Verify the ported flow-matching math against hand-computed values.

The v-space loss is easy to port subtly wrong (which denominator, which clamp,
which sign) and a wrong version still trains -- just to a different objective.
So these assert the exact algebra, not just "it runs".
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini3d import diffusion as D  # noqa: E402


class Const(torch.nn.Module):
    """Model that ignores its input and returns a fixed prediction."""
    def __init__(self, pred): super().__init__(); self.pred = pred
    def forward(self, x, t, ctx, mask): return self.pred.to(x.dtype).expand_as(x).clone()


def test_q_sample_endpoints():
    x0 = torch.randn(8, 3, 4, 4)
    eps = torch.randn_like(x0)
    xt0, _ = D.q_sample(x0, torch.zeros(8), noise=eps)
    xt1, _ = D.q_sample(x0, torch.ones(8), noise=eps)
    assert torch.allclose(xt0, eps), "t=0 must be pure noise"
    assert torch.allclose(xt1, x0), "t=1 must be clean data"
    print("  q_sample endpoints            OK")


def test_noise_scale():
    x0 = torch.zeros(4096, 4, 8, 8)
    _, noise = D.q_sample(x0, torch.zeros(4096), noise_scale=2.0)
    s = noise.std().item()
    assert abs(s - 2.0) < 0.02, f"noise std {s} != 2.0"
    print(f"  noise_scale=2.0 -> std {s:.4f}  OK")


def test_v_space_matches_hand_computed():
    """loss == mean( ((pred-x_t)/den - (x0-x_t)/den)^2 ), den = max(1-t, 0.05)."""
    torch.manual_seed(0)
    x0 = torch.randn(3, 2, 4, 4)
    eps = torch.randn_like(x0)
    pred = torch.randn(1, 1, 1, 1)
    model = Const(pred)
    ctx, mask = torch.randn(3, 5, 8), torch.ones(3, 5)

    for tval in (0.05, 0.5, 0.99):
        t = torch.full((3,), tval)
        loss, aux = D.training_loss(
            model, x0, ctx, mask, label_drop_rate=0.0, noise_scale=2.0,
            t=t, noise=eps * 2.0, recon_loss="v_mse",
        )
        x_t = x0 * tval + eps * 2.0 * (1 - tval)
        den = max(1.0 - tval, 0.05)
        p = pred.expand_as(x0)
        expect = (((p - x_t) / den) - ((x0 - x_t) / den)).pow(2).mean()
        assert torch.allclose(loss, expect, atol=1e-6), f"t={tval}: {loss} != {expect}"
        print(f"  t={tval:<5} v-loss matches hand-computed   OK  (den={den:.3f})")


def test_clamp_saturates():
    """At t=0.99 the true 1-t is 0.01, but the clamp must hold den at 0.05."""
    x0 = torch.randn(2, 2, 3, 3); eps = torch.randn_like(x0)
    model = Const(torch.zeros(1, 1, 1, 1))
    ctx, mask = torch.randn(2, 4, 8), torch.ones(2, 4)
    t = torch.full((2,), 0.99)
    loss, _ = D.training_loss(model, x0, ctx, mask, label_drop_rate=0.0,
                              t=t, noise=eps, noise_scale=1.0)
    x_t = x0 * 0.99 + eps * 0.01
    unclamped = ((0 - x_t) / 0.01 - (x0 - x_t) / 0.01).pow(2).mean()
    clamped = ((0 - x_t) / 0.05 - (x0 - x_t) / 0.05).pow(2).mean()
    assert torch.allclose(loss, clamped, atol=1e-6)
    assert not torch.allclose(loss, unclamped, atol=1e-3)
    ratio = (unclamped / clamped).item()
    assert abs(ratio - 25.0) < 0.1, ratio
    print(f"  clamp_min(0.05) saturates      OK  (unclamped would be {ratio:.1f}x)")


def test_v_loss_is_weighted_x0_loss():
    """Algebraic identity: v_mse == x0_mse / den^2. This is the whole point."""
    torch.manual_seed(1)
    x0 = torch.randn(4, 3, 4, 4); eps = torch.randn_like(x0)
    model = Const(torch.randn(1, 1, 1, 1))
    ctx, mask = torch.randn(4, 6, 8), torch.ones(4, 6)
    for tval in (0.1, 0.6, 0.95):
        t = torch.full((4,), tval)
        kw = dict(label_drop_rate=0.0, t=t, noise=eps, noise_scale=1.0)
        v, _ = D.training_loss(model, x0, ctx, mask, recon_loss="v_mse", **kw)
        m, _ = D.training_loss(model, x0, ctx, mask, recon_loss="x0_mse", **kw)
        den = max(1.0 - tval, 0.05)
        assert torch.allclose(v, m / den ** 2, rtol=1e-5), f"t={tval}"
        print(f"  t={tval:<5} v_mse == x0_mse/den^2         OK  (weight {1/den**2:6.1f}x)")


def test_cfg_dropout():
    mask = torch.ones(1000, 12)
    assert D.drop_mask_for_cfg(mask, 0.0).sum() == mask.sum()
    assert D.drop_mask_for_cfg(mask, 1.0).sum() == 0
    frac = 1.0 - D.drop_mask_for_cfg(mask, 0.1).any(1).float().mean().item()
    assert 0.06 < frac < 0.15, frac
    print(f"  CFG mask dropout               OK  (dropped {frac:.1%} at rate 0.1)")


def test_euler_reaches_t1():
    """Euler is EXACT for a perfect x0-oracle -- unless the denominator is clamped.

    e_{k+1} = e_k * (1 - h/(1-t_k)); unclamped the factors telescope to 0/1 = 0.
    MiniT2I's clamp_min(0.05) breaks that for the last 5 steps, leaving
    0.05 * 0.8^5 = 1.6384% of the initial error. Both regimes are asserted so a
    future change to either clamp cannot pass silently.
    """
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4, 4)
    class Oracle(torch.nn.Module):
        def forward(self, x, t, ctx, mask): return target[: x.shape[0]].clone()
    ctx, mask = torch.randn(2, 5, 8), torch.ones(2, 5)

    exact = D.euler_sample(Oracle(), ctx, mask, (3, 4, 4), steps=100,
                           cfg_scale=1.0, noise_scale=2.0, v_clamp_min=0.0,
                           exact_final=False)
    err = (exact - target).abs().max().item()
    assert err < 1e-5, err
    print(f"  euler exact (clamp 0)          OK  (max err {err:.2e})")

    # exact_final=False reproduces MiniT2I: the clamp is integrated through and
    # leaves a residual. Pinned so a change to either clamp cannot pass silently.
    torch.manual_seed(0)
    clamped = D.euler_sample(Oracle(), ctx, mask, (3, 4, 4), steps=100,
                             cfg_scale=1.0, noise_scale=2.0, v_clamp_min=0.05,
                             exact_final=False)
    torch.manual_seed(0)
    x0_noise = torch.randn(2, 3, 4, 4) * 2.0
    resid = (clamped - target) / (x0_noise - target)
    assert torch.allclose(resid, torch.full_like(resid, 0.016384), atol=1e-4), resid.flatten()[:3]
    print(f"  euler residual (clamp 0.05)    OK  (1.6384% of initial error, as predicted)")

    # exact_final=True (the default) kills that residual EXACTLY, at any step
    # count, while leaving the clamp in place for the rest of the trajectory.
    for steps in (25, 50, 100, 200):
        torch.manual_seed(0)
        fixed = D.euler_sample(Oracle(), ctx, mask, (3, 4, 4), steps=steps,
                               cfg_scale=1.0, noise_scale=2.0, v_clamp_min=0.05,
                               exact_final=True)
        e = (fixed - target).abs().max().item()
        assert e < 1e-6, (steps, e)
    print(f"  euler exact_final (clamp 0.05) OK  (zero residual at 25/50/100/200 steps)")


if __name__ == "__main__":
    for fn in [test_q_sample_endpoints, test_noise_scale,
               test_v_space_matches_hand_computed, test_clamp_saturates,
               test_v_loss_is_weighted_x0_loss, test_cfg_dropout,
               test_euler_reaches_t1]:
        print(f"{fn.__name__}:")
        fn()
    print("\nAll diffusion tests passed.")
