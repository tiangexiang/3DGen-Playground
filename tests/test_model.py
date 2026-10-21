"""Model structure checks. Deliberately light: runs on a login node.

Logic is exercised on a TINY config; full-size configs are only ever built one
at a time to count parameters (never forwarded), because a login node will
OOM-kill several 300M models held simultaneously.
"""
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini3d.config import TrainConfig, num_tokens  # noqa: E402
from mini3d.model import build_model  # noqa: E402


def tiny(**kw) -> TrainConfig:
    c = TrainConfig()
    c.hidden_size = c.txt_hidden_size = c.cond_vec_size = 64
    c.depth_double, c.text_preamble_depth = 2, 1
    c.num_heads, c.head_dim = 2, 32
    c.pca_channels, c.group_size = 16, 1024   # 16384/1024 = 16 tokens
    c.prompt_length = 8
    for k, v in kw.items():
        setattr(c, k, v)
    return c.validate()


def test_shapes_all_time_conds():
    B = 2
    for tc in ("none", "adaln_img", "adaln_both"):
        c = tiny(time_cond=tc)
        m = build_model(c).eval()
        img = torch.randn(B, c.in_channels, c.container_size, c.container_size)
        out = m(img, torch.rand(B), torch.randn(B, c.prompt_length, c.txt_input_size),
                torch.ones(B, c.prompt_length))
        assert out.shape == img.shape, (out.shape, img.shape)
        assert m.num_tokens == num_tokens(c) == 16
        print(f"  time_cond={tc:<11s} out={tuple(out.shape)} tokens={m.num_tokens}  OK")
        del m; gc.collect()


def test_plane2d_path():
    c = tiny(token_order="plane2d", patch_size=32)   # 128/32 = 4 -> 16 tokens
    m = build_model(c).eval()
    img = torch.randn(2, c.in_channels, 128, 128)
    out = m(img, torch.rand(2), torch.randn(2, 8, 1024), torch.ones(2, 8))
    assert out.shape == img.shape and m.num_tokens == 16
    print(f"  plane2d path  out={tuple(out.shape)} tokens={m.num_tokens}  OK")


def test_tokenizer_roundtrip_is_order_preserving():
    """unpatchify_1d must invert the row-major flatten the tokenizer assumes.

    If these disagree the model still trains -- it just silently learns a
    permuted container, and every render downstream is scrambled.
    """
    c = tiny()
    m = build_model(c).eval()
    B, C, S = 2, c.in_channels, c.container_size
    x = torch.arange(B * C * S * S, dtype=torch.float32).reshape(B, C, S, S)
    flat = x.reshape(B, C, S * S)                       # what BottleneckChunkEmbed sees
    tokens = flat.transpose(1, 2).reshape(B, m.num_tokens, c.group_size * C)
    assert torch.equal(m.unpatchify_1d(tokens), x), "unpatchify_1d is not the inverse flatten"
    print("  unpatchify_1d inverts flatten  OK")


def _trunk(m, img, t, ctx, msk, use_vec):
    """Run everything up to (not including) the zero-init final layer."""
    with torch.no_grad():
        c = torch.where(msk.bool()[:, :, None], ctx, m.mask_token)
        x, txt = m.img_embed(img) + m.pos_embed, m.txt_embed(c)
        for blk in m.txt_blocks:
            txt = blk(txt)
        vec = (m.t_embed(t) + m.pooled_embed(c.mean(1))) if use_vec else None
        for blk in m.blocks:
            x, txt = blk(x, txt, vec)
    return x


def test_adaln_gate_one_starts_equal_to_none():
    """gate_init='one' must reproduce a MiniT2I block EXACTLY at init.

    Not a formality: with gate_init='zero' (classic AdaLN-Zero) the block is an
    IDENTITY at init, which is a different model from MiniT2I -- it has no gates
    and passes residuals at full strength. Both behaviours are asserted so the
    none-vs-adaln ablation cannot silently compare two different starting points.
    """
    a = build_model(tiny(time_cond="adaln_img", adaln_gate_init="one")).eval()
    b = build_model(tiny(time_cond="none")).eval()
    # A shared seed is NOT enough: the adaln model builds extra modules, which
    # consume RNG draws and shift every later init. Copy the shared weights
    # explicitly; a's adaLN params stay at their zero init, which is the point.
    missing, unexpected = a.load_state_dict(b.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(("adaLN" in k) or ("t_embed" in k) or ("pooled_embed" in k) for k in missing), missing
    B, c = 2, tiny()
    img = torch.randn(B, c.in_channels, 128, 128)
    t, ctx, msk = torch.rand(B), torch.randn(B, 8, 1024), torch.ones(B, 8)
    d = (_trunk(a, img, t, ctx, msk, True) - _trunk(b, img, t, ctx, msk, False)).abs().max().item()
    assert d < 1e-5, d
    print(f"  gate=one  == none at init      OK  (max|diff| {d:.2e})")

    z = build_model(tiny(time_cond="adaln_img", adaln_gate_init="zero")).eval()
    z.load_state_dict(b.state_dict(), strict=False)
    xz = _trunk(z, img, t, ctx, msk, True)
    x_in = z.img_embed(img) + z.pos_embed
    assert torch.allclose(xz, x_in, atol=1e-5), (xz - x_in).abs().max()
    print("  gate=zero == identity at init  OK  (confirms the two differ)")


def test_uncond_branch_is_caption_independent():
    """An all-zero mask must erase the caption completely -- that IS the CFG null."""
    c = tiny(); m = build_model(c).eval()
    img, t = torch.randn(2, c.in_channels, 128, 128), torch.rand(2)
    z = torch.zeros(2, c.prompt_length)
    with torch.no_grad():
        o1 = m(img, t, torch.randn(2, c.prompt_length, 1024), z)
        o2 = m(img, t, torch.randn(2, c.prompt_length, 1024), z)
    assert torch.allclose(o1, o2, atol=1e-6), (o1 - o2).abs().max()
    print("  uncond ignores caption         OK")


def test_padding_is_masked_out():
    """Changing PADDED positions must not change the output."""
    c = tiny(); m = build_model(c).eval()
    img, t = torch.randn(2, c.in_channels, 128, 128), torch.rand(2)
    msk = torch.ones(2, c.prompt_length); msk[:, 4:] = 0
    base = torch.randn(2, c.prompt_length, 1024)
    alt = base.clone(); alt[:, 4:] = torch.randn(2, c.prompt_length - 4, 1024)
    with torch.no_grad():
        assert torch.allclose(m(img, t, base, msk), m(img, t, alt, msk), atol=1e-6)
    print("  padded positions ignored       OK")


def test_no_unused_parameters():
    """Every parameter must receive a gradient -- DDP treats otherwise as a hard error.

    The last double-stream block's text OUTPUT path (txt_proj / txt_norm2 /
    txt_mlp) is dead by construction: the final layer reads the image stream
    only. MiniT2I hides this with find_unused_parameters=True; we remove the path
    instead, which is output-identical (see test_skip_txt_output_is_equivalent).
    This test is what stops it silently coming back.
    """
    from mini3d import diffusion
    for tc in ("adaln_img", "adaln_both", "none"):
        c = tiny(time_cond=tc)
        m = build_model(c); m.train()
        x0 = torch.randn(2, c.in_channels, 128, 128)
        ctx = torch.randn(2, c.prompt_length, c.txt_input_size)
        msk = torch.ones(2, c.prompt_length, dtype=torch.bool)
        loss, _ = diffusion.training_loss(m, x0, ctx, msk, label_drop_rate=0.0)
        loss.backward()
        unused = [n for n, p in m.named_parameters() if p.grad is None]
        assert not unused, f"time_cond={tc}: {unused}"
        print(f"  time_cond={tc:<11s} 0 unused params          OK")
        del m; gc.collect()


def test_skip_txt_output_is_equivalent():
    """Removing the last block's dead text path must not change the output at all."""
    import mini3d.model as M
    for tc in ("adaln_img", "none"):
        c = tiny(time_cond=tc, final_layer_zero=False)
        new = build_model(c).eval()
        old = build_model(c).eval()
        blk = old.blocks[-1]
        blk.skip_txt_output = False
        blk.txt_norm2 = M.RMSNorm(c.hidden_size)
        blk.txt_proj = torch.nn.Linear(c.num_heads * c.head_dim, c.hidden_size)
        blk.txt_mlp = M.SwiGLU(c.hidden_size, int(c.hidden_size * c.mlp_ratio))
        old.load_state_dict(new.state_dict(), strict=False)
        x, t = torch.randn(2, c.in_channels, 128, 128), torch.rand(2)
        ctx = torch.randn(2, c.prompt_length, c.txt_input_size)
        msk = torch.ones(2, c.prompt_length)
        with torch.no_grad():
            d = (new(x, t, ctx, msk) - old(x, t, ctx, msk)).abs().max().item()
        assert d == 0.0, f"time_cond={tc}: max|diff| = {d}"
        print(f"  time_cond={tc:<11s} removal is output-identical  OK")
        del new, old; gc.collect()


def test_full_size_param_counts():
    """One at a time, no forward -- a login node cannot hold several of these."""
    for tc in ("none", "adaln_img", "adaln_both"):
        c = TrainConfig(); c.time_cond = tc; c.validate()
        m = build_model(c)
        n = sum(p.numel() for p in m.parameters())
        print(f"  full-size time_cond={tc:<11s} {n / 1e6:7.1f}M params")
        del m; gc.collect()


if __name__ == "__main__":
    for fn in [test_shapes_all_time_conds, test_plane2d_path,
               test_tokenizer_roundtrip_is_order_preserving,
               test_adaln_gate_one_starts_equal_to_none,
               test_uncond_branch_is_caption_independent,
               test_padding_is_masked_out, test_no_unused_parameters,
               test_skip_txt_output_is_equivalent, test_full_size_param_counts]:
        print(f"{fn.__name__}:")
        fn()
    print("\nAll model tests passed.")
