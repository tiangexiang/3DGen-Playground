"""Multimodal diffusion transformer for Gaussian splats, adapted from MiniT2I.

Inputs are (B, C, H, W) containers and frozen text-token embeddings. Hilbert
ordering maps contiguous groups of splats to tokens; the optional plane layout
uses 2D patches. Joint attention updates image and text streams, while AdaLN
conditions the image stream on time and pooled text. A learned mask token
provides unconditional classifier-free guidance.

The model module has no data-loader or renderer dependencies."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Primitives (verbatim from mini_t2i/model.py)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y if self.weight is None else y * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = math.ceil(hidden_dim / 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w3.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, 256).to(dtype=self.mlp[0].weight.dtype))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_1d_rope(x: torch.Tensor, start: int = 0, theta: int = 10000) -> torch.Tensor:
    """1D RoPE over ``(b, h, n, d)``, positions ``start .. start+n-1``.

    ``start`` is how the two streams are kept in disjoint position spaces: text
    occupies ``[0, L)`` and image ``[L, L+T)``. MiniT2I never needed this because
    its image stream used 2D RoPE (a different frequency layout, so no collision
    with the text positions). Ours is 1D on both sides, so without the offset
    text token i and image token i would receive an identical rotation and the
    attention would see a spurious positional alignment between caption position
    and splat-chunk index.
    """
    b, h, n, d = x.shape
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    pos = torch.arange(start, start + n, device=x.device, dtype=torch.float32)
    angles = torch.einsum("n,f->nf", pos, inv)
    angles = torch.cat([angles, angles], dim=-1)
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def apply_2d_rope_flat(x: torch.Tensor, grid: int, theta: int = 10000) -> torch.Tensor:
    """2D RoPE for the ``token_order="plane2d"`` path (verbatim from MiniT2I)."""
    b, h, n, d = x.shape
    if n != grid * grid:
        return apply_1d_rope(x, theta=theta)
    rope_dim = d // 2
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device=x.device, dtype=torch.float32) / rope_dim))
    t = torch.arange(grid, device=x.device, dtype=torch.float32)
    freqs = torch.einsum("n,f->nf", t, inv)
    f_h, f_w = torch.broadcast_tensors(freqs[:, None, :], freqs[None, :, :])
    angles = torch.cat([f_h, f_w], dim=-1)
    angles = torch.cat([angles, angles], dim=-1).reshape(n, d)
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    return x * cos + rotate_half(x) * sin


def attention_forward(q, k, v, head_dim: int, impl: str) -> torch.Tensor:
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False,
                                              scale=head_dim ** -0.5)
    if impl == "einsum":
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * (head_dim ** -0.5)
        return torch.einsum("bhqk,bhkd->bhqd", torch.softmax(scores, dim=-1), v)
    raise ValueError(f"unknown attention_impl={impl!r}")


def sincos_1d(embed_dim: int, length: int) -> torch.Tensor:
    """Sinusoidal embedding over ``length`` sequence positions."""
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = torch.einsum("n,d->nd", pos, omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_2d(embed_dim: int, grid: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / (10000 ** omega)
    out_y = torch.einsum("n,d->nd", y.flatten().float(), omega)
    out_x = torch.einsum("n,d->nd", x.flatten().float(), omega)
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class BottleneckChunkEmbed(nn.Module):
    """1D analogue of MiniT2I's ``BottleneckPatchEmbed``.

    One token per ``group_size`` consecutive splats along the Hilbert curve,
    through a low-rank ``pca_channels`` bottleneck. The bottleneck is JiT's
    trick: it forces the tokenizer to be a near-linear low-rank projection of raw
    data rather than a wide learned encoder, which is what keeps the model
    "denoising" rather than quietly learning a latent space.
    """

    def __init__(self, num_points: int, group_size: int, in_channels: int,
                 pca_channels: int, hidden_size: int):
        super().__init__()
        if num_points % group_size != 0:
            raise ValueError(f"num_points={num_points} not divisible by group_size={group_size}")
        self.num_points = num_points
        self.group_size = group_size
        self.num_tokens = num_points // group_size
        self.proj1 = nn.Conv1d(in_channels, pca_channels, kernel_size=group_size,
                               stride=group_size, bias=False)
        self.proj2 = nn.Conv1d(pca_channels, hidden_size, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj1.weight.view(self.proj1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.proj2.weight.view(self.proj2.weight.shape[0], -1))
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h * w != self.num_points:
            raise ValueError(f"container has {h * w} points, expected {self.num_points}")
        x = x.reshape(b, c, h * w)          # row-major == Hilbert sequence
        return self.proj2(self.proj1(x)).transpose(1, 2)


class BottleneckPatchEmbed(nn.Module):
    """MiniT2I's Conv2d tokenizer, kept for ``token_order="plane2d"``."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int,
                 pca_channels: int, hidden_size: int):
        super().__init__()
        self.grid = image_size // patch_size
        self.num_tokens = self.grid * self.grid
        self.proj1 = nn.Conv2d(in_channels, pca_channels, kernel_size=patch_size,
                               stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_channels, hidden_size, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj1.weight.view(self.proj1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.proj2.weight.view(self.proj2.weight.shape[0], -1))
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class PlainTextBlock(nn.Module):
    """Text preamble block: self-attention over caption tokens only."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float,
                 qk_norm: bool = True, attention_impl: str = "sdpa"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.attention_impl = attention_impl
        inner = num_heads * head_dim
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, inner * 3)
        self.proj = nn.Linear(inner, hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, txt: torch.Tensor) -> torch.Tensor:
        b, n, _ = txt.shape
        qkv = self.qkv(self.norm1(txt)).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        q, k = apply_1d_rope(q), apply_1d_rope(k)
        out = attention_forward(q, k, v, self.head_dim, self.attention_impl)
        txt = txt + self.proj(out.transpose(1, 2).reshape(b, n, -1))
        return txt + self.mlp(self.norm2(txt))


class DoubleStreamBlock(nn.Module):
    """Joint attention over [text; image] tokens with separate stream weights."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float,
                 image_tokens: int, text_tokens: int, token_order: str = "hilbert1d",
                 image_grid: int = 0, qk_norm: bool = True, attention_impl: str = "sdpa",
                 time_cond: str = "adaln_img", adaln_gate_init: str = "one",
                 skip_txt_output: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.attention_impl = attention_impl
        self.token_order = token_order
        self.image_grid = image_grid
        self.text_tokens = text_tokens
        self.time_cond = time_cond
        # Skip unused text output weights to avoid parameters without gradients in DDP.
        # Text Q/K/V remain available to image attention.
        self.skip_txt_output = bool(skip_txt_output)
        inner = num_heads * head_dim

        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv = nn.Linear(hidden_size, inner * 3)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        self.img_proj = nn.Linear(inner, hidden_size)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        if not self.skip_txt_output:
            self.txt_norm2 = RMSNorm(hidden_size)
            self.txt_proj = nn.Linear(inner, hidden_size)
            self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

        # AdaLN shift/scale start at zero. Gate initialization "one" preserves the
        # unmodulated block; "zero" makes its residual branches initially inactive.
        self.img_adaLN = None
        self.txt_adaLN = None
        self.img_adaLN_offset = None
        if time_cond in ("adaln_img", "adaln_both"):
            self.img_adaLN = self._make_adaLN(hidden_size, adaln_gate_init)
        if time_cond == "adaln_both" and not self.skip_txt_output:
            self.txt_adaLN = self._make_adaLN(hidden_size, adaln_gate_init)
        if time_cond == "adaln_shared":
            # The parent owns the shared AdaLN projection; each block learns an offset.
            # Zero offsets preserve the shared initialization.
            self.img_adaLN_offset = nn.Parameter(torch.zeros(6 * hidden_size))

    @staticmethod
    def _make_adaLN(hidden_size: int, gate_init: str) -> nn.Sequential:
        mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(mod[-1].weight)
        nn.init.zeros_(mod[-1].bias)
        if gate_init == "one":
            # chunk order is (shift1, scale1, gate1, shift2, scale2, gate2)
            with torch.no_grad():
                h = hidden_size
                mod[-1].bias[2 * h:3 * h] = 1.0
                mod[-1].bias[5 * h:6 * h] = 1.0
        elif gate_init != "zero":
            raise ValueError(f"adaln_gate_init must be one|zero, got {gate_init!r}")
        return mod

    def _rope_img(self, x: torch.Tensor) -> torch.Tensor:
        if self.token_order == "plane2d":
            return apply_2d_rope_flat(x, self.image_grid)
        return apply_1d_rope(x, start=self.text_tokens)

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor | None,
                shared_mod: torch.Tensor | None = None):
        b, li, _ = img.shape
        lt = txt.shape[1]

        if self.img_adaLN is not None:
            i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = self.img_adaLN(vec).chunk(6, dim=1)
            img_in = modulate(self.img_norm1(img), i_sh1, i_sc1)
        elif self.img_adaLN_offset is not None:
            mod = shared_mod + self.img_adaLN_offset.to(dtype=shared_mod.dtype)
            i_sh1, i_sc1, i_g1, i_sh2, i_sc2, i_g2 = mod.chunk(6, dim=1)
            img_in = modulate(self.img_norm1(img), i_sh1, i_sc1)
        else:
            i_g1 = i_g2 = None
            img_in = self.img_norm1(img)
        if self.txt_adaLN is not None:
            t_sh1, t_sc1, t_g1, t_sh2, t_sc2, t_g2 = self.txt_adaLN(vec).chunk(6, dim=1)
            txt_in = modulate(self.txt_norm1(txt), t_sh1, t_sc1)
        else:
            t_g1 = t_g2 = None
            txt_in = self.txt_norm1(txt)

        qi, ki, vi = self.img_qkv(img_in).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(txt_in).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        if self.qk_norm:
            qi, ki = self.q_norm(qi), self.k_norm(ki)
            qt, kt = self.q_norm(qt), self.k_norm(kt)
        qi, ki, vi = (z.transpose(1, 2) for z in (qi, ki, vi))
        qt, kt, vt = (z.transpose(1, 2) for z in (qt, kt, vt))

        q = torch.cat([apply_1d_rope(qt), self._rope_img(qi)], dim=2)
        k = torch.cat([apply_1d_rope(kt), self._rope_img(ki)], dim=2)
        v = torch.cat([vt, vi], dim=2)

        out = attention_forward(q, k, v, self.head_dim, self.attention_impl).transpose(1, 2)
        att_i = self.img_proj(out[:, lt:].reshape(b, li, -1))
        img = img + (att_i if i_g1 is None else i_g1.unsqueeze(1) * att_i)
        img_m = self.img_mlp(self.img_norm2(img) if i_g2 is None
                             else modulate(self.img_norm2(img), i_sh2, i_sc2))
        img = img + (img_m if i_g2 is None else i_g2.unsqueeze(1) * img_m)

        if not self.skip_txt_output:
            att_t = self.txt_proj(out[:, :lt].reshape(b, lt, -1))
            txt = txt + (att_t if t_g1 is None else t_g1.unsqueeze(1) * att_t)
            txt_m = self.txt_mlp(self.txt_norm2(txt) if t_g2 is None
                                 else modulate(self.txt_norm2(txt), t_sh2, t_sc2))
            txt = txt + (txt_m if t_g2 is None else t_g2.unsqueeze(1) * txt_m)
        return img, txt


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_features: int, time_cond: str, zero_init: bool = True):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_features)
        self.adaLN = None
        if time_cond in ("adaln_img", "adaln_both", "adaln_shared"):
            # Only ONE FinalLayer exists, so there is nothing to share here; it
            # keeps its own 2*hidden map (1.18M) under adaln_shared too.
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
            nn.init.zeros_(self.adaLN[-1].weight)
            nn.init.zeros_(self.adaLN[-1].bias)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, vec: torch.Tensor | None) -> torch.Tensor:
        x = self.norm_final(x)
        if self.adaLN is not None:
            shift, scale = self.adaLN(vec).chunk(2, dim=1)
            x = modulate(x, shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class MMJiT3D(nn.Module):
    def __init__(
        self,
        container_size: int = 128,
        in_channels: int = 14,
        token_order: str = "hilbert1d",
        group_size: int = 64,
        patch_size: int = 16,
        hidden_size: int = 768,
        txt_input_size: int = 1024,
        prompt_length: int = 64,
        depth_double: int = 17,
        text_preamble_depth: int = 2,
        txt_refine_depth: int | None = None,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        final_layer_zero: bool = True,
        attention_impl: str = "sdpa",
        time_cond: str = "adaln_img",
        adaln_gate_init: str = "one",
        pooled_in_vec: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.container_size = container_size
        self.in_channels = in_channels
        self.token_order = token_order
        self.group_size = group_size
        self.patch_size = patch_size
        self.prompt_length = prompt_length
        self.time_cond = time_cond
        self.pooled_in_vec = pooled_in_vec
        self.gradient_checkpointing = gradient_checkpointing
        self.uses_vec = time_cond in ("adaln_img", "adaln_both", "adaln_shared")

        num_points = container_size * container_size
        if token_order == "hilbert1d":
            self.img_embed = BottleneckChunkEmbed(num_points, group_size, in_channels,
                                                  pca_channels, hidden_size)
            self.image_grid = 0
            out_features = group_size * in_channels
            pos = sincos_1d(hidden_size, self.img_embed.num_tokens)
        elif token_order == "plane2d":
            self.img_embed = BottleneckPatchEmbed(container_size, patch_size, in_channels,
                                                  pca_channels, hidden_size)
            self.image_grid = self.img_embed.grid
            out_features = patch_size * patch_size * in_channels
            pos = sincos_2d(hidden_size, self.image_grid)
        else:
            raise ValueError(f"token_order must be hilbert1d|plane2d, got {token_order!r}")
        self.num_tokens = self.img_embed.num_tokens
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)

        self.txt_embed = nn.Linear(txt_input_size, hidden_size, bias=False)
        # Replace masked text positions with a learned token.
        # An all-zero mask selects unconditional conditioning for CFG.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, txt_input_size))
        nn.init.normal_(self.mask_token, std=0.02)

        if self.uses_vec:
            self.t_embed = TimestepEmbedder(hidden_size)
            if pooled_in_vec:
                self.pooled_embed = nn.Linear(txt_input_size, hidden_size, bias=False)
        self.shared_adaLN = None
        if time_cond == "adaln_shared":
            self.shared_adaLN = DoubleStreamBlock._make_adaLN(hidden_size, adaln_gate_init)

        self.txt_blocks = nn.ModuleList([
            PlainTextBlock(hidden_size, num_heads, head_dim, mlp_ratio,
                           attention_impl=attention_impl)
            for _ in range(text_preamble_depth)
        ])
        # Only the first txt_refine_depth joint blocks update the text stream;
        # later blocks still provide text K/V to image attention. None skips only
        # the final block's unused text output. The text preamble always runs.
        _txt_refine = depth_double - 1 if txt_refine_depth is None else int(txt_refine_depth)
        if not 0 <= _txt_refine <= depth_double:
            raise ValueError(f"txt_refine_depth must be in [0, {depth_double}], got {_txt_refine}")
        self.txt_refine_depth = _txt_refine
        self.blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, head_dim, mlp_ratio,
                              image_tokens=self.num_tokens, text_tokens=prompt_length,
                              token_order=token_order, image_grid=self.image_grid,
                              attention_impl=attention_impl, time_cond=time_cond,
                              adaln_gate_init=adaln_gate_init,
                              skip_txt_output=(i >= _txt_refine))
            for i in range(depth_double)
        ])
        self.final = FinalLayer(hidden_size, out_features, time_cond, zero_init=final_layer_zero)

    # -- detokenize ---------------------------------------------------------
    def unpatchify_1d(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        c, g = self.in_channels, self.group_size
        x = x.reshape(b, t * g, c)                 # (B, num_points, C) in sequence order
        side = self.container_size
        return x.transpose(1, 2).reshape(b, c, side, side)

    def unpatchify_2d(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        p, c = self.patch_size, self.in_channels
        g = int(n ** 0.5)
        x = x.view(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    # -- forward ------------------------------------------------------------
    def forward(self, img: torch.Tensor, t: torch.Tensor, context: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        """img (B,C,H,W); t (B,) in [0,1]; context (B,L,txt_input_size); attn_mask (B,L)."""
        mask = attn_mask.to(device=context.device, dtype=torch.bool)[:, :, None]
        context = torch.where(mask, context, self.mask_token.to(dtype=context.dtype))

        x = self.img_embed(img) + self.pos_embed.to(device=img.device, dtype=img.dtype)
        txt = self.txt_embed(context.to(dtype=self.txt_embed.weight.dtype))

        vec = None
        if self.uses_vec:
            vec = self.t_embed(t).to(dtype=x.dtype)
            if self.pooled_in_vec:
                # Plain mean over the POST-replacement context, exactly as MiniT2I.
                # On the unconditional branch every position is mask_token, so the
                # pooled vector is a well-defined learned constant -- no separate
                # null embedding, and no all-masked mean to divide by zero.
                pooled = context.mean(dim=1)
                vec = vec + self.pooled_embed(pooled.to(dtype=self.pooled_embed.weight.dtype)).to(dtype=x.dtype)

        for block in self.txt_blocks:
            txt = block(txt)
        shared_mod = self.shared_adaLN(vec) if self.shared_adaLN is not None else None
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, txt = checkpoint(block, x, txt, vec, shared_mod, use_reentrant=False)
            else:
                x, txt = block(x, txt, vec, shared_mod)

        out = self.final(x, vec)
        out = self.unpatchify_1d(out) if self.token_order == "hilbert1d" else self.unpatchify_2d(out)
        return out.float()


def build_model(cfg) -> MMJiT3D:
    """Construct from a TrainConfig."""
    return MMJiT3D(
        container_size=cfg.container_size,
        in_channels=cfg.in_channels,
        token_order=cfg.token_order,
        group_size=cfg.group_size,
        patch_size=cfg.patch_size,
        hidden_size=cfg.hidden_size,
        txt_input_size=cfg.txt_input_size,
        prompt_length=cfg.prompt_length,
        depth_double=cfg.depth_double,
        text_preamble_depth=cfg.text_preamble_depth,
        txt_refine_depth=cfg.txt_refine_depth,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        mlp_ratio=cfg.mlp_ratio,
        pca_channels=cfg.pca_channels,
        final_layer_zero=cfg.final_layer_zero,
        attention_impl=cfg.attention_impl,
        time_cond=cfg.time_cond,
        adaln_gate_init=cfg.adaln_gate_init,
        pooled_in_vec=cfg.pooled_in_vec,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )
