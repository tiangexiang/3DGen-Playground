# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
# Adapted for 3DGS generation (128x128 grid, 59 channels, 48 classes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class VisionRotaryEmbeddingFast(nn.Module):
    """2D RoPE over flattened square patch tokens."""

    def __init__(self, dim, pt_seq_len, ft_seq_len=None, theta=10000):
        super().__init__()
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
        freqs_1d = torch.einsum("n,d->nd", positions, freqs)
        freqs_1d = torch.repeat_interleave(freqs_1d, repeats=2, dim=-1)

        freqs_h = freqs_1d[:, None, :].expand(ft_seq_len, ft_seq_len, -1)
        freqs_w = freqs_1d[None, :, :].expand(ft_seq_len, ft_seq_len, -1)
        freqs_2d = torch.cat((freqs_h, freqs_w), dim=-1).reshape(ft_seq_len * ft_seq_len, -1)

        self.register_buffer("freqs_cos", freqs_2d.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs_2d.sin(), persistent=False)

    def forward(self, x):
        # Cache the dtype-converted buffers so we don't pay the .to() overhead
        # on every forward pass during stable mixed-precision training.
        if not hasattr(self, '_rope_cache_dtype') or self._rope_cache_dtype != x.dtype:
            self._cos_cache = self.freqs_cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            self._sin_cache = self.freqs_sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
            self._rope_cache_dtype = x.dtype
        return x * self._cos_cache + rotate_half(x) * self._sin_cache


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states * self.weight).to(input_dtype)


def scaled_dot_product_attention(query, key, value, dropout_p=0.0):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=dropout_p,
    )


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        x = scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0, bias=True):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.dropout(hidden))


class BottleneckPatchEmbed(nn.Module):
    """Patchify directly from the full-resolution grid via a low-rank bottleneck."""

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_chans=59,
        bottleneck_dim=128,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        grid_size = img_size // patch_size
        self.num_patches = grid_size * grid_size

        self.proj1 = nn.Conv2d(
            in_chans,
            bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.proj2 = nn.Conv2d(
            bottleneck_dim,
            embed_dim,
            kernel_size=1,
            stride=1,
            bias=bias,
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(
                f"Input size ({h}x{w}) does not match model patch embed size "
                f"({self.img_size[0]}x{self.img_size[1]})."
            )
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        # Pre-compute the frequency vector once; register as non-persistent buffer
        # so it moves with the model (device-aware) without appearing in state_dict.
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )
        self.register_buffer("_freqs", freqs, persistent=False)

    def timestep_embedding(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices (possibly fractional), one per batch element.
        :return: an (N, frequency_embedding_size) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        args = t[:, None].float() * self._freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core JiT Model                                #
#################################################################################

class JiTBlock(nn.Module):
    """A JiT block with adaLN-Zero, RMSNorm, qk-norm, RoPE, and SwiGLU."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final JiT output projection."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """JiT-style diffusion backbone for 3DGS feature grids."""

    def __init__(
        self,
        input_size=128,
        patch_size=16,
        in_channels=59,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=48,
        learn_sigma=False,
        gradient_checkpointing=True,
        bottleneck_dim=128,
        attn_drop=0.0,
        proj_drop=0.0,
        aux_classifier=False,
        label_embed_init_std=0.02,
    ):
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch_size}")

        self.learn_sigma = learn_sigma
        self.input_size = input_size
        self.sample_size = input_size
        # JiT operates directly on the full 128x128 latent grid. Patchification is tokenization only,
        # not UNet/DiT-style spatial folding.
        self.spatial_fold_factor = 1
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                f"Per-head hidden size must be divisible by 4 for 2D RoPE, got hidden_size={hidden_size}, num_heads={num_heads}"
            )

        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=self.head_dim // 2,
            pt_seq_len=input_size // patch_size,
        )

        self.blocks = nn.ModuleList([
            JiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.aux_classifier = nn.Linear(hidden_size, num_classes) if aux_classifier else None
        self._label_embed_init_std = float(label_embed_init_std)
        # Stash the pooled aux logits during forward so the training loop can
        # retrieve them without changing the forward() return signature (which
        # would break flow_matching_training_losses's shape assertion).
        self._aux_logits = None
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embed like a pair of linear projections.
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=self._label_embed_init_std)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in JiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module, rope):
        def ckpt_forward(x, c):
            return module(x, c, rope=rope)
        return ckpt_forward

    def forward(self, x, t, y, force_drop_ids=None):
        """
        Forward pass of JiT.
        x: (N, C, H, W) tensor of spatial inputs (3DGS features on grid)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        force_drop_ids: optional (N,) 0/1 tensor — when 1, replace that sample's
            label with the unconditional slot. Lets the trainer pre-sample the
            CFG drop mask so the aux classifier can skip dropped rows.
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)  # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(self.ckpt_wrapper(block, self.feat_rope), x, c, use_reentrant=False)
            else:
                x = block(x, c, rope=self.feat_rope)                                     # (N, T, D)
        # Aux classifier pools the transformer tokens before final_layer so
        # gradients flow through the full trunk but not through the adaLN head.
        if self.aux_classifier is not None:
            self._aux_logits = self.aux_classifier(x.mean(dim=1))
        else:
            self._aux_logits = None
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # Apply CFG to ALL channels (not just first 3 as in image DiT)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                              3DGS DiT Configs                                 #
#################################################################################

def _jit_factory(*, depth, hidden_size, patch_size, num_heads, bottleneck_dim):
    return lambda **kw: DiT(
        depth=depth,
        hidden_size=hidden_size,
        patch_size=patch_size,
        num_heads=num_heads,
        bottleneck_dim=bottleneck_dim,
        **kw,
    )


JiT_3DGS_models = {
    'JiT-XL/8': _jit_factory(depth=28, hidden_size=1152, patch_size=8, num_heads=16, bottleneck_dim=256),
    'JiT-XL/16': _jit_factory(depth=28, hidden_size=1152, patch_size=16, num_heads=16, bottleneck_dim=256),
    'JiT-XL/32': _jit_factory(depth=28, hidden_size=1152, patch_size=32, num_heads=16, bottleneck_dim=256),
    'JiT-L/8': _jit_factory(depth=24, hidden_size=1024, patch_size=8, num_heads=16, bottleneck_dim=128),
    'JiT-L/16': _jit_factory(depth=24, hidden_size=1024, patch_size=16, num_heads=16, bottleneck_dim=128),
    'JiT-L/32': _jit_factory(depth=24, hidden_size=1024, patch_size=32, num_heads=16, bottleneck_dim=128),
    'JiT-B/8': _jit_factory(depth=12, hidden_size=768, patch_size=8, num_heads=12, bottleneck_dim=128),
    'JiT-B/16': _jit_factory(depth=12, hidden_size=768, patch_size=16, num_heads=12, bottleneck_dim=128),
    'JiT-B/32': _jit_factory(depth=12, hidden_size=768, patch_size=32, num_heads=12, bottleneck_dim=128),
    'JiT-S/8': _jit_factory(depth=12, hidden_size=384, patch_size=8, num_heads=6, bottleneck_dim=64),
    'JiT-S/16': _jit_factory(depth=12, hidden_size=384, patch_size=16, num_heads=6, bottleneck_dim=64),
    'JiT-S/32': _jit_factory(depth=12, hidden_size=384, patch_size=32, num_heads=6, bottleneck_dim=64),
}


DiT_3DGS_models = {
    **JiT_3DGS_models,
    'DiT-XL/8': _jit_factory(depth=28, hidden_size=1152, patch_size=8, num_heads=16, bottleneck_dim=256),
    'DiT-L/8': _jit_factory(depth=24, hidden_size=1024, patch_size=8, num_heads=16, bottleneck_dim=128),
    'DiT-B/8': _jit_factory(depth=12, hidden_size=768, patch_size=8, num_heads=12, bottleneck_dim=128),
    'DiT-S/8': _jit_factory(depth=12, hidden_size=384, patch_size=8, num_heads=6, bottleneck_dim=64),
}
