"""CLIP Maximum Mean Discrepancy (CMMD) for generated renders.

Lower scores indicate closer feature distributions. Use the same sample count,
renderer, cameras, and sampling settings when comparing runs. The implementation
includes kernel diagonals, so finite-sample scores depend on sample count."""
from __future__ import annotations

import numpy as np
import torch


def cmmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 10.0, scale: float = 1000.0) -> float:
    """CLIP-MMD with a Gaussian RBF kernel between two feature sets.

    ``x``, ``y``: (N, D) and (M, D) features (L2-normalized CLIP embeddings).
    Lower is better. ``scale`` is the conventional 1000x reporting factor.
    """
    x = x.double()
    y = y.double()
    x_sq = (x * x).sum(dim=1)
    y_sq = (y * y).sum(dim=1)
    gamma = 1.0 / (2.0 * sigma ** 2)
    k_xx = torch.exp(-gamma * (x_sq[:, None] + x_sq[None, :] - 2 * (x @ x.T))).mean()
    k_yy = torch.exp(-gamma * (y_sq[:, None] + y_sq[None, :] - 2 * (y @ y.T))).mean()
    k_xy = torch.exp(-gamma * (x_sq[:, None] + y_sq[None, :] - 2 * (x @ y.T))).mean()
    return float(scale * (k_xx + k_yy - 2 * k_xy))


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """FID between two feature matrices. Provided for offline use only.

    Do NOT wire this into the training loop: FID is heavily biased at the sample
    counts a 3D pipeline can afford in-loop (measured unusable below ~10k), which
    is exactly why CMMD is the online metric here.
    """
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


class ClipFeatureExtractor:
    """CLIP-L/14 image features for CMMD. Kept separate from the text encoder."""

    def __init__(self, device: torch.device, model_id: str = "openai/clip-vit-large-patch14"):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.device = device
        self.processor = CLIPImageProcessor.from_pretrained(model_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """``images``: (N, H, W, 3) float in [0,1]. Returns L2-normalized (N, D)."""
        feats = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i: i + batch_size].detach().float().clamp(0, 1).cpu().numpy()
            inp = self.processor(images=list((chunk * 255).astype("uint8")),
                                 return_tensors="pt").to(self.device)
            f = self.model(**inp).image_embeds
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        return torch.cat(feats, dim=0)
