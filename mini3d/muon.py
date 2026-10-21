"""Muon optimizer (Newton-Schulz orthogonalized momentum) for 2D weight matrices,
with a decoupled-AdamW fallback for every other parameter, in a single
``torch.optim.Optimizer`` so the existing one-optimizer training loop (accelerate
prepare, per-group LR schedule, checkpoint state_dict) keeps working unchanged.

Muon (Jordan et al., 2024): for each 2D matrix parameter, take SGD-momentum and
then replace the raw update with its nearest semi-orthogonal matrix via a quintic
Newton-Schulz iteration (run in bf16). Empirically strong on transformer hidden
weights. It is NOT appropriate for embedding tables, the output head, or 1D
gains/biases — those stay on AdamW. Group membership is chosen by the caller via a
per-group ``use_muon`` flag; this class only implements the two update rules.

Single-process / DDP: under DDP the gradients are all-reduced before ``step``, so
every rank computes an identical Newton-Schulz update and parameters stay in sync.
The NS work is recomputed per rank (not sharded across ranks like the distributed
reference) — fine for our scale; revisit if it becomes a throughput bottleneck.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton-Schulz iteration approximating the orthogonal polar factor of G.

    Runs in bf16 (the iteration is robust to it and it is ~free vs fp32). Coefficients
    (a,b,c) are Jordan's tuned quintic. The transpose keeps the iterated matrix wide so
    the X @ X.T products are over the smaller dimension.
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Spectral-norm-ish normalization so the iteration starts in its convergence basin.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_update(grad: torch.Tensor, buf: torch.Tensor, *, beta: float,
                 ns_steps: int, nesterov: bool) -> torch.Tensor:
    """One Muon update direction for a single matrix. ``buf`` (the momentum EMA) is
    updated in place; ``grad`` is not modified."""
    buf.lerp_(grad, 1.0 - beta)                       # buf = beta*buf + (1-beta)*grad
    update = torch.lerp(grad, buf, beta) if nesterov else buf
    if update.ndim == 4:                              # conv filter -> (out, in*kh*kw)
        update = update.flatten(1)
    update = zeropower_via_newtonschulz5(update, ns_steps)
    # Scale so the update RMS roughly matches an SGD step regardless of aspect ratio.
    update = update * (max(1.0, update.size(-2) / update.size(-1)) ** 0.5)
    return update


class Muon(torch.optim.Optimizer):
    """Combined Muon (``use_muon=True`` groups) + decoupled AdamW (``use_muon=False``).

    Each param group MUST carry ``use_muon: bool``. Muon groups honor
    ``lr, momentum, weight_decay, ns_steps, nesterov``; AdamW groups honor
    ``lr, betas, eps, weight_decay``. The per-group ``lr`` is set externally each
    step by the trainer's LR schedule (Muon groups want a much larger lr than AdamW).
    """

    def __init__(self, param_groups):
        groups = list(param_groups)
        for g in groups:
            if "use_muon" not in g:
                raise ValueError("every Muon param group must set 'use_muon'")
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
                g.setdefault("nesterov", True)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.999))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            if g["use_muon"]:
                lr, beta, wd = g["lr"], g["momentum"], g["weight_decay"]
                ns_steps, nesterov = g["ns_steps"], g["nesterov"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = _muon_update(
                        p.grad, state["momentum_buffer"],
                        beta=beta, ns_steps=ns_steps, nesterov=nesterov,
                    )
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr)
            else:
                lr, (beta1, beta2) = g["lr"], g["betas"]
                eps, wd = g["eps"], g["weight_decay"]
                for p in g["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(grad, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    bc1 = 1.0 - beta1 ** t
                    bc2 = 1.0 - beta2 ** t
                    denom = (exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-lr / bc1)
        return loss


def classify_muon_params(model, exclude_ids=None):
    """Split ``model`` parameters into (muon, adamw) lists for the combined optimizer.

    Muon gets 2D weight matrices of the transformer body; AdamW gets everything else.
    Per the Muon recipe, embedding tables and the final output head stay on AdamW even
    though they are 2D. ``exclude_ids`` (ids already claimed by another group, e.g. the
    text-projection group) are skipped entirely. Name checks use substrings so they are
    robust to a ``torch.compile`` ``_orig_mod.`` prefix.
    """
    exclude_ids = exclude_ids or set()
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in exclude_ids:
            continue
        is_matrix = (
            p.ndim == 2
            and "embedding_table" not in name   # nn.Embedding lookup table
            and "final_layer.linear" not in name  # output head -> gaussian channels
        )
        (muon if is_matrix else adamw).append(p)
    return muon, adamw
