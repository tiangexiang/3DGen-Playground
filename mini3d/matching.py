"""Point matching losses for Gaussian splat reconstruction."""
import math
import os
import shutil
from pathlib import Path
import torch as th

# --- KeOps backend for the Chamfer kNN search (optional; ~2.4-2.8x faster) -----
# pytorch3d's kNN is correct but memory-bandwidth-bound. KeOps does the same
# nearest-neighbor search as an online reduction (no NxN materialization) and is
# substantially faster on this 16,384-point cloud. We use KeOps only for the
# (non-differentiable) argmin search and keep the differentiable distance in torch
# via gather — numerically identical to pytorch3d. Falls back to pytorch3d if
# pykeops or a CUDA toolchain is unavailable.
_KEOPS_STATE = {"checked": False, "ok": False}


def _keops_setup_cuda_env():
    """Point KeOps' runtime JIT at a CUDA toolchain (nvcc + libnvrtc) via env vars.

    KeOps compiles/links its kernels with g++/nvrtc at first use, which need the
    CUDA lib dir on LIBRARY_PATH/LD_LIBRARY_PATH. Idempotent; auto-detects a system
    CUDA install when CUDA_PATH/CUDA_HOME are unset. The launcher should also export
    these before process start so the compiled kernel loads reliably.
    """
    if os.environ.get("_KEOPS_CUDA_ENV_DONE"):
        return
    cuda = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if not cuda or not os.path.isdir(cuda):
        nvcc = shutil.which("nvcc")
        cuda = str(Path(nvcc).resolve().parent.parent) if nvcc else None
    if cuda and os.path.isdir(cuda):
        os.environ["CUDA_PATH"] = cuda
        os.environ.setdefault("CUDA_HOME", cuda)
        for var, sub in (("PATH", "bin"), ("LIBRARY_PATH", "lib64"), ("LD_LIBRARY_PATH", "lib64")):
            val = os.path.join(cuda, sub)
            cur = os.environ.get(var, "")
            if val not in cur.split(os.pathsep):
                os.environ[var] = val + (os.pathsep + cur if cur else "")
    os.environ["_KEOPS_CUDA_ENV_DONE"] = "1"


def _keops_available():
    """True if pykeops imports (and its CUDA env is set up); cached. Else pytorch3d."""
    if not _KEOPS_STATE["checked"]:
        _KEOPS_STATE["checked"] = True
        try:
            _keops_setup_cuda_env()
            from pykeops.torch import LazyTensor  # noqa: F401
            _KEOPS_STATE["ok"] = True
        except Exception:
            _KEOPS_STATE["ok"] = False
    return _KEOPS_STATE["ok"]


def _keops_argmin(x, y):
    """Index of nearest y-row (squared L2) per x-row. x:(B,N,Ds) y:(B,M,Ds) -> (B,N) long.

    Search only — argmin indices carry no gradient (as in any kNN). Inputs are
    detached so KeOps does no autograd bookkeeping; the differentiable distance is
    recomputed in torch by the caller on the gathered rows.
    """
    from pykeops.torch import LazyTensor
    x_i = LazyTensor(x.detach().unsqueeze(2).contiguous())  # (B,N,1,Ds)
    y_j = LazyTensor(y.detach().unsqueeze(1).contiguous())  # (B,1,M,Ds)
    return ((x_i - y_j) ** 2).sum(-1).argmin(dim=2)[..., 0].long()  # (B,N)


def _gather_rows(src, idx):
    """src:(B,M,C), idx:(B,N) -> (B,N,C): row idx[b,n] of src[b]."""
    return th.gather(src, 1, idx.unsqueeze(-1).expand(-1, -1, src.shape[-1]))


def _subsample_query_idx(B, N, M, device):
    """``(B, M)`` long: M distinct uniform query indices per batch element (no replacement).

    Used to draw a fresh random subset of *query* points for the Chamfer search each
    step. argsort-of-rand is the cheap vectorized way to sample without replacement per
    row; the cost is negligible next to the kNN search.
    """
    return th.rand(B, N, device=device).argsort(dim=1)[:, :M]


def _sinkhorn_log(C, eps, iters):
    """Entropic-OT transport plan via log-domain Sinkhorn (uniform marginals).

    ``C``: (B, nP, M, M) squared-distance cost. Returns the (B, nP, M, M) soft transport
    plan assigning M pred points to M target points within each patch, marginals a=b=1/M.
    Fixed ``iters`` + pure tensor ops keep it torch.compile-safe (no data-dependent control
    flow, no CPU sync). As ``eps`` -> 0 the plan -> a hard permutation (exact assignment);
    larger ``eps`` -> softer/blurrier plan (re-collapses toward the mean).
    """
    M = C.shape[-1]
    logK = -C / eps                                              # (B, nP, M, M)
    logm = -math.log(M)                                         # log uniform marginal
    f = th.zeros(C.shape[:-1], device=C.device, dtype=C.dtype)  # (B, nP, M)
    g = th.zeros_like(f)
    for _ in range(int(iters)):
        f = logm - th.logsumexp(logK + g.unsqueeze(-2), dim=-1)
        g = logm - th.logsumexp(logK + f.unsqueeze(-1), dim=-2)
    return th.exp(logK + f.unsqueeze(-1) + g.unsqueeze(-2))     # (B, nP, M, M) plan


# Lazily-compiled handle for _sinkhorn_log. The loop is memory-bound: eager, each iter
# materializes the full (B,nP,M,M) broadcast `logK + g.unsqueeze(...)` to HBM before the
# logsumexp reduction (200 such ~1GB round-trips/step at B=256). Inductor FUSES the broadcast
# into the reduction, keeping it in SRAM — measured ~8x on the Sinkhorn term (1463->184 ms,
# B=128/100 iters), numerically identical fp32 (same math, just fused). Compile-safe by design
# (fixed iters, no data-dependent control flow / CPU sync). Compiles once; shapes are static
# because training uses drop_last=True. Opt-in via compile_sinkhorn so probes/tests stay eager.
_sinkhorn_log_compiled = None


def _resolve_sinkhorn(compiled: bool):
    """Return the eager or (lazily) compiled _sinkhorn_log. Specializes on shape/eps/iters;
    fine since those are constant within a run (drop_last=True; iters launch-time fixed)."""
    global _sinkhorn_log_compiled
    if not compiled:
        return _sinkhorn_log
    if _sinkhorn_log_compiled is None:
        _sinkhorn_log_compiled = th.compile(_sinkhorn_log)
    return _sinkhorn_log_compiled


def _chamfer_recon_loss(pred, target, mode, channel_loss_weights=None, weight=1.0,
                        subsample=0, patch_size=8, rev_weight=1.0,
                        sinkhorn_eps=0.05, sinkhorn_iters=50, return_diag=False,
                        compile_sinkhorn=False, huber_delta=0.0, residual_mad=False,
                        dcd_alpha=50.0, dcd_lambda=1.0, chunk_1d=False):
    """Per-sample point-set reconstruction loss for (B,C,H,W) containers.

    Modes: chamfer_feature matches all features; chamfer_geometric matches XYZ
    and compares full features; chamfer_patch restricts nearest neighbors to
    patches; chamfer_density adds density-aware penalties; sinkhorn_patch and
    sinkhorn_patch_hard use soft transport and argmax transport assignments.

    chunk_1d interprets patch_size as a contiguous splat count. Otherwise it
    is a square patch width. channel_loss_weights scales the matching space.
    subsample limits queries for global matching; rev_weight scales the reverse
    Chamfer term. Neither changes the symmetric Sinkhorn objective.

    huber_delta > 0 applies a pseudo-Huber cost to patch matching. The transport
    plan is detached; gradients flow through matched distances. return_diag
    returns (loss, diagnostics), including convergence and optional residual
    statistics. Otherwise returns loss with shape (B,), scaled by weight."""
    def _ret(loss, diag=None):
        return (loss, diag or {}) if return_diag else loss

    # kNN kernels (both backends) require float32; cast keeps autocast (fp16/bf16)
    # forward passes from breaking them. The cast is differentiable so grads still
    # reach ``model_output`` through the torch-side distance.
    pred_pc = pred.flatten(2).transpose(1, 2).float().contiguous()      # (B, N, C)
    target_pc = target.flatten(2).transpose(1, 2).float().contiguous()  # (B, N, C)

    w = None
    if channel_loss_weights is not None:
        w = channel_loss_weights.to(device=pred_pc.device, dtype=pred_pc.dtype).clamp_min(0)

    use_keops = _keops_available()

    B, N = pred_pc.shape[0], pred_pc.shape[1]
    M = subsample if (subsample and 0 < subsample < N) else 0  # 0 => use all points

    if mode == "chamfer_feature":
        x, y = pred_pc, target_pc
        if w is not None:
            scale = w.sqrt().view(1, 1, -1)  # weighted Euclidean: Σ w_c (Δ_c)²
            x = x * scale
            y = y * scale
        # Query points for each direction: a fresh random subset of M when subsampling,
        # else the full cloud. The full target (y / x) is always searched, so each
        # query's nearest neighbour is exact.
        if M:
            xq = _gather_rows(x, _subsample_query_idx(B, N, M, x.device))  # (B,M,C)
            yq = _gather_rows(y, _subsample_query_idx(B, N, M, y.device))
        else:
            xq, yq = x, y
        if use_keops:
            # search full C-dim feature space; differentiable distance via gather.
            sq_f = (xq - _gather_rows(y, _keops_argmin(xq, y))) ** 2  # (B,M|N,C)
            sq_r = (yq - _gather_rows(x, _keops_argmin(yq, x))) ** 2
            # rev_weight scales the backward (GT-as-query) term — the coverage/recall
            # direction (every GT Gaussian must have a nearby prediction).
            loss_b = sq_f.sum(-1).mean(1) + rev_weight * sq_r.sum(-1).mean(1)
        else:
            from pytorch3d.loss import chamfer_distance
            # single_directional so we can feed the subsampled query set on each side.
            kw = dict(batch_reduction=None, point_reduction="mean", norm=2)
            loss_f, _ = chamfer_distance(xq, y, single_directional=True, **kw)  # (B,)
            loss_r, _ = chamfer_distance(yq, x, single_directional=True, **kw)
            loss_b = loss_f + rev_weight * loss_r
        return _ret(weight * loss_b)

    if mode == "chamfer_geometric":
        wv = w.view(1, 1, -1) if w is not None else None
        pred_xyz = pred_pc[..., :3].contiguous()
        gt_xyz = target_pc[..., :3].contiguous()
        # Subsample the query side (pred for forward, target for reverse); match against
        # the full opposite cloud by xyz, then squared error on all channels.
        if M:
            pred_q = _gather_rows(pred_pc, _subsample_query_idx(B, N, M, pred_pc.device))    # (B,M,C)
            gt_q = _gather_rows(target_pc, _subsample_query_idx(B, N, M, target_pc.device))  # (B,M,C)
            pred_q_xyz = pred_q[..., :3].contiguous()
            gt_q_xyz = gt_q[..., :3].contiguous()
        else:
            pred_q, gt_q = pred_pc, target_pc
            pred_q_xyz, gt_q_xyz = pred_xyz, gt_xyz
        if use_keops:
            gt_match = _gather_rows(target_pc, _keops_argmin(pred_q_xyz, gt_xyz))    # (B,M|N,C)
            pred_match = _gather_rows(pred_pc, _keops_argmin(gt_q_xyz, pred_xyz))    # (B,M|N,C)
        else:
            from pytorch3d.ops import knn_gather, knn_points
            gt_match = knn_gather(target_pc, knn_points(pred_q_xyz, gt_xyz, K=1).idx)[:, :, 0, :]
            pred_match = knn_gather(pred_pc, knn_points(gt_q_xyz, pred_xyz, K=1).idx)[:, :, 0, :]
        sq_fwd = (pred_q - gt_match) ** 2   # forward: pred query -> nearest target by xyz
        sq_rev = (gt_q - pred_match) ** 2   # reverse: target query -> nearest pred by xyz
        if wv is not None:
            sq_fwd = sq_fwd * wv
            sq_rev = sq_rev * wv
        return _ret(weight * 0.5 * (sq_fwd.mean(dim=(1, 2)) + rev_weight * sq_rev.mean(dim=(1, 2))))

    if mode in ("chamfer_patch", "chamfer_density", "sinkhorn_patch", "sinkhorn_patch_hard"):
        # Bidirectional matching WITHIN each token's Gaussian group. Works from the
        # original (B,C,H,W) atlas (not the flattened pred_pc/target_pc) so we can group
        # points exactly as the tokenizer does. Two groupings:
        #   chunk_1d=False: P×P atlas patch  (Conv2d patchify / permute_atlas('patch'))
        #   chunk_1d=True : a contiguous chunk of `patch_size` Gaussians along the 1D
        #                   (Hilbert) sequence (Conv1d chunk / permute_atlas('chunk')).
        B, C, H, W = pred.shape
        if chunk_1d:
            G = int(patch_size)  # patch_size carries the 1D group size in this mode
            N = H * W
            if N % G:
                raise ValueError(f"chunk_1d needs H*W divisible by group_size={G}; got N={N}")
            M = G  # Gaussians per token

            def to_patches(t):
                # (B,C,H,W)->(B,C,N)->(B,nP,M,C); row-major flatten = the token sequence.
                return t.float().reshape(B, C, N).reshape(B, C, N // G, G).permute(0, 2, 3, 1).contiguous()
        else:
            P = int(patch_size)
            if H % P or W % P:
                raise ValueError(
                    f"chamfer_patch needs H,W divisible by patch_size={P}; got {(H, W)}"
                )
            nH, nW = H // P, W // P
            M = P * P  # Gaussians per patch

            def to_patches(t):
                # (B,C,H,W) -> (B, nP, M, C); identical grouping to permute_atlas('patch').
                return (
                    t.float()
                    .reshape(B, C, nH, P, nW, P)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(B, nH * nW, C, M)
                    .transpose(-1, -2)
                    .contiguous()
                )

        xp = to_patches(pred)      # (B, nP, M, C)
        yp = to_patches(target)
        if w is not None:
            scale = w.sqrt().view(1, 1, 1, -1)  # weighted Euclidean, as in chamfer_feature
            xp = xp * scale
            yp = yp * scale
        # Squared in-patch distances via the Gram form ‖x‖²+‖y‖²−2x·y. Avoids the
        # (B,nP,M,M,C) broadcast (~15GB at B=256/C=14); materializes only (B,nP,M,M)
        # (~1.07GB at B=256/P=8). No sqrt -> gradient-safe at coincident points.
        aa = xp.pow(2).sum(-1)                       # (B, nP, M)
        bb = yp.pow(2).sum(-1)                       # (B, nP, M)
        ab = xp @ yp.transpose(-1, -2)              # (B, nP, M, M)
        sq = (aa.unsqueeze(-1) + bb.unsqueeze(-2) - 2 * ab).clamp_min(0)  # (B, nP, M, M)
        # Keep a handle on the RAW (pre-huber) squared distances so the residual-MAD
        # diagnostic always reports the natural residual radius r=√sq, in the same units
        # as `huber_delta`, regardless of whether the huber transform is applied below.
        # When huber_delta==0 (the intended diagnostic run) the transform is a no-op, so
        # sq_raw is exactly the matched cost. (Just a tensor reference — no copy; the huber
        # block rebinds `sq` to a new tensor, leaving sq_raw pointing at the original.)
        sq_raw = sq
        if huber_delta and huber_delta > 0.0:
            # Pseudo-Huber on the per-pair radius r=‖x−y‖ (r²=sq): robustifies the ground
            # cost so a few large-residual pairs (bad matches / high-noise outliers) stop
            # dominating both the OT assignment AND (via the hard gather / soft <P,sq>) the
            # gradient. Reusing the name `sq` makes every downstream consumer use the robust
            # cost. A single elementwise map of the (B,nP,M,M) sq tensor — keeps the Gram-form
            # memory trick (no (B,nP,M,M,C) broadcast) and stays outside the compiled Sinkhorn
            # loop (changing δ never recompiles). The 2δ² scaling fixes the INLIER scale to sq:
            #   small r → sq (= MSE cost, so chamfer_loss_weight stays calibrated)
            #   large r → 2δ·r (linear ⇒ bounded gradient δ/√sq)
            #   δ → ∞ recovers sq exactly. grad d/dsq = 1/√(1+sq/δ²) ∈ (0,1], smooth at r=0.
            # Use the rationalized form 2δ²(√(1+u)−1) = 2·sq/(√(1+u)+1), u=sq/δ², which is
            # algebraically identical but free of the √(1+u)−1 catastrophic cancellation that
            # underflows to 0 (instead of → sq) for large δ in fp32.
            d2 = huber_delta * huber_delta
            sq = 2.0 * sq / ((1.0 + sq / d2).sqrt() + 1.0)
        patch_diag = {}
        if mode == "chamfer_patch":
            fwd = sq.min(dim=-1).values              # pred  -> nearest in-patch target
            rev = sq.min(dim=-2).values              # target -> nearest in-patch pred
            # mean over points (patches × M) — sum-over-C already in sq, matches chamfer_feature scale.
            loss_b = fwd.mean(dim=(1, 2)) + rev_weight * rev.mean(dim=(1, 2))  # (B,)
        elif mode == "chamfer_density":
            # Density-aware Chamfer Distance (Wu et al. 2021, arXiv:2111.12702, Eq. 4), restricted
            # to within-patch matching (same grouping/cost as chamfer_patch). Two ingredients turn
            # the plain in-patch Chamfer into DCD:
            #   (1) a BOUNDED kernel  sim = exp(-α·r)  on the matched radius r=‖x−ŷ‖ (r²=sq_raw),
            #       so each point's cost  1−sim ∈ [0,1)  and far-outliers saturate instead of
            #       dominating (CD's failure mode). Gradient wrt coords is α·sim·(x−ŷ)/√(sq+eps),
            #       bounded by α (the (x−ŷ) factor cancels the 1/√sq from the sqrt at coincidence),
            #       so the +eps only regularises the exactly-coincident corner — it is NOT a scale knob.
            #   (2) a DENSITY term  n^(−λ): n_ŷ = how many in-patch query points collapsed onto the
            #       same nearest target ŷ. Many-to-one (mode-collapse) is penalised — the reward for a
            #       target is split across the n predictions that claim it. n is a discrete count →
            #       detached (no gradient), matching the reference impl. λ=1 ⇒ exact Eq. 4.
            # Per point:  1 − n^(−λ)·exp(−α·r).  Symmetric (both directions), ×½; rev_weight kept for
            # parity with the other modes (1.0 = the paper's symmetric DCD). Uses sq_raw so huber is a
            # no-op (DCD's kernel is its own robustifier); the argmin is unchanged by huber (monotone).
            eps = 1e-8
            d_fwd_sq, idx_fwd = sq_raw.min(dim=-1)    # pred  -> nearest in-patch target (idx over targets)
            d_rev_sq, idx_rev = sq_raw.min(dim=-2)    # target -> nearest in-patch pred   (idx over preds)
            d_fwd = (d_fwd_sq + eps).sqrt()           # (B, nP, M) matched radii r
            d_rev = (d_rev_sq + eps).sqrt()
            # n_ŷ / n_x̂: collision count of each argmin index within its patch (scatter-count → gather).
            n_fwd = th.zeros_like(d_fwd).scatter_add_(-1, idx_fwd, th.ones_like(d_fwd)).gather(-1, idx_fwd)
            n_rev = th.zeros_like(d_rev).scatter_add_(-1, idx_rev, th.ones_like(d_rev)).gather(-1, idx_rev)
            n_fwd = n_fwd.detach().clamp_min(1.0)
            n_rev = n_rev.detach().clamp_min(1.0)
            term_fwd = 1.0 - n_fwd.pow(-dcd_lambda) * th.exp(-dcd_alpha * d_fwd)   # (B, nP, M) ∈ [0,1)
            term_rev = 1.0 - n_rev.pow(-dcd_lambda) * th.exp(-dcd_alpha * d_rev)
            loss_b = 0.5 * (term_fwd.mean(dim=(1, 2)) + rev_weight * term_rev.mean(dim=(1, 2)))  # (B,)
            if return_diag:
                # The two quantities DCD is built to watch: mean collapse factor (1.0 = perfect
                # bijection, >1 = many preds per target) and mean matched radius r.
                patch_diag["dcd_mean_n"] = 0.5 * (n_fwd.mean() + n_rev.mean())
                patch_diag["dcd_mean_radius"] = 0.5 * (d_fwd.mean() + d_rev.mean())
        else:  # sinkhorn_patch (soft) / sinkhorn_patch_hard — optimal-assignment matching
            # Optimal assignment of the M pred points to the M target points within each patch,
            # approximated by entropic-OT log-Sinkhorn. The plan only decides the MATCHING —
            # the gradient flows through `sq`, never through the plan — so it is computed under
            # no_grad. Otherwise autograd records the iters-deep recursion for a backward that
            # never uses it (~1.1 GB/iter of dead graph at B=128 — the dominant training-memory
            # cost and a hard cap on sinkhorn_iters); under no_grad peak memory is O(1) in iters.
            # rev_weight is irrelevant (OT marginals already cover both directions).
            with th.no_grad():
                # compiled (fused logsumexp) when compile_sinkhorn — ~8x, numerically identical.
                P = _resolve_sinkhorn(compile_sinkhorn)(sq, sinkhorn_eps, sinkhorn_iters)  # (B,nP,M,M); no graph
                if mode == "sinkhorn_patch_hard":
                    sigma = P.argmax(dim=-1)                          # (B, nP, M) assigned target per pred
                if return_diag:
                    # Sinkhorn convergence monitor. The log-domain recursion ends on the column
                    # update, so the COLUMN marginal (sum over pred i, dim=-2) is exact (=1/M);
                    # the ROW marginal (sum over target j, dim=-1) carries the residual. mean
                    # |rowsum − 1/M| → 0 as the plan converges to doubly-stochastic, rising if
                    # sinkhorn_iters is too few for the eps / cost sharpness. ~free.
                    row = P.sum(dim=-1)                               # (B, nP, M)
                    patch_diag["sinkhorn_marginal_resid"] = (row - (1.0 / M)).abs().mean()
                    if mode == "sinkhorn_patch_hard":
                        # Collision fraction: preds whose argmax target is shared with another
                        # pred in the patch — i.e. how far the rounded assignment is from a true
                        # bijection. Floors at the eps non-permutation (≠0), not at full conv.
                        col_counts = th.zeros_like(row)              # (B, nP, M) counts over columns j
                        col_counts.scatter_add_(-1, sigma, th.ones_like(row))
                        extra = (col_counts - 1.0).clamp_min(0).sum(dim=-1)  # duplicate picks per patch
                        patch_diag["sinkhorn_collision_frac"] = (extra / M).mean()
                    if residual_mad:
                        # Robust scale of the matched-residual RADIUS r=‖pred−tgt_σ(i)‖ (r²=sq_raw
                        # at the assigned target) — the exact quantity huber_delta thresholds. Use
                        # the hard argmax assignment in BOTH sinkhorn modes (the soft plan has no
                        # single match) and gather from the RAW (pre-huber) cost so the number is
                        # the natural residual scale regardless of huber_delta. median/MAD/p90/p99
                        # over the whole micro-batch point set; the window mean is taken upstream.
                        sig = sigma if mode == "sinkhorn_patch_hard" else P.argmax(dim=-1)
                        r = sq_raw.gather(-1, sig.unsqueeze(-1)).squeeze(-1).clamp_min(0).sqrt().reshape(-1)
                        med = r.median()
                        patch_diag["recon_residual_median"] = med
                        patch_diag["recon_residual_mad"] = (r - med).abs().median()
                        qs = th.quantile(r, th.tensor([0.9, 0.99], device=r.device, dtype=r.dtype))
                        patch_diag["recon_residual_p90"] = qs[0]
                        patch_diag["recon_residual_p99"] = qs[1]
            if mode == "sinkhorn_patch_hard":
                # HARD assignment: MSE to the single argmax-matched target (DETR-style hard
                # matching). σ is detached, so the gradient is a crisp per-pred MSE toward
                # tgt_σ(i) — no soft blend, hence no mean-pull. Gradient flows through `sq`
                # gathered at the detached σ. mean_i sq[i,σ(i)] = the same scale as the soft
                # <P, sq> when the plan is a permutation.
                hard_sq = sq.gather(-1, sigma.unsqueeze(-1)).squeeze(-1)  # (B, nP, M) = sq[i,σ(i)]
                loss_b = hard_sq.mean(dim=(1, 2))                     # (B,)
            else:  # sinkhorn_patch (soft)
                # SOFT: <P, sq> is the plan-weighted (blended) matched distance; the induced
                # per-pred gradient pulls toward M·Σ_j P[i,j]·tgt_j (a blend → mean-pull when P
                # is blurry). Plan mass sums to 1 per patch → ~chamfer_patch scale.
                loss_b = (P * sq).sum(dim=(-1, -2)).mean(dim=1)      # (B,) gradient via `sq` only
        return _ret(weight * loss_b, patch_diag)

    raise ValueError(f"unknown recon_loss mode: {mode!r}")
