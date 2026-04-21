"""Tiny loss tracker for training: append-only CSV + overwrite-rendered PNGs.

One record per log print (step-level averages). Plots are redrawn on demand
— typically whenever training images are rendered — and written over the
previous PNGs so there's no growing directory of artifacts.

Memory: per-metric flat Python float lists. A 1M-step run with log_every=200
yields ~5k records × ~15 floats ≈ 600 KB resident. CSV is the persistent
source of truth; on resume the lists are rebuilt by reading it back.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


# Columns recorded per print. "bucket_*" holds the per-t-noise-bucket MSE
# breakdown (value + sample count). Any column that stays all-NaN is skipped
# at plot time so disabled losses don't produce empty figures.
_SCALAR_COLUMNS = (
    "step",
    "mse",
    "render_l1",
    "alpha_l1",
    "lpips",
    "aux",
    "grad_norm",
    "grad_norm_mse",
    "grad_norm_render_l1",
    "grad_norm_alpha_l1",
    "grad_norm_lpips",
    "grad_norm_aux",
    "lr",
    "p_mean",
    "steps_per_sec",
)


def _bucket_edges(num_buckets: int) -> list[tuple[float, float]]:
    return [(i / num_buckets, (i + 1) / num_buckets) for i in range(num_buckets)]


class LossTracker:
    """Collect per-print training metrics and render overwrite-style PNGs.

    Safe to call from rank 0 only — pass ``enabled=False`` on other ranks so
    calls no-op. When ``resume=True`` the tracker reads back any existing
    ``loss_log.csv`` and continues appending to it.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        num_t_buckets: int,
        enabled: bool = True,
        resume: bool = False,
    ) -> None:
        self.enabled = enabled
        self.num_t_buckets = int(num_t_buckets)
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "loss_plots")
        self.csv_path = os.path.join(self.plots_dir, "loss_log.csv")

        self._columns = list(_SCALAR_COLUMNS)
        for i in range(self.num_t_buckets):
            self._columns.append(f"bucket{i}_mse")
            self._columns.append(f"bucket{i}_count")

        self._data: dict[str, list[float]] = {c: [] for c in self._columns}
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None

        if not self.enabled:
            return

        os.makedirs(self.plots_dir, exist_ok=True)

        if resume and os.path.isfile(self.csv_path):
            self._load_existing_csv()
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._columns)
        else:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._columns)
            self._csv_writer.writeheader()
            self._csv_file.flush()

    def _load_existing_csv(self) -> None:
        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col in self._columns:
                        raw = row.get(col, "")
                        try:
                            self._data[col].append(float(raw))
                        except (TypeError, ValueError):
                            self._data[col].append(float("nan"))
        except Exception as exc:
            logger.warning("[loss_tracker] failed to read %s: %s (starting fresh)", self.csv_path, exc)
            self._data = {c: [] for c in self._columns}

    def record(
        self,
        *,
        step: int,
        mse: float,
        render_l1: Optional[float] = None,
        alpha_l1: Optional[float] = None,
        lpips: Optional[float] = None,
        aux: Optional[float] = None,
        grad_norm: Optional[float] = None,
        grad_norm_per_loss: Optional[dict[str, float]] = None,
        lr: Optional[float] = None,
        p_mean: Optional[float] = None,
        steps_per_sec: Optional[float] = None,
        bucket_means: Optional[list[float]] = None,
        bucket_counts: Optional[list[int]] = None,
    ) -> None:
        """Append a single print-interval record to memory + CSV."""
        if not self.enabled:
            return

        def _f(v):
            return float(v) if v is not None else float("nan")

        row = {
            "step": float(step),
            "mse": _f(mse),
            "render_l1": _f(render_l1),
            "alpha_l1": _f(alpha_l1),
            "lpips": _f(lpips),
            "aux": _f(aux),
            "grad_norm": _f(grad_norm),
            "grad_norm_mse": _f((grad_norm_per_loss or {}).get("mse")),
            "grad_norm_render_l1": _f((grad_norm_per_loss or {}).get("render_l1")),
            "grad_norm_alpha_l1": _f((grad_norm_per_loss or {}).get("alpha_l1")),
            "grad_norm_lpips": _f((grad_norm_per_loss or {}).get("lpips")),
            "grad_norm_aux": _f((grad_norm_per_loss or {}).get("aux")),
            "lr": _f(lr),
            "p_mean": _f(p_mean),
            "steps_per_sec": _f(steps_per_sec),
        }
        for i in range(self.num_t_buckets):
            mean_i = bucket_means[i] if (bucket_means and i < len(bucket_means)) else None
            cnt_i = bucket_counts[i] if (bucket_counts and i < len(bucket_counts)) else None
            row[f"bucket{i}_mse"] = _f(mean_i)
            row[f"bucket{i}_count"] = _f(cnt_i)

        for col, val in row.items():
            self._data[col].append(val)

        if self._csv_writer is not None:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def _has_any_finite(self, col: str) -> bool:
        return any(v == v and v != 0.0 for v in self._data[col])  # NaN filter via self-eq

    def flush_plots(self) -> None:
        """Render all PNGs, overwriting the previous files. Cheap (~tens of ms)."""
        if not self.enabled:
            return
        steps = self._data["step"]
        if not steps:
            return

        try:
            self._plot_single("mse.png", "MSE", [("mse", "MSE")])
            if self._has_any_finite("render_l1"):
                self._plot_single("render_l1.png", "Render L1", [("render_l1", "Render L1")])
            if self._has_any_finite("alpha_l1"):
                self._plot_single("alpha_l1.png", "Alpha-mask L1", [("alpha_l1", "Alpha L1")])
            if self._has_any_finite("lpips"):
                self._plot_single("lpips.png", "Render LPIPS", [("lpips", "LPIPS")])
            if self._has_any_finite("aux"):
                self._plot_single("aux.png", "Aux classifier CE", [("aux", "Aux CE")])

            if self._has_any_finite("grad_norm"):
                self._plot_single("grad_norm.png", "Grad norm", [("grad_norm", "||grad||")])

            per_loss_series = [
                (col, col.replace("grad_norm_", ""))
                for col in (
                    "grad_norm_mse",
                    "grad_norm_render_l1",
                    "grad_norm_alpha_l1",
                    "grad_norm_lpips",
                    "grad_norm_aux",
                )
                if self._has_any_finite(col)
            ]
            if per_loss_series:
                self._plot_single(
                    "grad_norm_per_loss.png",
                    "Per-loss gradient norms",
                    per_loss_series,
                    log_y=True,
                )

            self._plot_bucket_mse()
        except Exception as exc:
            logger.warning("[loss_tracker] plot flush failed: %s", exc)

    def _plot_single(
        self,
        filename: str,
        title: str,
        series: list[tuple[str, str]],
        *,
        log_y: bool = False,
    ) -> None:
        steps = self._data["step"]
        fig, ax = plt.subplots(figsize=(8, 4))
        plotted = False
        for col, label in series:
            ys = self._data[col]
            xs_clean, ys_clean = _finite_pairs(steps, ys)
            if not xs_clean:
                continue
            ax.plot(xs_clean, ys_clean, label=label, linewidth=1.2)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_xlabel("step")
        ax.set_ylabel(title)
        ax.set_title(title)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, filename), dpi=110)
        plt.close(fig)

    def _plot_bucket_mse(self) -> None:
        """One line per t-bucket: low-t = noisy samples, high-t = clean (FM convention)."""
        steps = self._data["step"]
        fig, ax = plt.subplots(figsize=(8, 4))
        edges = _bucket_edges(self.num_t_buckets)
        plotted = False
        cmap = plt.get_cmap("viridis")
        for i, (lo, hi) in enumerate(edges):
            col = f"bucket{i}_mse"
            ys = self._data[col]
            xs_clean, ys_clean = _finite_pairs(steps, ys)
            if not xs_clean:
                continue
            color = cmap(i / max(1, self.num_t_buckets - 1))
            ax.plot(xs_clean, ys_clean, label=f"t∈[{lo:.2f},{hi:.2f}]", linewidth=1.2, color=color)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_xlabel("step")
        ax.set_ylabel("MSE")
        ax.set_title("MSE by t-bucket (low t = noisy, high t = clean)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "mse_by_t_bucket.png"), dpi=110)
        plt.close(fig)

    def close(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None


def _finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    xs_out: list[float] = []
    ys_out: list[float] = []
    for x, y in zip(xs, ys):
        if y == y:  # NaN filter
            xs_out.append(x)
            ys_out.append(y)
    return xs_out, ys_out
