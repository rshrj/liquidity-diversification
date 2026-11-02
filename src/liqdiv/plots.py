"""Static figures (matplotlib). Palette: validated reference categorical slots 1-3
plus muted gray for the random baseline; diverging blue <-> gray <-> red for
correlations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

from .analysis import Experiment  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"ascending": "#2a78d6", "descending": "#eb6834", "oracle": "#1baf7a", "random": MUTED}
LABELS = {
    "ascending": "Correlation ascending",
    "descending": "Correlation descending",
    "random": "Random order (mean, 5–95% band)",
    "oracle": "Oracle greedy (in-sample bound)",
}
DIVERGING = LinearSegmentedColormap.from_list("corr", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9.5,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
    }
)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)


def plot_correlation(corr: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3.2, 0.55 * n + 1.6))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            r, g, b = to_rgb(DIVERGING((corr[i, j] + 1) / 2))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if lum < 0.5 else INK)
    ax.set_xticks(range(n), labels, rotation=90)
    ax.set_yticks(range(n), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED, labelcolor=INK2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_curves(exps: list[Experiment], titles: list[str], path: Path, suptitle: str) -> None:
    fig, axes = plt.subplots(1, len(exps), figsize=(5.6 * len(exps), 4.3), squeeze=False)
    handles = {}
    for ax, exp, title in zip(axes[0], exps, titles):
        _style(ax)
        n = len(next(iter(exp.curves.values())))
        k = np.arange(1, n + 1)
        lo, hi = exp.random_band
        ax.fill_between(k, lo, hi, color=MUTED, alpha=0.16, linewidth=0)
        ax.axhline(exp.target, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.9)
        ax.annotate("25% improvement", (n, exp.target), xytext=(0, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color=INK2)
        for key in ("random", "descending", "ascending", "oracle"):
            if key not in exp.curves:
                continue
            ls = (0, (5, 2.5)) if key == "oracle" else "-"
            (h,) = ax.plot(k, exp.curves[key], color=SERIES[key], linewidth=1.75, linestyle=ls,
                           marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                           label=LABELS[key], zorder=3)
            handles[key] = h
        ax.set_xticks(k)
        ax.set_xlabel("Number of venues in aggregator")
        ax.set_ylabel("Average inside spread (bp)")
        ax.set_title(title, loc="left", fontsize=10.5, pad=8)
    order = [h for key in ("ascending", "descending", "random", "oracle") if (h := handles.get(key))]
    fig.legend(handles=order, loc="upper center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, 0.93), fontsize=8.5, labelcolor=INK2)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=160)
    plt.close(fig)
