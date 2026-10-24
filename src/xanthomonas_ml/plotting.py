"""Shared, headless matplotlib setup and a consistent figure style.

The non-interactive ``Agg`` backend is selected before pyplot is imported, so
figures render on a headless Ubuntu server or in CI without a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import; no display required
import matplotlib.pyplot as plt  # noqa: E402

from xanthomonas_ml.utils import ensure_dir  # noqa: E402

# Colour roles (validated categorical palette; light-mode steps for PNG output).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots validate all-pairs
NEUTRAL = "#f0efec"
DIVERGING_NEG = "#2a78d6"  # blue pole
DIVERGING_POS = "#e34948"  # red pole

# Fixed colour per entity (colour follows the entity, never its rank).
CLASS_COLORS = {"non-pathogenic": SERIES[0], "pathogenic": SERIES[1]}
MODEL_COLORS = {"decision_tree": SERIES[0], "l1_logistic_regression": SERIES[1], "random_forest": SERIES[2]}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.family": "sans-serif",
        "figure.dpi": 100,
    }
)


def new_figure(width: float = 7.0, height: float = 4.5, nrows: int = 1, ncols: int = 1):
    """Create a figure and axes (an array of axes when nrows*ncols > 1) with the project style."""
    # Several panels: squeeze=False keeps a 2-D array of axes, so callers can index them the same way.
    return plt.subplots(nrows, ncols, figsize=(width, height), constrained_layout=True, squeeze=False) \
        if nrows * ncols > 1 else plt.subplots(figsize=(width, height), constrained_layout=True)


def save_figure(fig, path: Path, dpi: int = 150) -> Path:
    """Save ``fig`` as PNG (creating folders) and release its memory."""
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def add_note(fig, text: str) -> None:
    """Small muted footnote under the plot (e.g. data source, caveats)."""
    n_lines = text.count("\n") + 1
    bottom = 0.035 * n_lines + 0.01
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        # Reserve space at the bottom so constrained layout keeps axes clear of the note.
        engine.set(rect=(0, bottom, 1, 1 - bottom))
    fig.text(0.01, 0.008, text, fontsize=7.5, color=INK_MUTED, ha="left", va="bottom")
