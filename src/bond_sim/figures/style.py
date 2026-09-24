"""House style for signature figures: one look across every chart, every render.

Palette: blue / orange / aqua, validated as a set (dataviz validator, white surface,
all pairs): worst CVD dE 9.2, worst normal-vision dE 24.0. Aqua sits below 3:1
contrast on white, so every series is direct-labeled (never identified by color
alone). Three categorical series at most; a fourth folds into small multiples.
Bands use one hue (blue), light to dark; the outer model envelope also carries a
45-degree hatch so it survives grayscale print.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
INK = "#0b0b0b"          # titles, values
INK_2 = "#52514e"        # labels, captions
MUTED = "#898781"        # axis ticks, source line
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")          # fixed order, never cycled
BLUE = {"100": "#cde2fb", "200": "#9ec5f4", "250": "#86b6ef", "450": "#2a78d6", "600": "#184f95"}
CENTER, INNER, OUTER, HATCH = BLUE["600"], BLUE["200"], BLUE["100"], BLUE["250"]
EVENT = "#f0efec"        # neutral shading for historical episodes (never a series color)

# Render targets: size in inches and output format. Paper is vector and carries no
# headline (the claim goes to the caption); slide and social carry the claim headline.
RENDERS: Dict[str, Tuple[Tuple[float, float], str, int]] = {
    "paper": ((6.5, 3.6), "pdf", 300),
    "slide": ((10.0, 5.6), "png", 200),
    "social": ((8.0, 4.5), "png", 200),
}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlesize": 12, "axes.titleweight": "semibold",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 8.5,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42,       # embed TrueType so journals accept the PDF
    "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
}


@contextmanager
def house():
    """All figure code draws inside this context, so no global matplotlib state leaks."""
    with mpl.rc_context(RC):
        yield


def new_figure(render: str):
    size, _, _ = RENDERS[render]
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def source_line(fig, text: str):
    """Bottom-left line with the as-of date and data sources; required on every render."""
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.5, color=MUTED, transform=fig.transFigure)
