"""Two-layer uncertainty band (charter A9): the rule every signature figure follows.

Inner layer: spread within the primary specification (path quantiles, or the
sampling band of a probability). Outer layer: the model range, i.e. the same
quantity under both macro blocks and every WARN'd assumption. A path-only fan
looks falsely precise here: the persistence window alone moves P(trigger) from
0.998 to 0.467 while the two blocks differ by 0.007.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from . import style


@dataclass(frozen=True)
class Envelope:
    lo: np.ndarray
    hi: np.ndarray
    lo_driver: np.ndarray    # variant name setting the lower edge at each x
    hi_driver: np.ndarray    # variant name setting the upper edge at each x
    variants: tuple

    def drivers(self) -> Dict[str, str]:
        """Which variant sets each edge most often; goes into the caption."""
        def mode(a):
            vals, n = np.unique(a, return_counts=True)
            return str(vals[n.argmax()])
        return {"lower": mode(self.lo_driver), "upper": mode(self.hi_driver)}


def envelope(variants: Mapping[str, Sequence[float]]) -> Envelope:
    """Pointwise min/max across named variants of the same quantity on the same x grid."""
    if len(variants) < 2:
        raise ValueError("a model range needs at least two variants (both blocks, or a block and a WARN'd sensitivity)")
    names = tuple(variants)
    M = np.vstack([np.asarray(variants[n], float) for n in names])
    if not np.isfinite(M).all():
        raise ValueError("variants contain non-finite values")
    lo_i, hi_i = M.argmin(axis=0), M.argmax(axis=0)
    return Envelope(M.min(axis=0), M.max(axis=0), np.array(names)[lo_i], np.array(names)[hi_i], names)


def two_layer_band(ax, x, center, inner_lo, inner_hi, outer: Envelope, label: str,
                   inner_label: str = "Within the main specification",
                   outer_label: str = "Across models and assumptions", legend_loc: str = "lower right"):
    """Draw outer envelope (tint + hatch), inner band, central line, a direct end label,
    and a legend keying the two layers (identity is never left to shading alone).

    The outer layer is widened wherever it fails to contain the inner one, so the
    picture never shows model uncertainty smaller than within-spec uncertainty.
    """
    center, inner_lo, inner_hi = (np.asarray(a, float) for a in (center, inner_lo, inner_hi))
    if not (inner_lo <= center).all() or not (center <= inner_hi).all():
        raise ValueError("central line must lie inside the inner band")
    olo, ohi = np.minimum(outer.lo, inner_lo), np.maximum(outer.hi, inner_hi)
    ax.fill_between(x, olo, ohi, facecolor=style.OUTER, edgecolor=style.HATCH, hatch="////",
                    linewidth=0, label=outer_label, zorder=1)
    ax.fill_between(x, inner_lo, inner_hi, color=style.INNER, linewidth=0, label=inner_label, zorder=2)
    ax.plot(x, center, color=style.CENTER, zorder=3, label=label)
    ax.annotate(label, (x[-1], center[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=style.INK)
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, l in enumerate(labels) if l in (inner_label, outer_label)]
    ax.legend([handles[i] for i in keep], [labels[i] for i in keep], loc=legend_loc)
    return ax
