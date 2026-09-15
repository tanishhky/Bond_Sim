"""Admissibility filter: reject impossible paths, never clip (docs/math.md 7).

A rule is a hard feasibility statement about the world, not a modeling
preference. Every run reports how many paths each rule removed; results are
computed on admissible paths only, and both numbers go in the paper.

One structural floor is applied inside the macro blocks rather than here:
nominal yields cannot fall meaningfully below zero while cash exists (the
effective lower bound). Everything else is rejected, not clipped.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

CORE = ("u", "r10", "r3m", "g_nom", "gdp_mm", "debt_mm", "debt_gdp")   # must be finite on every path
# Inflation is only checked when the macro block supplies it (the VAR block does not, P-15).

RULES: Dict[str, Callable[[Dict[str, np.ndarray]], np.ndarray]] = {
    "nonfinite":          lambda p: ~np.all([np.isfinite(p[k]).all(axis=1) for k in CORE if k in p], axis=0),
    "gdp_nonpositive":    lambda p: (p["gdp_mm"] <= 0).any(axis=1) if "gdp_mm" in p else np.zeros(len(next(iter(p.values()))), bool),
    "u_outside_1_35":     lambda p: ((p["u"] < 1.0) | (p["u"] > 35.0)).any(axis=1),
    "yield_outside_0_30": lambda p: ((p["r10"] < 0.0) | (p["r10"] > 30.0) | (p["r3m"] < 0.0) | (p["r3m"] > 30.0)).any(axis=1),
    "infl_outside_-15_50": lambda p: ((p["infl"] < -15.0) | (p["infl"] > 50.0)).any(axis=1) if "infl" in p and np.isfinite(p["infl"]).any() else np.zeros(p["u"].shape[0], bool),
    "growth_outside_-30_40": lambda p: ((p["g_nom"] < -30.0) | (p["g_nom"] > 40.0)).any(axis=1),
    "debt_negative":      lambda p: (p["debt_mm"] < 0).any(axis=1) if "debt_mm" in p else np.zeros(p["u"].shape[0], bool),
}


def evaluate(paths: Dict[str, np.ndarray]) -> Tuple[np.ndarray, pd.Series]:
    """(admissible mask (K,), rejections per rule). A path failing several
    rules is counted under each; the mask is the union."""
    K = next(iter(paths.values())).shape[0]
    bad = np.zeros(K, bool)
    counts = {}
    for name, rule in RULES.items():
        with np.errstate(invalid="ignore"):
            hit = np.asarray(rule(paths), bool)
        counts[name] = int(hit.sum())
        bad |= hit
    return ~bad, pd.Series(counts, name="rejected_paths")
