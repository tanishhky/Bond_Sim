"""Endogenous fiscal risk premium: the yield's reaction to the debt path.

This is the load-bearing thesis assumption (README Phase 4, PARAMETERS.md
P-01/P-02). Three interchangeable models behind one interface; the simulator
does not care which is plugged in, so the paper can report the conclusion
under each and show how much depends on the choice.

    premium_pct = f(debt_gdp_pct)   added to the base 10y yield each quarter

* ``NoPremium``        : the null. Rates never react to debt. If the loop still
                         appears, it is pure arithmetic (r > g).
* ``LinearPremium``    : bps per point of debt/GDP above an anchor. The slope is
                         the one number the whole result hinges on; the default
                         is a placeholder pending Tanishk's calibration call.
* ``ThresholdPremium`` : linear below a kink, steeper above it (Reinhart-Rogoff
                         style nonlinearity). Kink and extra slope are placeholders.

All units: debt_gdp in percent of GDP, premium returned in percentage points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import RiskPremiumConfig


class RiskPremiumModel(Protocol):
    def premium(self, debt_gdp_pct: np.ndarray) -> np.ndarray: ...
    def describe(self) -> str: ...


@dataclass(frozen=True)
class NoPremium:
    def premium(self, debt_gdp_pct: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(debt_gdp_pct, dtype=float))

    def describe(self) -> str:
        return "none (rates do not react to debt)"


@dataclass(frozen=True)
class LinearPremium:
    bps_per_pct: float          # TODO(Tanishk) P-01
    anchor_pct: float = 100.0

    def premium(self, debt_gdp_pct: np.ndarray) -> np.ndarray:
        excess = np.maximum(np.asarray(debt_gdp_pct, dtype=float) - self.anchor_pct, 0.0)
        return excess * self.bps_per_pct / 100.0            # bps -> pct points

    def describe(self) -> str:
        return f"linear: {self.bps_per_pct} bps per pt of debt/GDP above {self.anchor_pct}%"


@dataclass(frozen=True)
class ThresholdPremium:
    bps_per_pct: float          # TODO(Tanishk) P-01 (below the kink)
    anchor_pct: float
    threshold_pct: float        # TODO(Tanishk) P-02
    extra_bps_per_pct: float    # TODO(Tanishk) P-02 (added slope above the kink)

    def premium(self, debt_gdp_pct: np.ndarray) -> np.ndarray:
        d = np.asarray(debt_gdp_pct, dtype=float)
        base = np.maximum(d - self.anchor_pct, 0.0) * self.bps_per_pct
        kink = np.maximum(d - self.threshold_pct, 0.0) * self.extra_bps_per_pct
        return (base + kink) / 100.0

    def describe(self) -> str:
        return (f"threshold: {self.bps_per_pct} bps/pt above {self.anchor_pct}%, "
                f"+{self.extra_bps_per_pct} bps/pt above {self.threshold_pct}%")


def premium_from_config(cfg: RiskPremiumConfig) -> RiskPremiumModel:
    if cfg.model == "none":
        return NoPremium()
    if cfg.model == "linear":
        return LinearPremium(cfg.bps_per_pct_debt_gdp, cfg.anchor_debt_gdp_pct)
    if cfg.model == "threshold":
        return ThresholdPremium(cfg.bps_per_pct_debt_gdp, cfg.anchor_debt_gdp_pct,
                                cfg.threshold_debt_gdp_pct, cfg.threshold_extra_bps_per_pct)
    raise ValueError(f"unknown risk premium model {cfg.model!r}")
