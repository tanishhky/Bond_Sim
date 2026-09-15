"""Correlation structure of the macro panel: static, rolling, regime-split, lead/lag, causal.

The rule this module enforces (decision 0003): no two model inputs are ever
treated as independent by default. For every ordered pair of series the
analyzer produces

    full_corr        : full-sample Pearson correlation of the transformed series
    rolling          : rolling-window correlation (window from config)
    range            : max - min of the rolling series; a wide range means the
                       relationship is regime-dependent and must not be
                       summarized by one number (feeds RegimeDetector)
    lead_lag         : corr(x_t, y_{t+lag}) for lag in [-L, L]; a positive best
                       lag means x leads y
    granger          : x -> y and y -> x p-values at a BIC-chosen VAR lag, with
                       Benjamini-Hochberg control across the whole grid so the
                       "significant" flags survive multiple testing

Transforms: differences by default (levels of debt, employment, and rates are
integrated; correlating levels manufactures spurious relationships).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import obs
from ..config import CorrelationConfig


@dataclass
class PairResult:
    a: str
    b: str
    n: int
    full_corr: float
    rolling: pd.Series
    rolling_range: float
    lead_lag: pd.Series                   # index: lag (months), value: corr(a_t, b_{t+lag})
    best_lag: int
    best_lag_corr: float
    granger_a_to_b_p: float = np.nan
    granger_b_to_a_p: float = np.nan
    granger_lag: int = 0
    regime_corr: Dict[int, float] = field(default_factory=dict)

    @property
    def is_regime_dependent(self) -> bool:
        """Rolling correlation swings more than 0.5 over the sample."""
        return bool(self.rolling_range > 0.5)

    @property
    def leads(self) -> Optional[str]:
        """Which series leads, if the best lag is not zero."""
        if self.best_lag > 0:
            return self.a
        if self.best_lag < 0:
            return self.b
        return None


class CorrelationAnalyzer:
    """Analyze a date x series panel (wide frame, monthly index)."""

    def __init__(self, panel: pd.DataFrame, cfg: CorrelationConfig, freq: str = "M"):
        """``freq`` is the panel's sampling frequency ("M" or "Q"); config windows
        and lags are stated in months and scaled to periods here. A mixed
        monthly/quarterly panel must be analyzed quarterly: forward-filled
        quarterly series have zero monthly differences two months in three,
        which manufactures spurious zero correlations."""
        self.raw = panel.sort_index()
        self.cfg = cfg
        scale = {"M": 1, "Q": 3}[freq]
        self.window = max(cfg.rolling_window_months // scale, 8)
        self.L = max(cfg.max_lag_months // scale, 4)
        self.X = self._transform(self.raw, cfg.transform)

    # ── preparation ─────────────────────────────────────────────────────────
    @staticmethod
    def _transform(df: pd.DataFrame, how: str) -> pd.DataFrame:
        if how == "level":
            return df.copy()
        if how == "diff":
            return df.diff()
        if how == "pct":
            return df.pct_change(fill_method=None)
        raise ValueError(f"unknown transform {how!r}")

    def _pair(self, a: str, b: str) -> pd.DataFrame:
        return self.X[[a, b]].dropna()

    # ── single-pair statistics ──────────────────────────────────────────────
    def rolling(self, a: str, b: str) -> pd.Series:
        p = self._pair(a, b)
        return p[a].rolling(self.window).corr(p[b]).rename(f"rolling_corr[{a},{b}]")

    def lead_lag(self, a: str, b: str) -> pd.Series:
        """corr(a_t, b_{t+lag}). Vectorized over lags with shift; each lag uses
        the overlapping sample only."""
        p = self._pair(a, b)
        L = self.L
        lags = np.arange(-L, L + 1)
        vals = [p[a].corr(p[b].shift(-int(k))) for k in lags]   # b shifted back by k = b_{t+k}
        return pd.Series(vals, index=pd.Index(lags, name="lag"), name=f"xcorr[{a}->{b}]")

    def granger(self, a: str, b: str, max_lag: int = 12) -> Tuple[float, float, int]:
        """p-values for a -> b and b -> a at the BIC-selected VAR order (capped).
        Uses statsmodels' VAR for order selection and its Granger test."""
        from statsmodels.tsa.api import VAR
        p = self._pair(a, b)
        if len(p) < 5 * max_lag:
            return np.nan, np.nan, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(p.to_numpy())
            try:
                order = int(model.select_order(maxlags=max_lag).bic)
            except Exception:  # noqa: BLE001 - singular designs on short samples
                order = 1
            order = max(order, 1)
            res = model.fit(order)
            p_ab = float(res.test_causality(caused=1, causing=0).pvalue)   # a (col 0) -> b (col 1)
            p_ba = float(res.test_causality(caused=0, causing=1).pvalue)
        return p_ab, p_ba, order

    def analyze_pair(self, a: str, b: str, regimes: Optional[pd.Series] = None,
                     with_granger: bool = True) -> PairResult:
        p = self._pair(a, b)
        roll = self.rolling(a, b)
        xc = self.lead_lag(a, b)
        best = int(xc.abs().idxmax())
        res = PairResult(a=a, b=b, n=len(p), full_corr=float(p[a].corr(p[b])), rolling=roll,
                         rolling_range=float(roll.max() - roll.min()) if roll.notna().any() else np.nan,
                         lead_lag=xc, best_lag=best, best_lag_corr=float(xc.loc[best]))
        if with_granger:
            res.granger_a_to_b_p, res.granger_b_to_a_p, res.granger_lag = self.granger(a, b)
        if regimes is not None:
            r = regimes.reindex(p.index)
            res.regime_corr = {int(k): float(g[a].corr(g[b])) for k, g in p.groupby(r) if len(g) > 10}
        return res

    # ── whole-panel views ───────────────────────────────────────────────────
    def full_matrix(self) -> pd.DataFrame:
        return self.X.corr()

    def report(self, columns: Optional[List[str]] = None, regimes: Optional[pd.Series] = None,
               with_granger: bool = True, group_of: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """One row per unordered pair with every statistic, plus BH-FDR flags on
        the Granger grid (both directions pooled into one family of tests).
        ``group_of`` restricts to cross-group pairs (e.g. fiscal x labor)."""
        cols = columns or [c for c in self.X.columns if self.X[c].notna().sum() > 2 * self.window]
        pairs = [(a, b) for a, b in combinations(cols, 2)
                 if group_of is None or group_of.get(a) != group_of.get(b)]
        rows = []
        with obs.timed(channel="analysis", kind="correlation.report", n_series=len(cols), n_pairs=len(pairs)):
            for a, b in pairs:
                r = self.analyze_pair(a, b, regimes=regimes, with_granger=with_granger)
                rows.append({
                    "a": a, "b": b, "n": r.n, "full_corr": r.full_corr,
                    "rolling_min": float(r.rolling.min()), "rolling_max": float(r.rolling.max()),
                    "rolling_range": r.rolling_range, "regime_dependent": r.is_regime_dependent,
                    "best_lag": r.best_lag, "best_lag_corr": r.best_lag_corr, "leads": r.leads,
                    "granger_a_to_b_p": r.granger_a_to_b_p, "granger_b_to_a_p": r.granger_b_to_a_p,
                    "granger_lag": r.granger_lag,
                    **{f"corr_regime_{k}": v for k, v in r.regime_corr.items()},
                })
        rep = pd.DataFrame(rows)
        if with_granger and len(rep):
            pv = pd.concat([rep["granger_a_to_b_p"], rep["granger_b_to_a_p"]]).to_numpy()
            sig = benjamini_hochberg(pv, self.cfg.fdr_alpha)
            n = len(rep)
            rep["granger_a_to_b_sig"] = sig[:n]
            rep["granger_b_to_a_sig"] = sig[n:]
        return rep.sort_values("rolling_range", ascending=False).reset_index(drop=True)


def benjamini_hochberg(pvals: np.ndarray, alpha: float) -> np.ndarray:
    """Boolean mask of discoveries under BH-FDR at level alpha (NaNs never significant)."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.zeros(len(p), bool)
    if ok.sum() == 0:
        return out
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = q[order] <= thresh
    if passed.any():
        k = np.max(np.nonzero(passed)[0])
        sig = np.zeros(m, bool)
        sig[order[: k + 1]] = True
        out[np.nonzero(ok)[0]] = sig
    return out
