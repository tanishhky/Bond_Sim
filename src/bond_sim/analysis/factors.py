"""Block factors: reduce each behavioral block to one interpretable state variable (docs/math.md 5).

Within a block every transformed, standardized series is

    x_i,t = lambda_i * F_b,t + u_i,t

with F_b the first principal component, sign-aligned so the block's anchor
loads with the stated sign (high factor = expansionary / loose / strong /
healthy), and u_i the idiosyncratic residual whose block covariance is kept.
The factor innovation is the systematic shock; u_i is the imperfect-
correlation shock at the series level; both are drawn by bootstrap later.

The quarterly block panel is built from the monthly macro panel plus the
derived fiscal series (trailing-4Q primary balance, debt/GDP, interest/GDP)
so the fiscal block carries the objects the debt engine actually uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import obs

# (series, transform). Transforms: level | log_diff (quarterly log change x100) | diff
BLOCKS: Dict[str, dict] = {
    "consumer": {"anchor": ("PCE", +1), "series": [("PCE", "log_diff"), ("DSPIC96", "log_diff"), ("PSAVERT", "level"),
                                                   ("TOTALSL", "log_diff"), ("REVOLSL", "log_diff"), ("UMCSENT", "level")]},
    "corporate": {"anchor": ("PNFI", +1), "series": [("PNFI", "log_diff"), ("CP", "log_diff"), ("BUSLOANS", "log_diff"),
                                                     ("INDPRO", "log_diff"), ("TCU", "level"), ("NCBDBIQ027S", "log_diff"),
                                                     ("GDP", "log_diff"), ("GDPC1", "log_diff")]},
    "financial": {"anchor": ("NFCI", -1), "series": [("NFCI", "level"), ("BAA10Y", "level"), ("VIXCLS", "level"),
                                                     ("T10Y2Y", "level"), ("DRTSCILM", "level")]},        # high = loose
    "fiscal": {"anchor": ("pb", +1), "series": [("pb", "level"), ("debt_gdp", "diff"), ("int_gdp", "level"),
                                                ("receipts_4q", "log_diff"), ("foreign_share", "diff")]},   # high = healthier
    "labor": {"anchor": ("UNRATE", -1), "series": [("UNRATE", "log"), ("PAYEMS", "log_diff"), ("ICSA", "log_diff")]},  # high = strong
    "rates": {"anchor": ("DGS10", +1), "series": [("DGS10", "level"), ("DGS3MO", "level"), ("DGS2", "level"),
                                                  ("DGS30", "level"), ("THREEFYTP10", "level")]},
    "prices": {"anchor": ("CPIAUCSL", +1), "series": [("CPIAUCSL", "log_diff"), ("CPILFESL", "log_diff"),
                                                      ("PCEPI", "log_diff"), ("MEDCPIM158SFRBCLE", "level")]},
}
# What the debt engine needs back from the factors, and how to undo the transform.
TARGETS: Dict[str, Tuple[str, str]] = {          # engine variable -> (series, transform)
    "r10": ("DGS10", "level"), "r3m": ("DGS3MO", "level"), "g_nom": ("GDP", "log_diff"),
    "pb": ("pb", "level"), "u": ("UNRATE", "log"), "infl": ("CPIAUCSL", "log_diff"),
}
FLOW_SERIES = ("MTSDS133FMS", "MTSR133FMS", "MTSO133FMS")


def build_block_panel(panel_wide: pd.DataFrame, interest_monthly: Optional[pd.Series] = None) -> pd.DataFrame:
    """Monthly date x series -> quarterly panel with the derived fiscal columns.
    ``interest_monthly`` ($mm, the book's bottom-up interest) replaces NIPA
    interest in the primary balance when given (see macro.build_quarterly_state)."""
    w = panel_wide.copy()
    q = w.resample("QS").mean()
    for s in FLOW_SERIES:
        if s in w.columns:
            q[s] = w[s].resample("QS").sum(min_count=3)
    if "GDP" in q and "MTSDS133FMS" in q:
        if interest_monthly is not None:
            interest_q = interest_monthly.resample("QS").sum(min_count=3) / 1e3
        elif "A091RC1Q027SBEA" in q:
            interest_q = q["A091RC1Q027SBEA"] / 4.0
        else:
            interest_q = None
        if interest_q is not None:
            primary_q = q["MTSDS133FMS"] / 1e3 + interest_q                        # $bn per quarter
            gdp_q = q["GDP"] / 4.0
            q["pb"] = 100.0 * primary_q.rolling(4).sum() / gdp_q.rolling(4).sum()
            q["int_gdp"] = 100.0 * interest_q.rolling(4).sum() / gdp_q.rolling(4).sum()
    if "FYGFGDQ188S" in q:
        q["debt_gdp"] = q["FYGFGDQ188S"]
    if "MTSR133FMS" in q:
        q["receipts_4q"] = q["MTSR133FMS"].rolling(4).sum()
    if "FDHBFIN" in q and "FYGFDPUN" in q:
        q["foreign_share"] = 100.0 * q["FDHBFIN"] * 1e3 / q["FYGFDPUN"]           # $bn -> $mm
    return q


def transform(series: pd.Series, how: str) -> pd.Series:
    if how == "level":
        return series
    if how == "log":                                   # positive level series (unemployment): floor is natural
        return np.log(series.where(series > 0))
    if how == "diff":
        return series.diff()
    if how == "log_diff":
        return 100.0 * np.log(series.where(series > 0)).diff()
    raise ValueError(how)


@dataclass
class BlockFit:
    name: str
    columns: List[str]
    transforms: List[str]
    mean: np.ndarray             # per-series mean of the transformed data
    std: np.ndarray
    loadings: np.ndarray         # lambda_i, unit-variance factor
    explained: float
    factor: pd.Series            # (T,) unit variance over the fit sample
    residuals: pd.DataFrame      # (T, n) idiosyncratic, standardized units
    resid_cov: np.ndarray
    idio_rho: np.ndarray         # (n,) AR(1) persistence of each idiosyncratic residual
    idio_innov: np.ndarray       # (T-1, n) innovations of the idiosyncratic AR(1), the bootstrap bank


class BlockFactors:
    def __init__(self, blocks: Dict[str, dict] = BLOCKS, start: Optional[str] = None):
        self.blocks = blocks
        self.start = start
        self.fits: Dict[str, BlockFit] = {}

    # ── estimation ──────────────────────────────────────────────────────────
    def fit(self, q: pd.DataFrame) -> "BlockFactors":
        for name, spec in self.blocks.items():
            cols, hows = zip(*[(s, h) for s, h in spec["series"] if s in q.columns])
            X = pd.concat([transform(q[c], h).rename(c) for c, h in zip(cols, hows)], axis=1)
            if self.start:
                X = X[X.index >= pd.Timestamp(self.start)]
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            mean, std = X.mean().to_numpy(), X.std(ddof=0).to_numpy()
            Z = (X.to_numpy() - mean) / std
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            f = Z @ Vt[0]
            f = f / f.std(ddof=0)
            lam = Z.T @ f / len(f)                                  # regression of each z_i on unit-variance f
            anchor, sign = spec["anchor"]
            if anchor in cols and lam[list(cols).index(anchor)] * sign < 0:
                f, lam = -f, -lam
            resid = Z - np.outer(f, lam)
            # Idiosyncratic residuals of level series are persistent (unemployment does
            # not jump a point a quarter around its factor); each gets an AR(1) with
            # its innovations banked for bootstrap, so reconstructed series keep the
            # persistence and the innovation size the data show.
            num = (resid[1:] * resid[:-1]).sum(axis=0)
            den = (resid[:-1] ** 2).sum(axis=0)
            rho = np.clip(np.where(den > 0, num / np.maximum(den, 1e-12), 0.0), -0.98, 0.98)
            innov = resid[1:] - resid[:-1] * rho
            self.fits[name] = BlockFit(name, list(cols), list(hows), mean, std, lam,
                                       float(S[0] ** 2 / (S ** 2).sum()), pd.Series(f, index=X.index, name=name),
                                       pd.DataFrame(resid, index=X.index, columns=cols), np.cov(resid.T), rho, innov)
            obs.event(channel="analysis", kind="factors.block", block=name, n=len(X), n_series=len(cols),
                      explained=round(self.fits[name].explained, 3),
                      loadings={c: round(float(l), 3) for c, l in zip(cols, lam)})
        return self

    def factors(self) -> pd.DataFrame:
        """(T, B) factor panel on the common sample."""
        return pd.concat([f.factor for f in self.fits.values()], axis=1).dropna()

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({b: {"n_series": len(f.columns), "explained_var": f.explained, "n_obs": len(f.factor),
                                 "start": f.factor.index.min().date()} for b, f in self.fits.items()}).T

    # ── back to series and engine units ─────────────────────────────────────
    def _locate(self, series: str) -> Tuple[str, int]:
        for b, f in self.fits.items():
            if series in f.columns:
                return b, f.columns.index(series)
        raise KeyError(series)

    def target_series(self) -> Dict[str, Tuple[str, int]]:
        """engine variable -> (block, column index) for every target present."""
        out = {}
        for var, (series, _) in TARGETS.items():
            try:
                out[var] = self._locate(series)
            except KeyError:
                continue
        return out

    def idio_start(self) -> Dict[str, np.ndarray]:
        """Last observed idiosyncratic residual of each target (standardized units)."""
        return {var: np.array([self.fits[b].residuals.iloc[-1, j]]) for var, (b, j) in self.target_series().items()}

    def idio_step(self, idio: Dict[str, np.ndarray], u_draw: np.ndarray) -> Dict[str, np.ndarray]:
        """One AR(1) step for every target's idiosyncratic residual: rho x previous +
        a bootstrapped innovation. ``u_draw`` (K,) uniforms pick the innovation row."""
        out = {}
        for var, (b, j) in self.target_series().items():
            f = self.fits[b]
            idx = (u_draw * len(f.idio_innov)).astype(int)
            out[var] = f.idio_rho[j] * idio[var] + f.idio_innov[idx, j]
        return out

    def engine_targets(self, F: np.ndarray, block_index: Dict[str, int],
                       idio: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Engine variables in natural units from (K, B) factors and (K,) idiosyncratic
        residuals: rates/u/pb in pct, g_nom and infl annualized pct from quarterly
        log changes."""
        out = {}
        for var, (b, j) in self.target_series().items():
            f = self.fits[b]
            z = f.loadings[j] * F[:, block_index[b]]
            if idio is not None and var in idio:
                z = z + idio[var]
            x = f.mean[j] + f.std[j] * z
            how = TARGETS[var][1]
            if how == "log_diff":
                out[var] = 100.0 * (np.exp(4.0 * x / 100.0) - 1.0)
            elif how == "log":
                out[var] = np.exp(x)
            else:
                out[var] = x
        return out

    def last_factors(self) -> np.ndarray:
        return self.factors().iloc[-1].to_numpy()

    def block_index(self) -> Dict[str, int]:
        return {b: i for i, b in enumerate(self.fits)}
