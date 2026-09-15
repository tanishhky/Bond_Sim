"""Sector employment responses, estimated from history (Phase 5).

For each sector series, quarterly log employment growth is regressed on the
change in unemployment and on lagged changes in the 10y yield:

    dlog(E_s,t) = a_s + b_s * d_u_t + sum_l c_s,l * d_r10_{t-l} + e_s,t

so a simulated path of (d_u, d_r10) maps to a sector employment path. The
residual covariance across sectors is kept, so sector shocks are drawn
together (a downturn hits construction and credit intermediation at once, as
it does in the data). ``beta_source="literature"`` is the documented
alternative (P-05) when a sector's fit is too weak to trust.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import obs


@dataclass
class SectorFit:
    sector: str
    beta_u: float
    beta_r: np.ndarray          # (L+1,) lags 0..L on d_r10
    intercept: float
    r2: float
    n: int


class LaborModel:
    def __init__(self, sectors: List[str], rate_lags: int = 4):
        self.sectors = list(sectors)
        self.L = rate_lags
        self.fits: Dict[str, SectorFit] = {}
        self.resid_cov: Optional[np.ndarray] = None
        self.chol: Optional[np.ndarray] = None

    def fit(self, panel_wide: pd.DataFrame, q_state: pd.DataFrame) -> "LaborModel":
        """panel_wide: monthly date x series (employment levels); q_state: the
        quarterly VAR state frame (needs d_u, d_r10)."""
        resid = {}
        for s in self.sectors:
            e = panel_wide[s].resample("QS").mean()
            y = 100.0 * np.log(e.where(e > 0)).diff()
            X = pd.DataFrame({"d_u": q_state["d_u"]})
            for l in range(self.L + 1):
                X[f"d_r10_l{l}"] = q_state["d_r10"].shift(l)
            df = pd.concat([y.rename("y"), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < 40:
                obs.event(channel="sim", kind="labor.skip", level="WARNING", sector=s, n=len(df))
                continue
            A = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
            coef, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)
            pred = A @ coef
            r2 = 1.0 - np.var(df["y"] - pred) / np.var(df["y"])
            self.fits[s] = SectorFit(s, float(coef[1]), coef[2:], float(coef[0]), float(r2), len(df))
            resid[s] = pd.Series(df["y"].to_numpy() - pred, index=df.index)
        R = pd.DataFrame(resid).dropna()
        self.resid_cov = R.cov().to_numpy()
        self.chol = np.linalg.cholesky(self.resid_cov + 1e-12 * np.eye(len(R.columns)))
        self.sectors = list(R.columns)
        obs.event(channel="sim", kind="labor.fit", sectors={s: round(f.r2, 3) for s, f in self.fits.items()})
        return self

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({s: {"beta_u": f.beta_u, "beta_r_sum": float(f.beta_r.sum()),
                                 "r2": f.r2, "n": f.n} for s, f in self.fits.items()}).T

    def simulate(self, d_u: np.ndarray, d_r10: np.ndarray, rng: np.random.Generator,
                 e0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """d_u, d_r10: (K, HQ) quarterly paths -> sector employment index paths
        (K, HQ), base 100 at the start (or ``e0`` levels)."""
        K, HQ = d_u.shape
        S = len(self.sectors)
        z = rng.standard_normal((K, HQ, S)) @ self.chol.T
        out = {}
        for j, s in enumerate(self.sectors):
            f = self.fits[s]
            g = np.full((K, HQ), f.intercept) + f.beta_u * d_u
            for l, c in enumerate(f.beta_r):
                lagged = np.concatenate([np.zeros((K, l)), d_r10[:, :HQ - l]], axis=1) if l else d_r10
                g += c * lagged
            g += z[:, :, j]
            level = (100.0 if e0 is None else e0[j]) * np.exp(np.cumsum(g, axis=1) / 100.0)
            out[s] = level
        return out
