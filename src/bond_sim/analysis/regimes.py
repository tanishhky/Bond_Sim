"""Regimes: when a relationship is not one number, find the states it lives in.

Pipeline (decision 0003, step 2):
1. A rolling correlation with a wide range is regime-dependent. Label its
   history into ``n_regimes`` states, either by quantile cuts of a driver
   series (transparent, no fitting) or by a Gaussian HMM (captures persistence).
2. Test whether those regimes are a market-wide phenomenon: if the label
   sequences derived from many different pairs agree in time (adjusted Rand
   index well above 0), one common state variable is driving all of them and
   the simulator should carry a single regime process. If they do not agree,
   the regimes are pair-specific and each pair keeps its own.
3. Test lead/lag between regime indicators: if series A's high-correlation
   regime switches on before B's, A is the leading indicator of the state
   change and the simulator's shock ordering should respect that.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import CorrelationConfig


class RegimeDetector:
    def __init__(self, cfg: CorrelationConfig):
        self.cfg = cfg

    # ── labeling ────────────────────────────────────────────────────────────
    def label(self, driver: pd.Series, method: Optional[str] = None) -> pd.Series:
        """Integer regime labels 0..n-1 ordered by the driver's level (0 = lowest)."""
        method = method or self.cfg.regime_method
        d = driver.dropna()
        if method == "quantile":
            lab = pd.qcut(d, self.cfg.n_regimes, labels=False, duplicates="drop")
        elif method == "hmm":
            lab = self._hmm(d)
        else:
            raise ValueError(f"unknown regime method {method!r}")
        return pd.Series(lab, index=d.index, name=f"regime[{driver.name}]").astype(int)

    def _hmm(self, d: pd.Series) -> np.ndarray:
        from hmmlearn.hmm import GaussianHMM   # optional dependency (extras: regimes)
        x = d.to_numpy().reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=self.cfg.n_regimes, covariance_type="full",
                            n_iter=200, random_state=0).fit(x)
            raw = m.predict(x)
        # Relabel so state 0 has the lowest mean: labels are comparable across pairs.
        order = np.argsort(m.means_.ravel())
        remap = {int(s): int(r) for r, s in enumerate(order)}
        return np.vectorize(remap.get)(raw)

    # ── consistency across the market ───────────────────────────────────────
    def consistency(self, labels: Dict[str, pd.Series]) -> pd.DataFrame:
        """Pairwise adjusted Rand index between label sequences (aligned on the
        common index). ARI ~ 0 means independent regime timing; ARI near 1 means
        the same regimes. The mean off-diagonal is the market-consistency score."""
        keys = list(labels)
        out = pd.DataFrame(np.eye(len(keys)), index=keys, columns=keys)
        for i, j in combinations(range(len(keys)), 2):
            a, b = labels[keys[i]].align(labels[keys[j]], join="inner")
            ari = adjusted_rand_index(a.to_numpy(), b.to_numpy()) if len(a) > 10 else np.nan
            out.iat[i, j] = out.iat[j, i] = ari
        return out

    # ── regime lead/lag ─────────────────────────────────────────────────────
    def regime_lead_lag(self, labels_a: pd.Series, labels_b: pd.Series, high: Optional[int] = None) -> pd.Series:
        """Cross-correlation of the "in high regime" indicators at lags -L..L.
        Positive best lag: A's regime switch precedes B's."""
        high = self.cfg.n_regimes - 1 if high is None else high
        a, b = labels_a.align(labels_b, join="inner")
        ia, ib = (a == high).astype(float), (b == high).astype(float)
        L = self.cfg.max_lag_months
        lags = np.arange(-L, L + 1)
        vals = [ia.corr(ib.shift(-int(k))) for k in lags]
        return pd.Series(vals, index=pd.Index(lags, name="lag"), name="regime_xcorr")


def adjusted_rand_index(x: np.ndarray, y: np.ndarray) -> float:
    """Hubert-Arabie adjusted Rand index from the contingency table (no sklearn)."""
    x = np.asarray(x); y = np.asarray(y)
    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)
    n = len(x)
    ct = np.zeros((len(xs), len(ys)), dtype=np.int64)
    np.add.at(ct, (xi, yi), 1)

    def comb2(v):
        v = np.asarray(v, dtype=np.float64)
        return (v * (v - 1) / 2.0).sum()

    sum_ij = comb2(ct.ravel())
    sum_a = comb2(ct.sum(1))
    sum_b = comb2(ct.sum(0))
    total = n * (n - 1) / 2.0
    expected = sum_a * sum_b / total if total else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    return float((sum_ij - expected) / denom) if denom else 1.0
