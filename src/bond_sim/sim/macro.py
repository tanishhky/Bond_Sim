"""Quarterly fiscal-macro VAR: the empirically estimated joint dynamics.

Why a VAR (decision 0005): the thesis needs rates, growth, the primary
balance, and unemployment to move together the way they historically have,
including who leads whom. A VAR(p) estimated on the real quarterly panel gives
(a) the cross-variable response coefficients and (b) the residual covariance
matrix, so simulated shocks are correlated exactly as observed. No dynamic
parameter in this block is hand-set.

State vector (quarterly, all in LEVELS):
    r10     10y Treasury yield, pct              (base rate, ex fiscal premium)
    r3m     3m bill yield, pct
    g_nom   nominal GDP growth, annualized pct
    pb      primary balance, trailing four quarters, pct of GDP
    u       unemployment rate, pct

Levels, not differences, for rates and unemployment: a VAR in differences
has no level anchor, so a 30-year simulation random-walks the 10y and the
unemployment rate into their floors and the medians become artifacts of
the clipping. In levels the estimated autoregressive coefficients supply
mean reversion toward the sample means, which are reported and can be
overridden (anchors, P-03/P-13/P-14) as explicit views.

The fiscal premium lives outside the VAR (premium.py). Its effect on the
real economy is transmitted with the VAR's own estimated coefficients on
the lagged 10y (``rate_transmission``), so "higher yields hurt growth,
the primary balance, and employment" has the magnitudes the data show,
while the base rate keeps its estimated mean reversion.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import obs

VARS = ("r10", "r3m", "g_nom", "pb", "u")
RATE_VARS = ("r10", "r3m")


@dataclass(frozen=True)
class MacroSpec:
    """Column names in the monthly macro panel used to build the quarterly state."""
    r10: str = "DGS10"
    r3m: str = "DGS3MO"
    gdp: str = "GDP"                     # nominal, SAAR $bn (quarterly, ffilled monthly)
    deficit: str = "MTSDS133FMS"         # monthly surplus(+)/deficit(-), $mm, not seasonally adjusted
    interest: str = "A091RC1Q027SBEA"    # federal interest payments, SAAR $bn (quarterly)
    unrate: str = "UNRATE"


def build_quarterly_state(panel_wide: pd.DataFrame, spec: MacroSpec = MacroSpec(),
                          interest_monthly: Optional[pd.Series] = None) -> pd.DataFrame:
    """Monthly panel (date x series) -> quarterly frame with VARS columns plus
    the differenced helpers (d_r10, d_u) the labor model uses.

    Rates and unemployment are quarter averages. The MTS deficit is not
    seasonally adjusted (April is a surplus month every year), so the primary
    balance is a trailing-four-quarter ratio: sum of (surplus + interest) over
    the last four quarters divided by GDP over the same four quarters.

    ``interest_monthly``: monthly interest in $ millions (the bond book's
    bottom-up cash interest on marketable debt). When given it replaces the
    NIPA interest series, so the primary balance, the effective rate, and
    the simulator all use the same interest concept. NIPA "interest payments"
    is a gross figure and overstates the primary balance by about a point of
    GDP.
    """
    p = panel_wide.copy()
    if interest_monthly is not None:
        interest_q = interest_monthly.resample("QS").sum(min_count=3) / 1e3          # $mm -> $bn per quarter
    else:
        interest_q = p[spec.interest].resample("QS").first() / 4.0                  # SAAR -> quarterly $bn
    q = pd.DataFrame({
        "r10": p[spec.r10].resample("QS").mean(),
        "r3m": p[spec.r3m].resample("QS").mean(),
        "u": p[spec.unrate].resample("QS").mean(),
        "gdp": p[spec.gdp].resample("QS").first(),                            # SAAR $bn
        "surplus_q": p[spec.deficit].resample("QS").sum(min_count=3) / 1e3,   # $mm -> $bn, quarter total
        "interest_q": interest_q,
    })
    q["g_nom"] = 100.0 * ((q["gdp"] / q["gdp"].shift(1)) ** 4 - 1.0)        # annualized q/q
    primary_q = q["surplus_q"] + q["interest_q"]
    q["pb"] = 100.0 * primary_q.rolling(4).sum() / (q["gdp"] / 4.0).rolling(4).sum()
    q["d_r10"] = q["r10"].diff()
    q["d_r3m"] = q["r3m"].diff()
    q["d_u"] = q["u"].diff()
    return q


class MacroVAR:
    """VAR(p) on VARS with BIC lag selection (capped) and residual covariance."""

    def __init__(self, max_lag: int = 4):
        self.max_lag = max_lag
        self.lag: int = 0
        self.coefs: Optional[np.ndarray] = None      # (p, V, V): coefs[l][i, j] effect of var j at lag l+1 on var i
        self.intercept: Optional[np.ndarray] = None  # (V,)
        self.cov: Optional[np.ndarray] = None        # residual covariance (V, V)
        self.chol: Optional[np.ndarray] = None
        self.sample: Optional[pd.DataFrame] = None

    # ── estimation ──────────────────────────────────────────────────────────
    # Unemployment enters the VAR in logs: a linear model on the level produced
    # sub-1% unemployment on 20-25% of 30-year paths (never observed; the US
    # minimum is 2.5%), which the admissibility filter then rejected, biasing the
    # surviving sample. log(u) keeps the floor natural. Callers see levels.
    LOG_VARS = ("u",)

    def fit(self, q: pd.DataFrame, start: Optional[str] = None) -> "MacroVAR":
        from statsmodels.tsa.api import VAR
        y = q[list(VARS)].replace([np.inf, -np.inf], np.nan).dropna()
        for v in self.LOG_VARS:
            y[v] = np.log(y[v])
        if start:
            y = y[y.index >= pd.Timestamp(start)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(y.to_numpy())
            try:
                self.lag = max(int(model.select_order(maxlags=self.max_lag).bic), 1)
            except Exception:  # noqa: BLE001
                self.lag = 1
            res = model.fit(self.lag)
        self.coefs = np.asarray(res.coefs)           # (p, V, V)
        self.intercept = np.asarray(res.intercept)   # (V,)
        self.cov = np.asarray(res.sigma_u)
        self.chol = np.linalg.cholesky(self.cov + 1e-12 * np.eye(len(VARS)))
        self.sample = y
        obs.event(channel="sim", kind="macro_var.fit", lag=self.lag, n=len(y),
                  start=str(y.index.min().date()), end=str(y.index.max().date()),
                  long_run_means=self.long_run_means().round(3).to_dict(),
                  max_eigenvalue=round(self.max_eigenvalue(), 4),
                  resid_corr=np.round(self.resid_corr().to_numpy(), 2).tolist())
        return self

    def resid_corr(self) -> pd.DataFrame:
        d = np.sqrt(np.diag(self.cov))
        return pd.DataFrame(self.cov / np.outer(d, d), index=VARS, columns=VARS)

    def max_eigenvalue(self) -> float:
        """Largest modulus eigenvalue of the companion matrix; < 1 means stable."""
        V, p = len(VARS), self.lag
        comp = np.zeros((V * p, V * p))
        comp[:V, :] = np.concatenate([self.coefs[l] for l in range(p)], axis=1)
        if p > 1:
            comp[V:, :-V] = np.eye(V * (p - 1))
        return float(np.abs(np.linalg.eigvals(comp)).max())

    # ── long-run means: the one place a view about levels is allowed ───────
    def long_run_means(self) -> pd.Series:
        """Unconditional means implied by the intercept: mu = (I - sum_l A_l)^-1 c,
        reported in levels (log variables exponentiated)."""
        V = len(VARS)
        A = self.coefs.sum(axis=0)
        mu = pd.Series(np.linalg.solve(np.eye(V) - A, self.intercept), index=VARS)
        for v in self.LOG_VARS:
            mu[v] = float(np.exp(mu[v]))
        return mu

    def set_long_run_means(self, means: Dict[str, float]) -> "MacroVAR":
        """Shift the intercept so the named variables' unconditional means equal
        ``means`` (levels) while every dynamic coefficient and the shock covariance
        stay estimated. Anchors are explicit views (P-03 primary balance, P-13
        long-run 10y, P-14 natural rate) and are logged as such."""
        V = len(VARS)
        A = self.coefs.sum(axis=0)
        mu = np.linalg.solve(np.eye(V) - A, self.intercept)
        for k, v in means.items():
            mu[self.index_of(k)] = float(np.log(v)) if k in self.LOG_VARS else float(v)
        self.intercept = (np.eye(V) - A) @ mu
        obs.event(channel="sim", kind="macro_var.long_run_means", means=dict(zip(VARS, np.round(mu, 3).tolist())))
        return self

    # ── simulation ──────────────────────────────────────────────────────────
    def draw_shocks(self, K: int, H: int, rng: np.random.Generator) -> np.ndarray:
        """(K, H, V) correlated innovations: Cholesky of the estimated covariance."""
        z = rng.standard_normal((K, H, len(VARS)))
        return z @ self.chol.T

    def step(self, history: np.ndarray, shock: np.ndarray, exog_add: Optional[np.ndarray] = None) -> np.ndarray:
        """One quarter ahead for K paths.

        history : (K, p, V) most recent p states, history[:, -1] is the latest
        shock   : (K, V) correlated innovation
        exog_add: (K, V) additive intervention (policy or premium), applied after
        """
        K = history.shape[0]
        y = np.broadcast_to(self.intercept, (K, len(VARS))).copy()
        for l in range(self.lag):
            y += history[:, -1 - l, :] @ self.coefs[l].T
        y += shock
        if exog_add is not None:
            y += exog_add
        return y

    def rate_transmission(self, delta_r10: np.ndarray) -> np.ndarray:
        """(K, V) response of the non-rate equations to a shift of ``delta_r10``
        in the lagged 10y, using the estimated coefficients summed over lags.
        Rows for r10 and r3m are zero: the base rate is not pushed by its own
        premium. This is how the fiscal premium reaches growth, the primary
        balance, and unemployment."""
        A = self.coefs.sum(axis=0)                     # (V, V)
        col = A[:, self.index_of("r10")].copy()
        for r in RATE_VARS:
            col[self.index_of(r)] = 0.0
        return np.outer(np.asarray(delta_r10, float), col)

    def credit_channel_transmission(self, delta_r10: np.ndarray, output_elasticity: float = -0.006,
                                    okun_coefficient: float = 0.4, u_ref: float = 5.0) -> np.ndarray:
        """Alternative to ``rate_transmission`` (decision 0005, P-20): transmits
        the premium through a sovereign-risk pass-through elasticity from the
        literature instead of the VAR's own 1985-2026 lagged-10y coefficients,
        which were estimated on a sample with essentially no US sovereign-risk
        variation, a Fed-hike-shaped channel, not a sovereign-risk-shaped one
        (2026-09-17 steelman review of decision 0005).

        ``output_elasticity``: pp of annualized output growth per pp (100bp)
        of premium. Default -0.006 is Arellano, Bai & Bocola's (2017) central
        estimate: sovereign risk explained ~30% of Italy's 6.4pp 2012 output
        decline against a ~3.1pp average spread rise over 2011-2013 (their
        Table 7). Their own sensitivity range is 10%-50% attribution, call
        with output_elasticity in roughly [-0.002, -0.010] for a low/high
        case, not just the central one; never report only the point estimate.

        ``okun_coefficient``, ``u_ref``: the growth shock is converted to an
        unemployment response via a standard Okun's-law coefficient (~0.4),
        linearized around ``u_ref`` percent since the VAR carries log(u) and
        Okun's law is stated in levels. This piece is NOT sourced from the
        sovereign-risk papers, a standard macro rule of thumb standing in for
        a channel Bocola/Arellano-Bai-Bocola don't directly estimate,
        flagged as such rather than presented as equally well-grounded.
        ``pb`` and ``infl`` are left at zero, same as ``rate_transmission``."""
        d = np.asarray(delta_r10, float)
        g_shock = output_elasticity * d
        d_log_u = -okun_coefficient * g_shock / u_ref
        out = np.zeros((len(d), len(VARS)))
        out[:, self.index_of("g_nom")] = g_shock
        out[:, self.index_of("u")] = d_log_u
        return out

    def simulate(self, K: int, H: int, rng: np.random.Generator, init: Optional[np.ndarray] = None,
                 shocks: Optional[np.ndarray] = None, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """Unconditional (K, H, V) simulation, useful for validating the VAR
        alone (moments, mean reversion) before the doom loop is layered on."""
        p, V = self.lag, len(VARS)
        hist = np.zeros((K, p, V)) if init is None else np.broadcast_to(init, (K, p, V)).copy()
        shocks = self.draw_shocks(K, H, rng) if shocks is None else shocks
        out = np.empty((K, H, V))
        for t in range(H):
            y = self.step(hist, shocks[:, t, :], None if exog is None else exog[:, t, :])
            out[:, t, :] = y
            hist = np.concatenate([hist[:, 1:, :], y[:, None, :]], axis=1) if p > 1 else y[:, None, :]
        return out

    def last_history(self) -> np.ndarray:
        """(p, V) last observed states, the simulation's starting condition."""
        return self.sample.to_numpy()[-self.lag:, :]

    def index_of(self, var: str) -> int:
        return VARS.index(var)
