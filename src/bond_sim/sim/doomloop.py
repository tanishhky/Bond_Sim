"""The doom-loop state machine.

Two clocks, one grid:
* Quarterly macro block: the estimated levels VAR advances (r10, r3m, g_nom,
  pb, u) with correlated shocks. The fiscal premium implied by last quarter's
  debt/GDP is added on top of the base 10y to give the market 10y, and its
  change is transmitted to growth, the primary balance, and unemployment
  through the VAR's own estimated coefficients on the lagged 10y. Policies
  intervene on pb, g, the change in u, and the issuance yield.
* Monthly debt accounting: the legacy bond book's interest and principal
  schedules are exact (from ``BondBook.aggregates``); new issuance since as_of
  lives in a rolling pool with an average coupon that moves as maturing debt
  is refinanced at the current issuance yield (a bill/coupon blend). Deficit
  = interest - primary surplus; every dollar of deficit is new debt priced at
  that quarter's issuance yield.

The loop closes because debt/GDP enters the premium, the premium enters the
market yield, the yield enters both interest expense (arithmetic) and the
real economy (estimated), and both raise next quarter's debt/GDP.

Diagnostics use trailing-12-month interest, never a single month: coupon
payments cluster on 15th/month-end Feb-May-Aug-Nov cycles, so a single month
can be double the average without meaning anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import obs
from ..config import Config
from .macro import VARS, MacroVAR
from .policy import Policy, StatusQuo
from .premium import RiskPremiumModel

# Hard bounds on the quarterly macro state (levels). Wide enough to be inert in
# any sane run; they keep a bad configuration diagnosable instead of overflowing.
_BOUNDS = {"r10": (0.0, 25.0), "r3m": (0.0, 25.0), "g_nom": (-25.0, 25.0), "pb": (-25.0, 25.0), "u": (1.0, 30.0)}


@dataclass(frozen=True)
class InitialState:
    """Everything known at as_of. Units: $ millions and annualized percent."""
    debt_public_mm: float           # marketable debt stock (MSPD Table 1)
    gdp_saar_mm: float              # nominal GDP, SAAR, $mm
    r10_pct: float
    r3m_pct: float
    u_pct: float
    pb_pct_gdp: float               # trailing-4Q primary balance
    g_nom_pct: float                # trailing-4Q average nominal growth (annualized)
    legacy_interest_mm: np.ndarray  # (H,) monthly interest on the existing book after as_of
    legacy_principal_mm: np.ndarray # (H,) monthly maturing principal of the existing book
    avg_new_maturity_months: float = 72.0   # refinancing tenor: average remaining maturity of the stock (P-11)
    bill_share: float = 0.22        # share of the marketable stock in bills; issuance yield blend weight
    trailing_interest_mm: Optional[np.ndarray] = None   # (12,) realized interest before as_of, for the 12m ratios
    receipts_gdp_pct: float = 17.0  # federal receipts / GDP (estimated in setup.py; P-04 uses it)


@dataclass
class SimResult:
    dates: pd.DatetimeIndex
    paths: Dict[str, np.ndarray]            # name -> (K, H)
    policy: str
    premium_desc: str
    config_hash: str
    trigger_month: np.ndarray = field(default_factory=lambda: np.array([]))   # (K,) first trigger month or -1

    def quantiles(self, var: str, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        arr = self.paths[var]
        return pd.DataFrame({f"q{int(q*100):02d}": np.quantile(arr, q, axis=0) for q in qs}, index=self.dates)

    def trigger_probability(self) -> float:
        return float((self.trigger_month >= 0).mean())

    def time_to_trigger_years(self) -> pd.Series:
        m = self.trigger_month[self.trigger_month >= 0]
        return pd.Series(m / 12.0, name="years_to_trigger")

    def summary(self) -> pd.Series:
        end = {k: float(np.median(v[:, -1])) for k, v in self.paths.items()}
        return pd.Series({"policy": self.policy, "premium": self.premium_desc,
                          "P(trigger)": self.trigger_probability(),
                          "median_years_to_trigger": float(self.time_to_trigger_years().median()) if self.trigger_probability() else np.nan,
                          **{f"median_end_{k}": v for k, v in end.items()}})


class DoomLoopSimulator:
    def __init__(self, cfg: Config, var: MacroVAR, premium: RiskPremiumModel,
                 init: InitialState, policy: Optional[Policy] = None):
        self.cfg, self.var, self.premium, self.init = cfg, var, premium, init
        self.policy = policy or StatusQuo(cfg.policy)

    @staticmethod
    def _bounded(y: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        for v, (lo, hi) in _BOUNDS.items():
            y[:, idx[v]] = np.clip(y[:, idx[v]], lo, hi)
        return y

    # ── the run ─────────────────────────────────────────────────────────────
    def run(self, K: Optional[int] = None, seed: Optional[int] = None,
            shocks: Optional[np.ndarray] = None, start: Optional[pd.Timestamp] = None) -> SimResult:
        cfg, init = self.cfg, self.init
        K = K or cfg.sim.n_paths
        H = len(init.legacy_interest_mm)                      # months
        HQ = (H + 2) // 3                                     # quarters
        rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)
        shocks = self.var.draw_shocks(K, HQ, rng) if shocks is None else shocks
        assert shocks.shape == (K, HQ, len(VARS)), "shock array must match (K, quarters, VARS)"

        i = {v: self.var.index_of(v) for v in VARS}
        p = self.var.lag
        hist = np.broadcast_to(self.var.last_history(), (K, p, len(VARS))).copy()

        # State (K,)
        debt = np.full(K, init.debt_public_mm)
        gdp = np.full(K, init.gdp_saar_mm)
        r10_base = np.full(K, init.r10_pct)
        r3m = np.full(K, init.r3m_pct)
        u = np.full(K, init.u_pct)
        g = np.full(K, init.g_nom_pct)
        pb = np.full(K, init.pb_pct_gdp)
        pool = np.zeros(K)                                    # new-issuance stock since as_of
        pool_cpn = np.full(K, init.r10_pct)                   # its average coupon (pct)
        premium = np.zeros(K)
        prev_premium = np.zeros(K)
        r10 = r10_base.copy()
        issue_y = r10.copy()
        u_hist = [u.copy()] * 4
        seed12 = init.trailing_interest_mm if init.trailing_interest_mm is not None else np.full(12, init.legacy_interest_mm[:12].mean())
        ring = np.tile(np.asarray(seed12, float), (K, 1))
        tau = init.avg_new_maturity_months
        rshare = init.receipts_gdp_pct / 100.0
        wb = init.bill_share
        state: dict = {}

        out = {k: np.empty((K, H)) for k in ("debt_gdp", "r10", "r10_base", "r3m", "premium", "u", "g_nom", "pb",
                                             "interest_mm", "interest_12m_gdp", "interest_12m_receipts",
                                             "deficit_mm", "issuance_yield", "pool_coupon")}
        trigger = np.full(K, -1)

        for t in range(H):
            if t % 3 == 0:
                # ── quarterly macro block ───────────────────────────────
                tq = t // 3
                premium = np.minimum(self.premium.premium(np.minimum(100.0 * debt / gdp, 1000.0)), cfg.sim.max_rate_pct)
                d_prem = np.clip(premium - prev_premium, -5.0, 5.0)
                prev_premium = premium
                # premium change reaches g, pb, u through the estimated lagged-10y coefficients
                y = self.var.step(hist, shocks[:, tq, :], self.var.rate_transmission(d_prem))
                u_prev = u.copy()
                state.update(premium=premium, u=u, u_4q_ago=u_hist[0])
                y[:, i["g_nom"]] = self.policy.growth(tq, y[:, i["g_nom"]], state)
                y[:, i["pb"]] = self.policy.primary_balance(tq, y[:, i["pb"]], state)
                d_u = self.policy.unemployment_change(tq, y[:, i["u"]] - u_prev, state)
                y[:, i["u"]] = u_prev + d_u
                y = self._bounded(y, i)
                g, pb, u = y[:, i["g_nom"]], y[:, i["pb"]], y[:, i["u"]]
                r10_base, r3m = y[:, i["r10"]], y[:, i["r3m"]]
                r10 = np.clip(r10_base + premium, 0.0, cfg.sim.max_rate_pct)
                u_hist = u_hist[1:] + [u.copy()]
                hist = np.concatenate([hist[:, 1:, :], y[:, None, :]], axis=1) if p > 1 else y[:, None, :]
                # issuance yield: bills at the 3m rate, coupons at the market 10y (stock-mix weights)
                blend = wb * r3m + (1.0 - wb) * r10
                blend_base = wb * r3m + (1.0 - wb) * r10_base
                issue_y = np.clip(self.policy.issuance_yield(tq, blend, blend_base, state), 0.0, cfg.sim.max_rate_pct)

            # ── monthly debt accounting ────────────────────────────────
            gdp = gdp * np.maximum(1.0 + g / 100.0, 0.05) ** (1.0 / 12.0)
            pool_interest = pool * pool_cpn / 100.0 / 12.0
            interest = init.legacy_interest_mm[t] + pool_interest
            primary_surplus = pb / 100.0 * gdp / 12.0
            deficit = interest - primary_surplus                   # > 0 adds debt
            matured_pool = pool / tau
            refinanced = init.legacy_principal_mm[t] + matured_pool
            issued = refinanced + np.maximum(deficit, 0.0)
            new_pool = pool - matured_pool + issued
            blended = np.divide((pool - matured_pool) * pool_cpn + issued * issue_y, new_pool,
                                out=pool_cpn.copy(), where=new_pool > 0)
            pool_cpn = np.clip(blended, 0.0, cfg.sim.max_rate_pct)
            pool = new_pool
            debt = np.maximum(debt + deficit, 0.0)
            ring[:, t % 12] = interest
            int_12m = ring.sum(axis=1)
            debt_gdp = 100.0 * debt / gdp
            int_gdp = 100.0 * int_12m / gdp
            int_rcpt = int_gdp / rshare

            hit = (debt_gdp >= cfg.doomloop.loop_trigger_debt_gdp_pct) | (int_rcpt >= cfg.doomloop.loop_trigger_interest_rev_pct)
            trigger = np.where((trigger < 0) & hit, t, trigger)

            for k, v in (("debt_gdp", debt_gdp), ("r10", r10), ("r10_base", r10_base), ("r3m", r3m), ("premium", premium),
                         ("u", u), ("g_nom", g), ("pb", pb), ("interest_mm", interest), ("interest_12m_gdp", int_gdp),
                         ("interest_12m_receipts", int_rcpt), ("deficit_mm", deficit), ("issuance_yield", issue_y),
                         ("pool_coupon", pool_cpn)):
                out[k][:, t] = v

        dates = pd.date_range(start or pd.Timestamp.today().to_period("M").to_timestamp(), periods=H, freq="MS")
        res = SimResult(dates=dates, paths=out, policy=self.policy.name, premium_desc=self.premium.describe(),
                        config_hash=cfg.content_hash(), trigger_month=trigger)
        obs.event(channel="sim", kind="doomloop.run", policy=self.policy.name, K=K, H=H,
                  p_trigger=res.trigger_probability(), premium=self.premium.describe(),
                  median_end_debt_gdp=float(np.median(out["debt_gdp"][:, -1])))
        return res
