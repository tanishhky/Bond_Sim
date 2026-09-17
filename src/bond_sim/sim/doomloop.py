"""The doom-loop state machine.

Two clocks, one grid:
* Quarterly macro block (pluggable, ``macroblock.py``): the VAR or the
  latent-state chain produces (base 10y, 3m, growth, primary balance,
  unemployment, inflation). The fiscal premium implied by last quarter's
  debt/GDP is added on top of the base 10y to give the market 10y; the
  premium change is passed to the block, which transmits it to the real
  economy its own way (estimated rate coefficients for the VAR; financial-
  conditions shift and stress hazard for the chain).
* Monthly debt accounting: the legacy bond book's interest and principal
  schedules are exact; new issuance lives in a rolling pool refinanced at a
  bill/coupon blend of current yields; deficit = interest - primary surplus,
  financed at that blend.

No clipping of macro variables (decision: admissibility, math.md 7). The
premium is capped at the model ceiling as a documented boundary. Paths are
filtered afterwards by ``admissible.evaluate`` and every rejection is
reported. Two triggers are recorded per path: the identity-based one (pb*
above feasible pb for N periods, math.md 4) when a ``FeasiblePB`` is given,
and the legacy threshold one for comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import obs
from ..config import Config
from . import admissible
from .policy import Policy, StatusQuo
from .premium import RiskPremiumModel
from .sustainability import FeasiblePB, evaluate_paths


@dataclass(frozen=True)
class InitialState:
    """Everything known at as_of. Units: $ millions and annualized percent."""
    debt_public_mm: float
    gdp_saar_mm: float
    r10_pct: float
    r3m_pct: float
    u_pct: float
    pb_pct_gdp: float               # trailing-4Q primary balance
    g_nom_pct: float                # trailing-4Q average nominal growth (annualized)
    legacy_interest_mm: np.ndarray  # (H,) monthly interest on the existing book after as_of
    legacy_principal_mm: np.ndarray # (H,) monthly maturing principal of the existing book
    avg_new_maturity_months: float = 72.0
    bill_share: float = 0.22
    trailing_interest_mm: Optional[np.ndarray] = None
    receipts_gdp_pct: float = 17.0


@dataclass
class SimResult:
    dates: pd.DatetimeIndex
    paths: Dict[str, np.ndarray]            # name -> (K, H)
    policy: str
    premium_desc: str
    block_desc: str
    config_hash: str
    trigger_month: np.ndarray               # (K,) identity-based if feasible given, else threshold
    trigger_month_threshold: np.ndarray     # (K,) legacy threshold trigger, always computed
    admissible: np.ndarray                  # (K,) bool
    rejections: pd.Series                   # rule -> count
    sustainability: Optional[object] = None  # SustainabilityDiagnostics when available

    def _ok(self) -> np.ndarray:
        return self.admissible

    def quantiles(self, var: str, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        arr = self.paths[var][self._ok()]
        return pd.DataFrame({f"q{int(q*100):02d}": np.quantile(arr, q, axis=0) for q in qs}, index=self.dates)

    def trigger_probability(self, which: str = "identity") -> float:
        t = self.trigger_month if which == "identity" else self.trigger_month_threshold
        return float((t[self._ok()] >= 0).mean())

    def time_to_trigger_years(self, which: str = "identity") -> pd.Series:
        t = (self.trigger_month if which == "identity" else self.trigger_month_threshold)[self._ok()]
        return pd.Series(t[t >= 0] / 12.0, name="years_to_trigger")

    def summary(self) -> pd.Series:
        ok = self._ok()
        end = {k: (float(np.nanmedian(v[ok, -1])) if ok.any() and np.isfinite(v[ok, -1]).any() else np.nan)
               for k, v in self.paths.items()}
        p_id, p_th = self.trigger_probability("identity"), self.trigger_probability("threshold")
        # Rejection is a selection rule, so P(trigger) above is P(trigger | admissible). The
        # bracket below bounds the unconditional probability: rejected paths counted as
        # never triggered (lo) or all triggered (hi); the joint share says how many of the
        # rejected paths had in fact triggered before they were rejected (decision 0007).
        trig = self.trigger_month >= 0
        p_lo = float((trig & ok).mean())
        return pd.Series({"policy": self.policy, "block": self.block_desc, "premium": self.premium_desc,
                          "admissible_share": float(ok.mean()),
                          "P(trigger)": p_id, "P(trigger_threshold)": p_th,
                          "P(trigger_rejected)": float((trig & ~ok).mean()),
                          "P(trigger)_lo": p_lo, "P(trigger)_hi": p_lo + float((~ok).mean()),
                          "median_years_to_trigger": float(self.time_to_trigger_years().median()) if p_id else np.nan,
                          **{f"median_end_{k}": v for k, v in end.items()}})


class DoomLoopSimulator:
    def __init__(self, cfg: Config, block, premium: RiskPremiumModel, init: InitialState,
                 policy: Optional[Policy] = None, feasible: Optional[FeasiblePB] = None,
                 trigger_persistence_months: Optional[int] = None, feasible_which: Optional[str] = None,
                 require_r_gt_g: Optional[bool] = None):
        self.cfg, self.block, self.premium, self.init = cfg, block, premium, init
        self.policy = policy or StatusQuo(cfg.policy)
        dl = cfg.doomloop
        self.feasible = feasible
        self.n_persist = dl.trigger_persistence_months if trigger_persistence_months is None else trigger_persistence_months
        self.feasible_which = dl.feasible_benchmark if feasible_which is None else feasible_which
        self.require_r_gt_g = dl.trigger_require_r_gt_g if require_r_gt_g is None else require_r_gt_g

    def run(self, K: Optional[int] = None, seed: Optional[int] = None, draws=None,
            start: Optional[pd.Timestamp] = None) -> SimResult:
        cfg, init = self.cfg, self.init
        K = K or cfg.sim.n_paths
        H = len(init.legacy_interest_mm)
        HQ = (H + 2) // 3
        rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)
        draws = self.block.prepare(K, HQ, rng) if draws is None else draws
        self.block.reset(K, init.u_pct)

        debt = np.full(K, init.debt_public_mm)
        gdp = np.full(K, init.gdp_saar_mm)
        r10_base = np.full(K, init.r10_pct)
        r3m = np.full(K, init.r3m_pct)
        u = np.full(K, init.u_pct)
        g = np.full(K, init.g_nom_pct)
        pb = np.full(K, init.pb_pct_gdp)
        infl = np.full(K, np.nan)
        state = np.full(K, -1)
        pool, pool_cpn = np.zeros(K), np.full(K, init.r10_pct)
        premium, prev_premium = np.zeros(K), np.zeros(K)
        r10, issue_y = r10_base.copy(), r10_base.copy()
        u_hist = [u.copy()] * 4
        seed12 = init.trailing_interest_mm if init.trailing_interest_mm is not None else np.full(12, init.legacy_interest_mm[:12].mean())
        ring = np.tile(np.asarray(seed12, float), (K, 1))
        tau, rshare, wb = init.avg_new_maturity_months, init.receipts_gdp_pct / 100.0, init.bill_share
        ctx: dict = {}
        names = ("debt_gdp", "debt_mm", "gdp_mm", "r10", "r10_base", "r3m", "premium", "u", "g_nom", "pb", "infl", "state",
                 "interest_mm", "interest_12m_gdp", "interest_12m_receipts", "r_eff", "deficit_mm", "issuance_yield", "pool_coupon")
        out = {k: np.empty((K, H)) for k in names}
        trig_th = np.full(K, -1)

        for t in range(H):
            if t % 3 == 0:
                tq = t // 3
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    dg = np.where(gdp > 0, 100.0 * debt / gdp, np.inf)
                premium = np.minimum(self.premium.premium(np.minimum(dg, 1000.0)), cfg.sim.max_rate_pct)
                d_prem = premium - prev_premium
                prev_premium = premium
                ctx.update(premium=premium, u=u, u_4q_ago=u_hist[0])
                m = self.block.advance(tq, draws, d_prem, self.policy, ctx)
                r10_base, r3m, g, pb, u = m["r10_base"], m["r3m"], m["g_nom"], m["pb"], m["u"]
                infl = m.get("infl", infl)
                state = m.get("state", state)
                r10 = r10_base + premium
                u_hist = u_hist[1:] + [u.copy()]
                blend = wb * r3m + (1.0 - wb) * r10
                blend_base = wb * r3m + (1.0 - wb) * r10_base
                issue_y = self.policy.issuance_yield(tq, blend, blend_base, ctx)

            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                gdp = gdp * np.sign(1.0 + g / 100.0) * np.abs(1.0 + g / 100.0) ** (1.0 / 12.0)
                pool_interest = pool * pool_cpn / 100.0 / 12.0
                interest = init.legacy_interest_mm[t] + pool_interest
                primary_surplus = pb / 100.0 * gdp / 12.0
                deficit = interest - primary_surplus
                matured_pool = pool / tau
                refinanced = init.legacy_principal_mm[t] + matured_pool
                issued = refinanced + np.maximum(deficit, 0.0)
                new_pool = pool - matured_pool + issued
                pool_cpn = np.divide((pool - matured_pool) * pool_cpn + issued * issue_y, new_pool,
                                     out=pool_cpn.copy(), where=new_pool > 0)
                pool = new_pool
                debt = debt + deficit
                ring[:, t % 12] = interest
                int_12m = ring.sum(axis=1)
                debt_gdp = 100.0 * debt / gdp
                int_gdp = 100.0 * int_12m / gdp
                int_rcpt = int_gdp / rshare
                r_eff = 100.0 * int_12m / np.maximum(debt, 1e-9)
                hit = (debt_gdp >= cfg.doomloop.loop_trigger_debt_gdp_pct) | (int_rcpt >= cfg.doomloop.loop_trigger_interest_rev_pct)
            trig_th = np.where((trig_th < 0) & hit, t, trig_th)
            for k, v in (("debt_gdp", debt_gdp), ("debt_mm", debt), ("gdp_mm", gdp), ("r10", r10), ("r10_base", r10_base),
                         ("r3m", r3m), ("premium", premium), ("u", u), ("g_nom", g), ("pb", pb), ("infl", infl),
                         ("state", state), ("interest_mm", interest), ("interest_12m_gdp", int_gdp),
                         ("interest_12m_receipts", int_rcpt), ("r_eff", r_eff), ("deficit_mm", deficit),
                         ("issuance_yield", issue_y), ("pool_coupon", pool_cpn)):
                out[k][:, t] = v

        ok, rejections = admissible.evaluate({k: out[k] for k in ("u", "r10", "r3m", "g_nom", "infl", "gdp_mm", "debt_mm", "debt_gdp")})
        trig_id = trig_th.copy()
        diag = None
        if self.feasible is not None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                diag = evaluate_paths(out["debt_gdp"] / 100.0, out["r_eff"] / 100.0, out["g_nom"] / 100.0, out["pb"] / 100.0,
                                      self.feasible, self.n_persist, which=self.feasible_which, u=out["u"],
                                      require_r_gt_g=self.require_r_gt_g,
                                      growth_smoothing=cfg.doomloop.trigger_growth_smoothing_months)
            trig_id = diag.trigger_index
        dates = pd.date_range(start or pd.Timestamp.today().to_period("M").to_timestamp(), periods=H, freq="MS")
        res = SimResult(dates=dates, paths=out, policy=self.policy.name, premium_desc=self.premium.describe(),
                        block_desc=self.block.describe(), config_hash=cfg.content_hash(), trigger_month=trig_id,
                        trigger_month_threshold=trig_th, admissible=ok, rejections=rejections, sustainability=diag)
        obs.event(channel="sim", kind="doomloop.run", policy=self.policy.name, block=self.block.describe(), K=K, H=H,
                  admissible=float(ok.mean()), rejections=rejections.to_dict(),
                  p_trigger=res.trigger_probability(), p_trigger_threshold=res.trigger_probability("threshold"))
        return res
