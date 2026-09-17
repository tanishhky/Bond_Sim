"""Helpers that keep the phase notebooks readable: one call per step, every
step's logic living in the package where the tests can reach it.

A notebook cell reads like the sentence it explains:

    ctx  = nb.load_context(AS_OF)                  # data, grid, panels
    book = nb.build_book(ctx)                       # Phase 2
    bf, chain = nb.fit_factors_and_chain(ctx)       # Phase 3
    hist, feas = nb.history_sustainability(ctx, book)   # Phase 4
    runs = nb.run_scenarios(ctx, book, ...)         # Phases 5-6
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as cfgmod
from .analysis import BLOCKS, BlockFactors, build_block_panel
from .calendar import MonthlyGrid
from .data import SERIES, load_auctions, load_macro_panel, load_mspd_summary
from .data.fred import panel_wide
from .engine.book import BondBook
from .sim import (DoomLoopSimulator, FeasiblePB, LatentStateModel, MacroVAR, StateBlock, VARBlock,  # noqa: F401 (re-exported for notebooks)
                  build_quarterly_state, evaluate_history, policy_from_name, premium_from_config)
from .sim.setup import build_initial_state, realized_short_rate

warnings.filterwarnings("ignore", message=".*encountered in matmul")
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", message="Mean of empty slice")

SIM_SERIES = ["DGS10", "DGS3MO", "GDP", "MTSDS133FMS", "MTSR133FMS", "A091RC1Q027SBEA", "UNRATE"]


@dataclass
class Context:
    as_of: pd.Timestamp
    cfg: cfgmod.Config
    grid: MonthlyGrid
    w: pd.DataFrame          # monthly wide panel, history only
    book: BondBook           # the reconstructed marketable book (its interest feeds the primary balance)
    auctions: pd.DataFrame
    agg: pd.DataFrame        # book aggregates on the full grid ($mm)
    short: np.ndarray        # realized 3m rate path (decimal) for FRN accrual
    q: pd.DataFrame          # quarterly block panel with derived fiscal columns
    qs: pd.DataFrame         # quarterly VAR state frame

    @property
    def t0(self) -> int:
        return self.grid.index_as_of(self.as_of)


def all_series_ids(extra: Optional[List[str]] = None) -> List[str]:
    ids = {s for b in BLOCKS.values() for s, _ in b["series"] if s in SERIES}
    ids |= set(SIM_SERIES) | {"FYGFGDQ188S", "FYGFDPUN", "FDHBFIN", "FDHBFRBN", "GDPC1", "CPIAUCSL", "THREEFYTP10",
                              "PAYEMS", "ICSA", "JTSLDL"}
    ids |= set(cfgmod.load().labor.sectors)
    ids |= set(extra or [])
    return sorted(ids)


def load_context(as_of, config_path=None, extra_series: Optional[List[str]] = None) -> Context:
    as_of = pd.Timestamp(as_of)
    cfg = cfgmod.load(config_path)
    grid = MonthlyGrid.from_history_plus_horizon(cfg.data.history_start, as_of, cfg.calendar.horizon_years)
    w = panel_wide(load_macro_panel(all_series_ids(extra_series), as_of=as_of, grid=grid), grid)
    w = w.loc[w.index <= as_of]
    auctions = load_auctions(as_of=as_of)
    book = BondBook.from_auctions(auctions, grid, cfg.book)
    short = realized_short_rate(w, grid)
    agg = book.aggregates(short_rate=short).frame
    interest_hist = agg.loc[agg.index <= as_of, "interest"]
    return Context(as_of, cfg, grid, w, book, auctions, agg, short,
                   build_block_panel(w, interest_monthly=interest_hist),
                   build_quarterly_state(w, interest_monthly=interest_hist))


# ── Phase 2 ─────────────────────────────────────────────────────────────────

def build_book(ctx: Context):
    """(book, auctions, aggregates, short-rate path); built once in load_context."""
    return ctx.book, ctx.auctions, ctx.agg, ctx.short


def effective_rate(agg: pd.DataFrame, as_of) -> pd.Series:
    """Bottom-up effective interest rate: trailing-12m interest / stock 12m earlier (annual decimal)."""
    r = agg["interest"].rolling(12).sum() / agg["outstanding"].shift(12)
    return r.loc[r.index <= pd.Timestamp(as_of)].rename("r_eff")


# ── Phase 3 ─────────────────────────────────────────────────────────────────

def fit_factors_and_chain(ctx: Context, n_states: Optional[int] = None):
    st = ctx.cfg.states
    bf = BlockFactors(BLOCKS, start=st.start).fit(ctx.q)
    chain = LatentStateModel(n_states=n_states or st.n_states, candidates=st.candidates,
                             prior_strength=st.prior_strength, seed=st.seed, winsor_sd=st.winsor_sd,
                             min_state_occupancy=st.min_state_occupancy,
                             hazard_beta_override=st.hazard_beta).fit(bf.factors())
    return bf, chain


def fit_var(ctx: Context) -> MacroVAR:
    dl = ctx.cfg.doomloop
    var = MacroVAR(max_lag=dl.var_max_lag).fit(ctx.qs, start=dl.var_start)
    means = {k: v for k, v in (("pb", dl.pb_anchor_pct_gdp), ("r10", dl.r10_anchor_pct), ("u", dl.u_anchor_pct)) if v is not None}
    if means:
        var.set_long_run_means(means)
    return var


# ── Phase 4 ─────────────────────────────────────────────────────────────────

def history_frame(ctx: Context, agg: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Quarterly (d, r_eff, g_nom, pb, u) in the identity's units (ratios / decimals).
    ``d`` is marketable debt from the book over GDP so the ratio, the effective
    rate, and the primary balance share one debt concept."""
    agg = ctx.agg if agg is None else agg
    r = effective_rate(agg, ctx.as_of).resample("QS").mean()
    stock_q = agg.loc[agg.index <= ctx.as_of, "outstanding"].resample("QS").mean()      # $mm
    d = stock_q / (ctx.qs["gdp"] * 1e3)
    h = pd.DataFrame({"d": d, "r_eff": r, "g_nom": ctx.qs["g_nom"] / 100.0, "pb": ctx.qs["pb"] / 100.0, "u": ctx.qs["u"]})
    return h.replace([np.inf, -np.inf], np.nan).dropna()


def history_sustainability(ctx: Context, agg: Optional[pd.DataFrame] = None, n_quarters: int = 4,
                           which: Optional[str] = None, quantile: Optional[float] = None):
    dl = ctx.cfg.doomloop
    h = history_frame(ctx, agg)
    feas = FeasiblePB.from_history(h["pb"], h["d"], h["u"], quantile=dl.feasible_quantile if quantile is None else quantile)
    H = evaluate_history(h, feas, n_quarters=n_quarters, which=dl.feasible_benchmark if which is None else which,
                         require_r_gt_g=dl.trigger_require_r_gt_g,
                         growth_smoothing_quarters=max(dl.trigger_growth_smoothing_months // 3, 1))
    return H, feas, h


def realtime_history_sustainability(as_of_dates, config_path=None, n_quarters: int = 4,
                                    which: Optional[str] = None, quantile: Optional[float] = None):
    """Real-time counterpart of ``history_sustainability``: rebuilds the context
    as of each date (vintage macro panel, book through that date), so the
    trigger, the growth smoothing and the feasible envelope only see what was
    known then. Slow (one ``load_context`` per date; the FRED panel is disk
    cached, the book is rebuilt), so run it on quarter starts, not months.
    Returns (realtime_table, comparison_with_todays_vintage)."""
    from bond_sim.sim import evaluate_history_realtime, compare_realtime_final
    cfg = cfgmod.load(config_path)
    dl = cfg.doomloop
    rt = evaluate_history_realtime(
        lambda t: history_frame(load_context(t, config_path)), as_of_dates, n_quarters=n_quarters,
        which=dl.feasible_benchmark if which is None else which, require_r_gt_g=dl.trigger_require_r_gt_g,
        growth_smoothing_quarters=max(dl.trigger_growth_smoothing_months // 3, 1),
        quantile=dl.feasible_quantile if quantile is None else quantile)
    H_final, _, _ = history_sustainability(load_context(pd.Timestamp(max(as_of_dates)), config_path),
                                           n_quarters=n_quarters, which=which, quantile=quantile)
    return rt, compare_realtime_final(rt, H_final)


# ── Phases 5-7 ──────────────────────────────────────────────────────────────

def initial_state(ctx: Context, book: BondBook, auctions: pd.DataFrame):
    return build_initial_state(book, ctx.w, ctx.qs, load_mspd_summary(as_of=ctx.as_of), auctions, ctx.grid, ctx.as_of)


def make_blocks(ctx: Context, var: MacroVAR, bf: BlockFactors, chain: LatentStateModel, shocks=None) -> Dict[str, object]:
    return {"VAR": VARBlock(var), "states": StateBlock(bf, chain, shocks=shocks,
                                                       premium_to_financial=ctx.cfg.states.premium_to_financial)}


def run_scenarios(ctx: Context, init, feasible: FeasiblePB, block_factory, policies: List[str], K: int,
                  shocks_by_name: Optional[Dict[str, list]] = None, policy_start_quarter: int = 0,
                  premium=None, seed: Optional[int] = None) -> Dict[str, object]:
    """Run every (policy, shock) combination on identical draws. ``block_factory(shocks)``
    returns a fresh macro block; the draws are prepared once from the first block."""
    cfg = ctx.cfg
    premium = premium or premium_from_config(cfg.doomloop.risk_premium)
    HQ = (len(init.legacy_interest_mm) + 2) // 3
    rng = np.random.default_rng(cfg.sim.seed if seed is None else seed)
    draws = block_factory(None).prepare(K, HQ, rng)
    shocks_by_name = shocks_by_name or {"none": None}
    out = {}
    for sname, shocks in shocks_by_name.items():
        for pname in policies:
            sim = DoomLoopSimulator(cfg, block_factory(shocks), premium, init,
                                    policy_from_name(pname, cfg.policy, policy_start_quarter), feasible=feasible)
            out[(sname, pname)] = sim.run(K=K, draws=draws, start=ctx.as_of + pd.DateOffset(months=1))
    return out


def summary_table(runs: Dict[tuple, object]) -> pd.DataFrame:
    rows = []
    for (s, p), r in runs.items():
        row = r.summary()
        row["shock"], row["policy"] = s, p
        rows.append(row)
    cols = ["shock", "policy", "block", "admissible_share", "P(trigger)", "P(trigger_threshold)", "median_years_to_trigger",
            "median_end_debt_gdp", "median_end_r10", "median_end_u", "median_end_g_nom", "median_end_pb",
            "median_end_interest_12m_gdp", "median_end_r_eff"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols and c.startswith("median_end_infl")]]


def shape_stats(res, var: str = "debt_gdp") -> pd.Series:
    """Distribution shape of the terminal value on admissible paths: the
    non-normality the state model is supposed to deliver."""
    from scipy import stats
    x = res.paths[var][res.admissible, -1]
    x = x[np.isfinite(x)]
    return pd.Series({"median": np.median(x), "mean": x.mean(), "std": x.std(), "skew": stats.skew(x),
                      "excess_kurtosis": stats.kurtosis(x), "p05": np.quantile(x, .05), "p95": np.quantile(x, .95)})


def quarterly_changes(monthly: np.ndarray, x0: float) -> np.ndarray:
    """(K, H) monthly levels -> (K, HQ) end-of-quarter changes, first change vs x0."""
    eoq = monthly[:, 2::3]
    prev = np.concatenate([np.full((monthly.shape[0], 1), x0), eoq[:, :-1]], axis=1)
    return eoq - prev


def book_profile(book: BondBook, dates) -> pd.DataFrame:
    """Face-weighted average remaining maturity (months) and bill share of the
    stock outstanding at each date."""
    rows = []
    for d in dates:
        t = int(book.grid.index_of([d])[0])
        live = (book.issue_period <= t) & (book.maturity_period > t)
        w = book.face[live]
        if w.sum() == 0:
            continue
        rem = (book.maturity_period[live] - t).astype(float)
        rows.append({"date": pd.Timestamp(d), "wam_months": float((rem * w).sum() / w.sum()),
                     "bill_share": float(w[book.kind[live] == "bill"].sum() / w.sum()),
                     "outstanding_tn": float(w.sum() / 1e6)})
    return pd.DataFrame(rows).set_index("date")


def cached(name: str, fn, refresh: bool = False, root: str = "runs/notebook_cache"):
    """Pickle-memoize an expensive notebook step so a rerun of a later section
    does not redo the earlier ones. Delete the folder or pass refresh=True."""
    import pickle
    from pathlib import Path
    p = Path(root) / f"{name}.pkl"
    if p.exists() and not refresh:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    obj = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(obj, fh)
    return obj


# ── plotting ────────────────────────────────────────────────────────────────

def fan(ax, res, var: str, label: str, band: bool = True, color=None):
    qq = res.quantiles(var)
    (line,) = ax.plot(qq.index, qq["q50"], label=label, color=color)
    if band:
        ax.fill_between(qq.index, qq["q05"], qq["q95"], alpha=0.12, color=line.get_color())
    return ax
