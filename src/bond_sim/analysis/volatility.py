"""Realized and (eventually) implied rates volatility.

Phase 1 of the volatility program (2026-09-21): realized volatility of daily
yield changes, computed point-in-time from data already in the pipeline (no
new data source). The question this exists to answer: does rate volatility
rise *before* the model's own fiscal-stress diagnostics move, i.e. is it a
leading indicator of entering the doom loop, or does it just move alongside
everything else. Answered empirically in ``scripts/vol_leadlag_analysis.py``,
not assumed here.

Phase 2 (scoped, not yet built): a market-implied volatility series (MOVE
index or a Treasury-options-implied proxy) alongside the realized series here
gives a rates variance risk premium, implied minus realized, the same P-vs-Q
construction VolEdge uses for equities (BKM risk-neutral moments vs GARCH
physical moments), applied to rates instead. Free data source not yet
verified; do not assume one exists until it has been checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import obs
from ..calendar import MonthlyGrid

TRADING_DAYS_PER_YEAR = 252


def realized_vol(daily_yield: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling realized volatility of daily yield changes, in annualized
    percentage points (i.e. vol of the yield level itself, not log-returns,
    the standard convention for rates since yields can sit near zero and log
    returns are undefined there).

    Point-in-time by construction: ``pandas.rolling`` with the default
    ``center=False`` uses only observations up to and including the current
    date, so ``realized_vol(s)[t]`` cannot change when later observations are
    appended, no as-of filtering is needed beyond what ``daily_yield`` itself
    already carries. ``window=20`` is roughly one trading month; the first
    ``window`` points are NaN (one lost to the initial ``diff()``, the rest
    to the rolling std's ``min_periods``), not zero, an early value must
    never be read as "no volatility"."""
    if window < 2:
        raise ValueError("window must be >= 2 to compute a standard deviation")
    d = daily_yield.sort_index().diff()
    vol = d.rolling(window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol.name = f"{daily_yield.name}_rvol{window}"
    return vol


def monthly_realized_vol(daily_yield: pd.Series, grid: MonthlyGrid, window: int = 20,
                         annualize: bool = True, how: str = "mean") -> pd.Series:
    """Daily realized vol resampled to the monthly grid (``how``: mean or
    eop), for feeding into ``CorrelationAnalyzer``'s monthly panel alongside
    the macro state. Resampling a lagging-window statistic to monthly does
    not reintroduce lookahead: each daily value already only used data up to
    that day, and a monthly mean/eop of already-point-in-time values is still
    point-in-time."""
    rv = realized_vol(daily_yield, window=window, annualize=annualize)
    agg = {"mean": rv.resample("MS").mean, "eop": lambda: rv.resample("MS").last()}[how]()
    out = agg.reindex(grid.dates)
    obs.event(channel="analysis", kind="realized_vol", series=str(daily_yield.name),
              window=window, n=int(out.notna().sum()))
    return out
