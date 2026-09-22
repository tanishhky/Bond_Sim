"""Realized and (eventually) implied rates volatility.

Phase 1 of the volatility program (2026-09-21): realized volatility of daily
yield changes, computed point-in-time from data already in the pipeline (no
new data source). The question this exists to answer: does rate volatility
rise *before* the model's own fiscal-stress diagnostics move, i.e. is it a
leading indicator of entering the doom loop, or does it just move alongside
everything else. Answered empirically in ``scripts/vol_leadlag_analysis.py``,
not assumed here.

Phase 2 (2026-09-22): a market-implied volatility series alongside the
realized series here gives a rates variance risk premium, implied minus
realized, the same P-vs-Q construction VolEdge uses for equities (BKM
risk-neutral moments vs GARCH physical moments), applied to rates instead.

Checked live against FRED's API on 2026-09-21: MOVE itself is not on FRED,
and the only rates-specific implied-vol series FRED ever carried, VXTYN
(CBOE's TYVIX, 10-Year Treasury Note Volatility), was discontinued
2020-05-15. There is no free, current, point-in-time-vintage-compatible
implied-vol series for Treasuries as of this writing. VXTYN's own vintage
metadata in ALFRED does not reflect genuine historical publication timing
either (the whole discontinued series carries one recent realtime_start),
so any VXTYN-based analysis here uses final-vintage values and is an
explicitly bounded, backward-looking event study (2003-2020), never fed
into the live as-of simulation pipeline. A current MOVE series would need
a source outside FRED (Bloomberg, WRDS/ICE) to extend this past 2020.
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


def variance_risk_premium(implied: pd.Series, realized: pd.Series) -> pd.Series:
    """Implied minus realized vol, on whatever common index the two series
    share (inner join). Positive means the market is pricing more rate
    volatility than has actually been realized, the same sign convention as
    the equity VRP: a persistently positive premium is compensation for
    crash/tail risk, not free edge (VolEdge's own finding for equities,
    tested here, not assumed, for rates)."""
    idx = implied.index.intersection(realized.index)
    if len(idx) == 0:
        raise ValueError("implied and realized series share no common dates")
    vrp = (implied.loc[idx] - realized.loc[idx]).rename("vrp")
    return vrp
