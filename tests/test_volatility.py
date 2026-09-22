"""Realized volatility: point-in-time invariance and basic sanity, on synthetic
daily yield paths so it runs offline."""
import numpy as np
import pandas as pd
import pytest

from bond_sim.analysis.volatility import realized_vol, monthly_realized_vol
from bond_sim.calendar import MonthlyGrid


def _daily_series(n=300, seed=0, vol_bump_at=None, bump_factor=5.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    steps = rng.normal(0, 0.02, n)   # ~2bp/day baseline
    if vol_bump_at is not None:
        steps[vol_bump_at:vol_bump_at + 20] *= bump_factor
    return pd.Series(4.0 + np.cumsum(steps), index=idx, name="DGS10")


def test_realized_vol_first_window_points_are_nan():
    s = _daily_series()
    rv = realized_vol(s, window=20)
    assert rv.iloc[:20].isna().all()
    assert rv.iloc[20:].notna().all()


def test_realized_vol_is_not_annualized_when_asked_not_to_be():
    s = _daily_series()
    raw = realized_vol(s, window=20, annualize=False)
    ann = realized_vol(s, window=20, annualize=True)
    ratio = (ann / raw).dropna()
    assert np.allclose(ratio, np.sqrt(252), atol=1e-6)


def test_realized_vol_no_lookahead_appending_future_data_does_not_change_the_past():
    full = _daily_series(n=300)
    early = full.iloc[:150]
    rv_early = realized_vol(early, window=20)
    rv_full = realized_vol(full, window=20)
    pd.testing.assert_series_equal(rv_early, rv_full.loc[rv_early.index])


def test_realized_vol_detects_a_real_vol_spike():
    s = _daily_series(n=300, vol_bump_at=150, bump_factor=8.0)
    rv = realized_vol(s, window=20)
    calm = rv.iloc[100:140].mean()
    stressed = rv.iloc[150:170].mean()
    assert stressed > 3 * calm


def test_monthly_realized_vol_aligns_to_grid_month_starts():
    s = _daily_series(n=300)
    grid = MonthlyGrid.from_history_plus_horizon("2020-01-01", "2020-08-01", horizon_years=0)
    m = monthly_realized_vol(s, grid, window=20)
    assert list(m.index) == list(grid.dates)
    assert m.index.freqstr == "MS" or (m.index[1] - m.index[0]).days in (28, 29, 30, 31)
