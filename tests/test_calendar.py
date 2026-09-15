import numpy as np
import pandas as pd
import pytest

from bond_sim.calendar import MonthlyGrid


def test_grid_size_and_dates():
    g = MonthlyGrid.build("1980-01-15", "1980-12-31")
    assert g.M == 12
    assert g.start == pd.Timestamp("1980-01-01")
    assert list(g.dates[[0, -1]]) == [pd.Timestamp("1980-01-01"), pd.Timestamp("1980-12-01")]


def test_index_roundtrip_and_off_grid():
    g = MonthlyGrid.build("2000-01-01", "2001-12-01")
    idx = g.index_of(["2000-01-31", "2000-06-15", "2001-12-01", "1999-11-01", "2002-03-01"])
    np.testing.assert_array_equal(idx, [0, 5, 23, -2, 26])
    np.testing.assert_array_equal(g.contains(idx), [True, True, True, False, False])
    assert g.date_of(5) == pd.Timestamp("2000-06-01")


def test_history_plus_horizon_and_as_of_split():
    g = MonthlyGrid.from_history_plus_horizon("1980-01-01", "2026-09-15", 30)
    assert g.start == pd.Timestamp("1980-01-01")
    assert g.end == pd.Timestamp("2056-09-01")
    assert g.index_as_of("2026-09-15") == (2026 - 1980) * 12 + 8
    with pytest.raises(ValueError):
        g.index_as_of("2070-01-01")
