"""No-look-ahead invariance: the platform's signature test.

A point-in-time view stamped at as_of must not change when later vintages or
later observations are appended. Built on a synthetic VintageFrame so it runs
offline; the same assertions hold for live ALFRED frames.
"""
import numpy as np
import pandas as pd
import pytest

from bond_sim.data.pit import FAR_FUTURE, VintageFrame
from bond_sim.decorators import point_in_time


def _rows(series, date, values_with_windows):
    return [{"series_id": series, "date": date, "value": v, "realtime_start": s, "realtime_end": e}
            for v, s, e in values_with_windows]


def _frame(extended: bool) -> VintageFrame:
    rows = []
    # 2009Q3 GDP: initial 14301.5 on 2009-10-29, revised twice, then (if extended) again later
    rows += _rows("GDP", "2009-07-01", [(14301.5, "2009-10-29", "2009-11-23"),
                                        (14266.3, "2009-11-24", "2009-12-21"),
                                        (14242.1, "2009-12-22", "2010-07-29" if extended else FAR_FUTURE)])
    if extended:
        rows += _rows("GDP", "2009-07-01", [(14114.7, "2010-07-30", FAR_FUTURE)])
        rows += _rows("GDP", "2009-10-01", [(14453.8, "2010-01-29", FAR_FUTURE)])   # a later observation
    rows += _rows("UNRATE", "2009-09-01", [(9.8, "2009-10-02", FAR_FUTURE)])
    return VintageFrame.from_rows(pd.DataFrame(rows))


def test_as_of_view_is_invariant_to_future_vintages_and_observations():
    now, later = _frame(False), _frame(True)
    for as_of in ("2009-10-15", "2009-11-15", "2009-12-01", "2010-03-01"):
        a = now.as_of(as_of).sort_values(["series_id", "date"]).reset_index(drop=True)
        b = later.as_of(as_of).sort_values(["series_id", "date"]).reset_index(drop=True)
        # Before 2010-07-30 nothing that later.df adds was public, so the views must match exactly
        if pd.Timestamp(as_of) < pd.Timestamp("2010-01-29"):
            pd.testing.assert_frame_equal(a, b)
    # The 2009Q4 GDP observation exists only from 2010-01-29
    gdp = lambda d: later.as_of(d).query("series_id == 'GDP'")["date"].max()   # noqa: E731
    assert gdp("2010-01-28") == pd.Timestamp("2009-07-01")
    assert gdp("2010-01-29") == pd.Timestamp("2009-10-01")


def test_vintage_selection_matches_publication_history():
    vf = _frame(True)
    assert vf.as_of("2009-11-01").set_index("date").loc["2009-07-01", "value"] == 14301.5
    assert vf.as_of("2009-12-01").set_index("date").loc["2009-07-01", "value"] == 14266.3
    assert vf.as_of("2011-01-01").set_index("date").loc["2009-07-01", "value"] == 14114.7
    assert vf.initial_release().set_index("date").loc["2009-07-01", "value"] == 14301.5
    assert vf.latest().set_index("date").loc["2009-07-01", "value"] == 14114.7
    assert len(vf.as_of("2009-10-01")) == 0            # nothing public yet


def test_point_in_time_decorator_rejects_leaks():
    @point_in_time(date_col="date", release_col="realtime_start")
    def leaky(*, as_of):
        return pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
                             "realtime_start": pd.to_datetime(["2020-02-01", "2021-02-01"]), "value": [1.0, 2.0]})

    with pytest.raises(AssertionError):
        leaky(as_of="2020-06-01")

    @point_in_time(date_col="date", release_col="realtime_start")
    def clean(*, as_of):
        d = leaky.__wrapped__(as_of=as_of)
        return d[d["realtime_start"] <= pd.Timestamp(as_of)]

    assert len(clean(as_of="2020-06-01")) == 1
    with pytest.raises(ValueError):
        clean(as_of=None)
