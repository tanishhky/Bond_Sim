"""The shared calendar axis every array in the engine is indexed against.

The toy notebook used ``periods = arange(M)`` with semiannual steps. The real
engine uses a monthly grid (decision 0001): it is the finest frequency at which
the Treasury Monthly Statement of the Public Debt, the Monthly Treasury
Statement, and BLS employment all report natively, and quarterly GDP can be
interpolated onto it without inventing intra-month structure.

A ``MonthlyGrid`` maps calendar dates to integer period indices and back. Every
bond, every macro series, and every simulated path lives on the same axis, so
column sums across bonds and matmuls across scenarios line up by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np
import pandas as pd

DateLike = Union[str, pd.Timestamp, np.datetime64]


@dataclass(frozen=True)
class MonthlyGrid:
    """Inclusive month grid from ``start`` to ``end`` (both snapped to month start).

    Attributes
    ----------
    start, end : first and last month on the grid (month-start timestamps)
    M          : number of periods (columns) on the grid
    """
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def build(cls, start: DateLike, end: DateLike) -> "MonthlyGrid":
        """Construct from any date-like pair; snaps both ends to month start."""
        s = pd.Timestamp(start).to_period("M").to_timestamp()
        e = pd.Timestamp(end).to_period("M").to_timestamp()
        if e < s:
            raise ValueError(f"grid end {e.date()} precedes start {s.date()}")
        return cls(start=s, end=e)

    @classmethod
    def from_history_plus_horizon(cls, history_start: DateLike, as_of: DateLike,
                                  horizon_years: int) -> "MonthlyGrid":
        """History from ``history_start`` through ``as_of``, then ``horizon_years``
        of forward months for simulation. The as_of month is the last observed
        column; everything after it is simulated."""
        a = pd.Timestamp(as_of).to_period("M").to_timestamp()
        return cls.build(history_start, a + pd.DateOffset(years=horizon_years))

    # ── size and axis ───────────────────────────────────────────────────────
    @property
    def M(self) -> int:
        return (self.end.year - self.start.year) * 12 + (self.end.month - self.start.month) + 1

    @property
    def periods(self) -> np.ndarray:
        """``arange(M)``: the integer axis every (N, M) / (K, M) array shares."""
        return np.arange(self.M)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Month-start timestamp for every period, length M."""
        return pd.date_range(self.start, periods=self.M, freq="MS")

    # ── date <-> index (vectorized, no loops) ──────────────────────────────
    def index_of(self, dates: Iterable[DateLike]) -> np.ndarray:
        """Integer period index for each date. Dates before ``start`` map to
        negative indices and dates after ``end`` map past M-1; callers decide
        whether to clip, mask, or raise, because a bond issued before the grid
        is a legitimate (still-outstanding) row, not an error."""
        d = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        return ((d.year - self.start.year) * 12 + (d.month - self.start.month)).to_numpy()

    def date_of(self, idx: Union[int, np.ndarray]) -> Union[pd.Timestamp, pd.DatetimeIndex]:
        """Month-start timestamp for an integer index (or array of them)."""
        if np.isscalar(idx):
            return self.start + pd.DateOffset(months=int(idx))
        return pd.DatetimeIndex([self.start + pd.DateOffset(months=int(i)) for i in np.asarray(idx)])

    def contains(self, idx: np.ndarray) -> np.ndarray:
        """Boolean mask: which indices fall on the grid."""
        idx = np.asarray(idx)
        return (idx >= 0) & (idx < self.M)

    def index_as_of(self, as_of: DateLike) -> int:
        """Column index of the as_of month; the split between observed and simulated."""
        i = int(self.index_of([as_of])[0])
        if not (0 <= i < self.M):
            raise ValueError(f"as_of {pd.Timestamp(as_of).date()} is off the grid "
                             f"[{self.start.date()}, {self.end.date()}]")
        return i

    def __repr__(self) -> str:
        return f"MonthlyGrid({self.start.date()} .. {self.end.date()}, M={self.M})"
