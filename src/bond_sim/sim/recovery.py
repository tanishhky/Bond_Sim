"""How long does the suffering last: recovery half-lives.

Two sources, compared side by side in the paper:
* Historical episodes: for each recession, the months from the unemployment
  peak until half of the peak-to-pre-recession gap is closed.
* Simulated paths: the same statistic on every path of every policy run, so
  "how long" is a distribution, not a point.

Episode definitions are US recessions with a clear unemployment peak; the
cross-country comparator (Greece 2010s) is a documented extension since it
needs a non-FRED source. TODO(Tanishk) P-10: which episodes count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

EPISODES = {                       # name: (pre-recession trough window start, peak search window end)
    "volcker_1981": ("1979-01-01", "1986-12-01"),
    "early_1990s": ("1989-01-01", "1995-12-01"),
    "dotcom_2001": ("2000-01-01", "2006-12-01"),
    "gfc_2008": ("2007-01-01", "2016-12-01"),
    "covid_2020": ("2019-06-01", "2023-12-01"),
}


@dataclass(frozen=True)
class Episode:
    name: str
    trough_date: pd.Timestamp
    trough_u: float
    peak_date: pd.Timestamp
    peak_u: float
    half_life_months: Optional[int]     # None if the gap never half-closed in the window


class RecoveryModel:
    def __init__(self, episodes: Optional[Dict[str, tuple]] = None):
        self.episodes = episodes or EPISODES

    @staticmethod
    def half_life_from_path(u: np.ndarray, peak_idx: int, trough_u: float) -> Optional[int]:
        """Months after the peak until u falls below trough + 0.5 x (peak - trough)."""
        target = trough_u + 0.5 * (u[peak_idx] - trough_u)
        after = u[peak_idx:]
        below = np.nonzero(after <= target)[0]
        return int(below[0]) if len(below) else None

    def historical(self, unrate: pd.Series) -> pd.DataFrame:
        rows = []
        for name, (start, end) in self.episodes.items():
            w = unrate[(unrate.index >= start) & (unrate.index <= end)].dropna()
            if len(w) < 24:
                continue
            peak_date = w.idxmax()
            pre = w[w.index < peak_date]
            trough_date = pre.idxmin()
            hl = self.half_life_from_path(w.to_numpy(), int(w.index.get_loc(peak_date)), float(pre.min()))
            rows.append(Episode(name, trough_date, float(pre.min()), peak_date, float(w.max()), hl))
        return pd.DataFrame([e.__dict__ for e in rows]).set_index("name")

    def simulated(self, u_paths: np.ndarray, u0: float) -> pd.Series:
        """Half-life per path: peak of u over the path, trough = starting level."""
        K = u_paths.shape[0]
        out = np.full(K, np.nan)
        peaks = np.argmax(u_paths, axis=1)
        for k in range(K):
            if u_paths[k, peaks[k]] - u0 < 0.5:      # no meaningful downturn on this path
                continue
            hl = self.half_life_from_path(u_paths[k], int(peaks[k]), u0)
            out[k] = np.nan if hl is None else hl
        return pd.Series(out, name="half_life_months")
