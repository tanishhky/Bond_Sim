"""VintageFrame: the data structure that makes point-in-time queries trivial.

ALFRED returns every observation once per real-time window: the value that was
public from ``realtime_start`` through ``realtime_end``. A ``VintageFrame`` is
that long table for one or more series. Asking "what did the world know on
date X" is then a single boolean mask, and no downstream code ever touches a
revised number by accident.

Columns (long format, one row per series x observation x vintage window):
    series_id       str
    date            observation date (period start)
    value           float
    realtime_start  first day this value was public
    realtime_end    last day this value was current (9999-12-31 for the latest)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

# ALFRED marks the current vintage with realtime_end = 9999-12-31, which is
# outside pandas' nanosecond Timestamp range (max 2262-04-11). Every such
# sentinel is remapped to this in-bounds one on ingest; comparisons are unaffected.
FAR_FUTURE = pd.Timestamp("2200-01-01")
_ALFRED_OPEN_END = "9999-12-31"
REQUIRED = ("series_id", "date", "value", "realtime_start", "realtime_end")


@dataclass(frozen=True)
class VintageFrame:
    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED if c not in self.df.columns]
        if missing:
            raise ValueError(f"VintageFrame missing columns {missing}")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def from_rows(cls, rows: pd.DataFrame) -> "VintageFrame":
        """Normalize dtypes (dates -> Timestamp, value -> float) and sort."""
        df = rows.copy()
        df["realtime_end"] = df["realtime_end"].replace(_ALFRED_OPEN_END, FAR_FUTURE.strftime("%Y-%m-%d"))
        for c in ("date", "realtime_start", "realtime_end"):
            df[c] = pd.to_datetime(df[c])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values(["series_id", "date", "realtime_start"])
        return cls(df.reset_index(drop=True))

    @classmethod
    def concat(cls, frames: Iterable["VintageFrame"]) -> "VintageFrame":
        return cls(pd.concat([f.df for f in frames], ignore_index=True))

    # ── point-in-time views ────────────────────────────────────────────────
    def as_of(self, as_of) -> pd.DataFrame:
        """Rows current on ``as_of``: the vintage whose real-time window contains
        it. Observations first published after ``as_of`` have no such window and
        drop out, so the future is excluded by construction, not by a second
        filter. Returns [series_id, date, value, realtime_start]."""
        cut = pd.Timestamp(as_of)
        m = (self.df["realtime_start"] <= cut) & (self.df["realtime_end"] >= cut)
        out = self.df.loc[m, ["series_id", "date", "value", "realtime_start"]]
        # One row per (series, date) is guaranteed by ALFRED's non-overlapping
        # windows; the assert makes a provider bug loud instead of silent.
        dup = out.duplicated(["series_id", "date"]).sum()
        assert dup == 0, f"{dup} overlapping vintage windows at {cut.date()}"
        return out.reset_index(drop=True)

    def initial_release(self) -> pd.DataFrame:
        """First-published value of every observation (RateWalk's output_type=4 view)."""
        return (self.df.sort_values("realtime_start")
                .drop_duplicates(["series_id", "date"], keep="first")
                .sort_values(["series_id", "date"]).reset_index(drop=True))

    def latest(self) -> pd.DataFrame:
        """Fully revised series as of today (what plain FRED shows)."""
        return self.df[self.df["realtime_end"] == FAR_FUTURE].reset_index(drop=True)

    def revisions(self, series_id: str, date) -> pd.DataFrame:
        """Full revision history of one observation, oldest vintage first."""
        d = pd.Timestamp(date)
        m = (self.df["series_id"] == series_id) & (self.df["date"] == d)
        return self.df.loc[m].sort_values("realtime_start").reset_index(drop=True)

    # ── convenience ─────────────────────────────────────────────────────────
    def series_ids(self) -> list:
        return sorted(self.df["series_id"].unique().tolist())

    def wide(self, as_of, series_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Date x series matrix as known on ``as_of``."""
        v = self.as_of(as_of)
        if series_ids is not None:
            v = v[v["series_id"].isin(list(series_ids))]
        return v.pivot(index="date", columns="series_id", values="value").sort_index()

    def __len__(self) -> int:
        return len(self.df)
