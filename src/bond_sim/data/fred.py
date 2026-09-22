"""FRED / ALFRED client with full vintage history and a verified series registry.

Every series id below was resolved live against ``fred/series`` on 2026-09-15
(frequency, units, and date range recorded from the API response, not from
memory). ``FredClient.fetch_vintages`` pulls ``output_type=1`` ("Observations
by Real-Time Period": one row per observation per vintage window, i.e. the
complete revision history).

Two facts about ALFRED that shape this module (decision 0002):

1. A request may span at most 2000 vintage dates. Daily series get a vintage
   date every trading day (DGS10: 5108 since 2005-06-28), so the real-time
   axis is chunked by vintage-date windows and the chunks are coalesced.
2. Vintages start when ALFRED began tracking a series (GDP 1991-12-04, DGS10
   2005-06-28). Observations older than that carry the first vintage's date
   as ``realtime_start`` and would vanish from any earlier ``as_of`` view. For
   those rows only, ``realtime_start`` is backfilled to ``date + pub_lag_days``
   (a conservative typical publication lag per series) and the row is flagged
   ``backfilled=True`` so the paper can state exactly which observations are
   true vintages and which are first-vintage values with an assumed lag.

Point-in-time contract: ``load_macro_panel(as_of=...)`` returns a monthly panel
built only from vintages current on ``as_of``; observations first published
later do not exist in the result.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

from .. import obs
from ..calendar import MonthlyGrid
from ..decorators import disk_cache, point_in_time
from .pit import FAR_FUTURE, VintageFrame

_BASE = "https://api.stlouisfed.org/fred/series/observations"
_VINTAGES = "https://api.stlouisfed.org/fred/series/vintagedates"
_LIMIT = 100000                      # documented maximum rows per request
_MAX_VINTAGES = 1800                 # documented cap is 2000 per request; keep headroom


@dataclass(frozen=True)
class SeriesSpec:
    id: str
    title: str
    freq: str            # D | W | M | Q | A (native, from fred/series)
    units: str
    group: str           # fiscal | rates | macro | labor | holders | prices
    pub_lag_days: int    # conservative typical lag from period end to publication (pre-vintage backfill only)
    agg: str = "avg"     # higher-frequency -> monthly: avg | sum | eop
    fill: str = "ffill"  # lower-frequency -> monthly: ffill | interp


# Verified 2026-09-15 against api.stlouisfed.org/fred/series (title/freq/units/range).
# pub_lag_days are deliberately conservative (later than typical) so a backfilled
# realtime_start can only understate, never overstate, what was known.
SERIES: Dict[str, SeriesSpec] = {s.id: s for s in [
    # ── fiscal stock and flow ───────────────────────────────────────────────
    SeriesSpec("GFDEBTN", "Federal Debt: Total Public Debt", "Q", "Mil. of $", "fiscal", 10),
    SeriesSpec("FYGFDPUN", "Federal Debt Held by the Public", "Q", "Mil. of $", "fiscal", 10),
    SeriesSpec("GFDEGDQ188S", "Total Public Debt as Percent of GDP", "Q", "% of GDP", "fiscal", 35),
    SeriesSpec("FYGFGDQ188S", "Debt Held by the Public as Percent of GDP", "Q", "% of GDP", "fiscal", 35),
    SeriesSpec("MTSDS133FMS", "Federal Surplus or Deficit [-] (monthly, MTS)", "M", "Mil. of $", "fiscal", 14, agg="sum"),
    SeriesSpec("MTSR133FMS", "Total Federal Receipts (monthly, MTS)", "M", "Mil. of $", "fiscal", 14, agg="sum"),
    SeriesSpec("MTSO133FMS", "Total Federal Outlays (monthly, MTS)", "M", "Mil. of $", "fiscal", 14, agg="sum"),
    SeriesSpec("A091RC1Q027SBEA", "Federal current expenditures: Interest payments (SAAR)", "Q", "Bil. of $", "fiscal", 35),
    SeriesSpec("FYOIGDA188S", "Federal Outlays: Interest as Percent of GDP", "A", "% of GDP", "fiscal", 120),
    SeriesSpec("FGRECPT", "Federal Government Current Receipts (SAAR)", "Q", "Bil. of $", "fiscal", 35),
    SeriesSpec("FGEXPND", "Federal Government Current Expenditures (SAAR)", "Q", "Bil. of $", "fiscal", 35),
    # ── output and prices ───────────────────────────────────────────────────
    SeriesSpec("GDP", "Gross Domestic Product (nominal, SAAR)", "Q", "Bil. of $", "macro", 35),
    SeriesSpec("GDPC1", "Real Gross Domestic Product (SAAR)", "Q", "Bil. of Chn. 2017 $", "macro", 35),
    SeriesSpec("CPIAUCSL", "CPI-U All Items (SA)", "M", "Index 1982-1984=100", "prices", 20),
    # ── rates and term structure (never revised; published next business day) ──
    SeriesSpec("DFF", "Federal Funds Effective Rate (daily)", "D", "%", "rates", 2),
    SeriesSpec("FEDFUNDS", "Federal Funds Effective Rate (monthly)", "M", "%", "rates", 3),
    SeriesSpec("DGS3MO", "3-Month Treasury Constant Maturity", "D", "%", "rates", 2),
    SeriesSpec("DGS2", "2-Year Treasury Constant Maturity", "D", "%", "rates", 2),
    SeriesSpec("DGS10", "10-Year Treasury Constant Maturity", "D", "%", "rates", 2),
    SeriesSpec("DGS30", "30-Year Treasury Constant Maturity", "D", "%", "rates", 2),
    SeriesSpec("THREEFYTP10", "ACM 10-Year Term Premium", "D", "%", "rates", 7),
    SeriesSpec("MORTGAGE30US", "30-Year Fixed Mortgage Rate", "W", "%", "rates", 2),
    # ── who holds the debt ──────────────────────────────────────────────────
    SeriesSpec("FDHBFIN", "Federal Debt Held by Foreign and International Investors", "Q", "Bil. of $", "holders", 80),
    SeriesSpec("FDHBFRBN", "Federal Debt Held by Federal Reserve Banks", "Q", "Bil. of $", "holders", 80),
    # ── labor: aggregate and rate-sensitive sectors (BLS: ~1st Friday; JOLTS ~5-6 weeks) ──
    SeriesSpec("UNRATE", "Unemployment Rate", "M", "%", "labor", 10),
    SeriesSpec("PAYEMS", "All Employees, Total Nonfarm", "M", "Thous.", "labor", 10),
    SeriesSpec("ICSA", "Initial Claims (weekly)", "W", "Number", "labor", 6, agg="avg"),
    SeriesSpec("JTSLDL", "JOLTS Layoffs and Discharges: Total Nonfarm", "M", "Thous.", "labor", 45),
    SeriesSpec("JTSJOL", "JOLTS Job Openings: Total Nonfarm", "M", "Thous.", "labor", 45),
    SeriesSpec("USCONS", "All Employees, Construction", "M", "Thous.", "labor", 10),
    SeriesSpec("CES5553000001", "All Employees, Real Estate and Rental and Leasing", "M", "Thous.", "labor", 10),
    SeriesSpec("CES3133600101", "All Employees, Motor Vehicles and Parts", "M", "Thous.", "labor", 10),
    SeriesSpec("CES5552200001", "All Employees, Credit Intermediation", "M", "Thous.", "labor", 10),
    SeriesSpec("USFIRE", "All Employees, Financial Activities", "M", "Thous.", "labor", 10),
    SeriesSpec("CES9091000001", "All Employees, Federal Government", "M", "Thous.", "labor", 10),
    SeriesSpec("USPBS", "All Employees, Professional and Business Services", "M", "Thous.", "labor", 10),
    SeriesSpec("USTRADE", "All Employees, Retail Trade", "M", "Thous.", "labor", 10),
    SeriesSpec("MANEMP", "All Employees, Manufacturing", "M", "Thous.", "labor", 10),
    SeriesSpec("USGOVT", "All Employees, Government", "M", "Thous.", "labor", 10),
    # ── behavioral blocks for the latent-state model (verified 2026-09-15) ──
    # consumer (BEA personal income release ~30 days; G.19 credit ~40 days; UMich final ~0-3 days)
    SeriesSpec("PCE", "Personal Consumption Expenditures (SAAR)", "M", "Bil. of $", "consumer", 32),
    SeriesSpec("DSPIC96", "Real Disposable Personal Income (SAAR)", "M", "Bil. of Chn. 2017 $", "consumer", 32),
    SeriesSpec("PSAVERT", "Personal Saving Rate", "M", "%", "consumer", 32),
    SeriesSpec("TOTALSL", "Total Consumer Credit Owned and Securitized", "M", "Mil. of $", "consumer", 40),
    SeriesSpec("REVOLSL", "Revolving Consumer Credit Owned and Securitized", "M", "Mil. of $", "consumer", 40),
    SeriesSpec("UMCSENT", "University of Michigan Consumer Sentiment (NSA)", "M", "Index 1966:Q1=100", "consumer", 5),
    # corporate / activity (BEA advance ~30 days; profits ~60; G.17 ~15; H.8 ~10; Z.1 ~75)
    SeriesSpec("PNFI", "Private Nonresidential Fixed Investment (SAAR)", "Q", "Bil. of $", "corporate", 35),
    SeriesSpec("CP", "Corporate Profits After Tax (SAAR)", "Q", "Bil. of $", "corporate", 65),
    SeriesSpec("BUSLOANS", "Commercial and Industrial Loans, All Commercial Banks", "M", "Bil. of $", "corporate", 12),
    SeriesSpec("INDPRO", "Industrial Production: Total Index", "M", "Index 2017=100", "corporate", 18),
    SeriesSpec("TCU", "Capacity Utilization: Total Index", "M", "%", "corporate", 18),
    SeriesSpec("NCBDBIQ027S", "Nonfinancial Corporate Business: Debt Securities, Liability (NSA)", "Q", "Mil. of $", "corporate", 80),
    # financial conditions (daily/weekly market data; SLOOS quarterly ~30 days)
    SeriesSpec("NFCI", "Chicago Fed National Financial Conditions Index", "W", "Index", "financial", 6),
    SeriesSpec("BAA10Y", "Moody's Baa Corporate Yield Relative to 10-Year Treasury", "D", "%", "financial", 2),
    SeriesSpec("VIXCLS", "CBOE Volatility Index: VIX", "D", "Index", "financial", 2),
    SeriesSpec("T10Y2Y", "10-Year Minus 2-Year Treasury Constant Maturity", "D", "%", "financial", 2),
    SeriesSpec("DRTSCILM", "Net Pct of Banks Tightening Standards for C&I Loans to Large Firms (SLOOS)", "Q", "%", "financial", 35),
    # prices (BLS CPI ~12-15 days; BEA PCE price ~30 days; Cleveland Fed median CPI same day as CPI)
    SeriesSpec("CPILFESL", "CPI-U: All Items Less Food and Energy (SA)", "M", "Index 1982-1984=100", "prices", 20),
    SeriesSpec("PCEPI", "PCE Chain-type Price Index (SA)", "M", "Index 2017=100", "prices", 32),
    SeriesSpec("MEDCPIM158SFRBCLE", "Median CPI (annualized pct change)", "M", "% Chg. at Annual Rate", "prices", 20),
    # holders
    SeriesSpec("FDHBPIN", "Federal Debt Held by Private Investors", "Q", "Bil. of $", "holders", 80),
]}


def _load_dotenv() -> None:
    """Load KEY=VALUE from the repo-root .env into os.environ (setdefault only)."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _api_key(env: str = "FRED_API_KEY") -> str:
    _load_dotenv()
    key = os.getenv(env)
    if not key:
        raise RuntimeError(f"{env} not set (put it in .env at the repo root)")
    return key


def _get(session: requests.Session, url: str, params: dict, what: str) -> dict:
    r = session.get(url, params=params, timeout=120)
    if not r.ok:
        # Never propagate the URL (it carries the key); surface FRED's own message.
        raise RuntimeError(f"FRED {r.status_code} on {what}: {r.text[:300]}")
    return r.json()


def _vintage_dates(session: requests.Session, series_id: str, api_key: str) -> List[str]:
    j = _get(session, _VINTAGES, {"series_id": series_id, "api_key": api_key, "file_type": "json", "limit": 10000}, series_id)
    return list(j.get("vintage_dates", []))


def _windows(vintage_dates: List[str]) -> List[tuple]:
    """Split the real-time axis into windows of at most _MAX_VINTAGES vintage dates."""
    if len(vintage_dates) <= _MAX_VINTAGES:
        return [("1776-07-04", "9999-12-31")]
    edges = vintage_dates[::_MAX_VINTAGES]
    out = []
    for i, start in enumerate(edges):
        if i + 1 < len(edges):
            end = (pd.Timestamp(edges[i + 1]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            end = "9999-12-31"
        out.append(("1776-07-04" if i == 0 else start, end))
    return out


def _coalesce(df: pd.DataFrame) -> pd.DataFrame:
    """Merge adjacent windows that carry the same value for the same observation
    (an artifact of chunking, and of ALFRED registering a vintage date for a
    daily series even when nothing about an old observation changed)."""
    df = df.sort_values(["date", "realtime_start"]).reset_index(drop=True)
    prev_end = df.groupby("date")["realtime_end"].shift()
    prev_val = df.groupby("date")["value"].shift()
    new_run = (df["value"] != prev_val) | (df["realtime_start"] != prev_end + pd.Timedelta(days=1))
    run = new_run.cumsum()
    g = df.groupby(run)
    out = g.agg(date=("date", "first"), value=("value", "first"),
                realtime_start=("realtime_start", "min"), realtime_end=("realtime_end", "max"))
    return out.reset_index(drop=True)


@disk_cache("fred_vintages")
def _fetch_vintage_rows(series_id: str, observation_start: str, api_key: str) -> pd.DataFrame:
    """Raw ALFRED pull (chunked, coalesced, pre-vintage backfilled). Cached as the
    full history so one download serves any as_of query."""
    spec = SERIES[series_id]
    with requests.Session() as s:
        vdates = _vintage_dates(s, series_id, api_key)
        chunks = []
        for rt_start, rt_end in _windows(vdates):
            offset = 0
            while True:
                j = _get(s, _BASE, {
                    "series_id": series_id, "api_key": api_key, "file_type": "json",
                    "output_type": 1, "realtime_start": rt_start, "realtime_end": rt_end,
                    "observation_start": observation_start, "limit": _LIMIT, "offset": offset,
                }, series_id)
                batch = j.get("observations", [])
                chunks.extend(batch)
                offset += len(batch)
                if len(batch) < _LIMIT or offset >= int(j.get("count", 0)):
                    break
    if not chunks:
        raise RuntimeError(f"ALFRED returned no observations for {series_id}")
    df = pd.DataFrame(chunks)[["date", "value", "realtime_start", "realtime_end"]]
    df["realtime_end"] = df["realtime_end"].replace("9999-12-31", FAR_FUTURE.strftime("%Y-%m-%d"))
    for c in ("date", "realtime_start", "realtime_end"):
        df[c] = pd.to_datetime(df[c])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = _coalesce(df)

    # Pre-vintage backfill: rows whose realtime_start is the very first vintage
    # date but whose observation predates it by more than the publication lag
    # were not "first published" on that date; ALFRED simply started tracking then.
    first_vintage = pd.Timestamp(vdates[0]) if vdates else df["realtime_start"].min()
    lag = pd.Timedelta(days=spec.pub_lag_days)
    back = (df["realtime_start"] == first_vintage) & (df["date"] + lag < first_vintage)
    df.loc[back, "realtime_start"] = df.loc[back, "date"] + lag
    df["backfilled"] = back
    df.insert(0, "series_id", series_id)
    obs.event(channel="data", kind="fred.vintages.raw", series=series_id, n_rows=len(df),
              n_vintage_dates=len(vdates), first_vintage=str(first_vintage.date()),
              n_backfilled=int(back.sum()), n_windows=len(_windows(vdates)))
    return df.reset_index(drop=True)


class FredClient:
    """Thin, cached ALFRED client. One instance per process is plenty."""

    def __init__(self, api_key: Optional[str] = None, key_env: str = "FRED_API_KEY"):
        self._key = api_key or _api_key(key_env)

    def fetch_vintages(self, series_id: str, observation_start: str = "1900-01-01") -> VintageFrame:
        if series_id not in SERIES:
            raise KeyError(f"{series_id} is not in the verified registry; add it to SERIES after checking fred/series")
        with obs.timed(channel="data", kind="fred.vintages", series=series_id):
            raw = _fetch_vintage_rows(series_id, observation_start, api_key=self._key)
        vf = VintageFrame.from_rows(raw)
        obs.event(channel="data", kind="fred.vintages.done", series=series_id,
                  n_rows=len(vf), n_obs=int(vf.df["date"].nunique()))
        return vf

    def fetch_many(self, series_ids: Iterable[str], observation_start: str = "1900-01-01") -> VintageFrame:
        return VintageFrame.concat(self.fetch_vintages(s, observation_start) for s in series_ids)

    def vintage_coverage(self, series_ids: Iterable[str]) -> pd.DataFrame:
        """Data-dictionary table: first true vintage per series and how many
        observations are pre-vintage backfills (for the paper's data section)."""
        rows = []
        for sid in series_ids:
            vf = self.fetch_vintages(sid)
            d = vf.df
            true_v = d.loc[~d["backfilled"], "realtime_start"].min() if "backfilled" in d else d["realtime_start"].min()
            rows.append({"series_id": sid, "title": SERIES[sid].title, "freq": SERIES[sid].freq,
                         "first_obs": d["date"].min().date(), "first_true_vintage": true_v.date(),
                         "n_obs": int(d["date"].nunique()), "n_vintage_rows": len(d),
                         "n_backfilled_obs": int(d.loc[d.get("backfilled", False), "date"].nunique()) if "backfilled" in d else 0,
                         "pub_lag_days_assumed": SERIES[sid].pub_lag_days})
        return pd.DataFrame(rows).set_index("series_id")


# ── monthly panel on the grid ───────────────────────────────────────────────

def _to_monthly(series: pd.Series, spec: SeriesSpec) -> pd.Series:
    """Bring one as_of series to month-start frequency using the registry's
    rule: average/sum/end-of-period for higher frequencies; forward-fill or
    interpolate for lower ones. Index in, index out: month-start timestamps."""
    s = series.dropna().sort_index()
    if spec.freq in ("D", "W"):
        how = {"avg": "mean", "sum": "sum", "eop": "last"}[spec.agg]
        return getattr(s.resample("MS"), how)()
    if spec.freq == "M":
        return s.resample("MS").first()
    m = s.resample("MS").first()
    return m.interpolate("linear") if spec.fill == "interp" else m.ffill()


@point_in_time(date_col="date", release_col="realtime_start")
def load_macro_panel(series_ids: Iterable[str], *, as_of, grid: MonthlyGrid,
                     client: Optional[FredClient] = None, observation_start: str = "1900-01-01") -> pd.DataFrame:
    """Monthly panel (long format [date, series_id, value, realtime_start]) on the
    grid, built from vintages current on ``as_of``. Values after ``as_of`` cannot
    appear: they had no vintage yet. The decorator re-checks that invariant."""
    client = client or FredClient()
    vf = client.fetch_many(series_ids, observation_start)
    known = vf.as_of(as_of)
    out = []
    for sid, g in known.groupby("series_id"):
        spec = SERIES[sid]
        s = g.set_index("date")["value"]
        m = _to_monthly(s, spec)
        m = m[(m.index >= grid.start) & (m.index <= pd.Timestamp(as_of))]
        rel = g.set_index("date")["realtime_start"].resample("MS").max().reindex(m.index).ffill()
        out.append(pd.DataFrame({"date": m.index, "series_id": sid, "value": m.values,
                                 "realtime_start": rel.values}))
    panel = pd.concat(out, ignore_index=True)
    obs.event(channel="data", kind="macro_panel", as_of=str(pd.Timestamp(as_of).date()),
              n_series=panel["series_id"].nunique(), n_rows=len(panel))
    return panel


def panel_wide(panel: pd.DataFrame, grid: MonthlyGrid) -> pd.DataFrame:
    """Date x series matrix aligned to the full grid (NaN where not yet observed
    or beyond as_of). Use for correlation/regime analysis."""
    w = panel.pivot(index="date", columns="series_id", values="value")
    return w.reindex(grid.dates)


def load_daily_series(series_ids: Iterable[str], *, as_of, client: Optional[FredClient] = None,
                      observation_start: str = "1900-01-01") -> Dict[str, pd.Series]:
    """Raw (unaggregated) as-of daily series, keyed by series id. Unlike
    ``load_macro_panel`` this skips ``_to_monthly``: realized volatility needs
    the actual daily observations, a monthly average of levels throws away
    the within-month variation that IS the quantity being measured. Same
    point-in-time contract as ``load_macro_panel``: nothing published after
    ``as_of`` appears."""
    client = client or FredClient()
    vf = client.fetch_many(series_ids, observation_start)
    known = vf.as_of(as_of)
    out = {}
    for sid, g in known.groupby("series_id"):
        s = g.set_index("date")["value"].sort_index()
        out[sid] = s[s.index <= pd.Timestamp(as_of)]
    obs.event(channel="data", kind="daily_series", as_of=str(pd.Timestamp(as_of).date()),
              n_series=len(out), n_rows=sum(len(s) for s in out.values()))
    return out
