"""Assemble the simulator's starting point from the data layer.

Everything in ``InitialState`` is read off real data as of the as_of date:
the legacy book's forward interest and principal schedules, the marketable
debt stock (MSPD Table 1), nominal GDP, the latest rates and unemployment,
the trailing-four-quarter primary balance and growth, the outstanding
stock's average remaining maturity (the refinancing tenor, P-11) and its
bill share (the issuance-yield blend weight), and receipts/GDP.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .. import obs
from ..calendar import MonthlyGrid
from ..engine.book import BondBook
from .doomloop import InitialState


def realized_short_rate(panel_wide: pd.DataFrame, grid: MonthlyGrid, r3m_col: str = "DGS3MO") -> np.ndarray:
    """(M,) annual decimal 3m rate on the grid: history where observed, held at
    the last observed value afterwards (legacy FRN accrual only; simulated
    issuance uses the simulated path)."""
    s = panel_wide[r3m_col].reindex(grid.dates).ffill().bfill()
    return (s / 100.0).to_numpy()


def stock_maturity_and_bill_share(book: BondBook, t0: int) -> tuple:
    """Face-weighted average remaining maturity (months) and bill share of the
    tranches outstanding at grid period t0."""
    live = (book.issue_period <= t0) & (book.maturity_period > t0)
    w = book.face[live]
    rem = (book.maturity_period[live] - t0).astype(float)
    tau = float((rem * w).sum() / w.sum())
    bill = float(w[book.kind[live] == "bill"].sum() / w.sum())
    return tau, bill


def build_initial_state(book: BondBook, panel_wide: pd.DataFrame, q_state: pd.DataFrame,
                        mspd_summary: pd.DataFrame, auctions: pd.DataFrame, grid: MonthlyGrid, as_of,
                        avg_new_maturity_months: Optional[float] = None) -> InitialState:
    t0 = grid.index_as_of(as_of)
    short = realized_short_rate(panel_wide, grid)
    agg = book.aggregates(short_rate=short, by_kind=False).frame
    legacy_interest = agg["interest"].to_numpy()[t0 + 1:]
    legacy_principal = agg["principal"].to_numpy()[t0 + 1:]
    trailing = agg["interest"].to_numpy()[max(t0 - 11, 0): t0 + 1]

    m = mspd_summary[(mspd_summary["security_type_desc"] == "Marketable") & (mspd_summary["record_date"] <= pd.Timestamp(as_of))]
    latest = m[m["record_date"] == m["record_date"].max()]
    debt_mm = float(latest["total_mil_amt"].sum())

    last = lambda col: float(panel_wide[col].dropna().iloc[-1])   # noqa: E731
    qs = q_state.replace([np.inf, -np.inf], np.nan).dropna(subset=["pb", "g_nom"])
    pb0 = float(qs["pb"].iloc[-1])                      # already a trailing-4Q ratio
    g0 = float(qs["g_nom"].iloc[-4:].mean())            # trailing-4Q average growth
    gdp_mm = last("GDP") * 1e3
    receipts_12m = float(panel_wide["MTSR133FMS"].dropna().iloc[-12:].sum()) if "MTSR133FMS" in panel_wide else np.nan
    rshare = 100.0 * receipts_12m / gdp_mm if np.isfinite(receipts_12m) else 17.0
    tau_stock, bill_share = stock_maturity_and_bill_share(book, t0)
    tau = avg_new_maturity_months or tau_stock

    init = InitialState(debt_public_mm=debt_mm, gdp_saar_mm=gdp_mm,
                        r10_pct=last("DGS10"), r3m_pct=last("DGS3MO"), u_pct=last("UNRATE"),
                        pb_pct_gdp=pb0, g_nom_pct=g0,
                        legacy_interest_mm=legacy_interest, legacy_principal_mm=legacy_principal,
                        avg_new_maturity_months=tau, bill_share=bill_share,
                        trailing_interest_mm=trailing, receipts_gdp_pct=rshare)
    obs.event(channel="sim", kind="initial_state", as_of=str(pd.Timestamp(as_of).date()),
              debt_mm=debt_mm, gdp_saar_mm=gdp_mm, debt_gdp_pct=round(100 * debt_mm / gdp_mm, 1),
              r10=init.r10_pct, r3m=init.r3m_pct, u=init.u_pct, pb=round(pb0, 3), g_nom=round(g0, 3),
              stock_avg_remaining_maturity_months=round(tau_stock, 1), bill_share=round(bill_share, 3),
              horizon_months=len(legacy_interest), legacy_interest_next12_mm=float(legacy_interest[:12].sum()),
              trailing_interest_12m_mm=float(trailing.sum()), receipts_gdp_pct=round(rshare, 2))
    return init
