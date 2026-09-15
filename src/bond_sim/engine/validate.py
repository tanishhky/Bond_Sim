"""Reconcile the reconstructed book against what the Treasury publishes.

Three independent checks, each a monthly series of (book, published, ratio):

1. Outstanding by class vs MSPD Table 1 (2001-01+). Tests the issue/maturity
   mapping and the face convention. Expected gaps: buybacks (book keeps
   bought-back tranches to maturity), TIPS inflation accrual (book carries
   nominal face), and any auction rows dropped for missing fields.
2. Interest vs Fiscal Data "Interest Expense on the Public Debt Outstanding"
   (2010-05+), accrual basis. The book's coupons are cash flows, so the
   comparison is on 12-month rolling sums where accrual and cash converge.
3. Marketable total vs Debt to the Penny (daily, 1993+): debt held by the
   public includes non-marketables, so the ratio is expected below 1 and
   stable; a trend break flags a book problem.

Nothing here adjusts the book. It measures, and the paper reports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import obs
from .book import BondBook

CLASS_MAP: Dict[str, str] = {           # book kind -> MSPD Table 1 security_class_desc
    "bill": "Bills", "note": "Notes", "bond": "Bonds",
    "tips": "Treasury Inflation-Protected Securities", "frn": "Floating Rate Notes",
}
EXPENSE_MAP: Dict[str, str] = {         # substring of expense_type_desc -> book kind
    "Bills": "bill", "Notes": "note", "Bonds": "bond", "Inflation": "tips", "Floating": "frn",
}


@dataclass
class ValidationReport:
    outstanding: pd.DataFrame          # date x [book_<class>, mspd_<class>, ratio_<class>, ..., total]
    interest: pd.DataFrame             # date x [book_12m, published_12m, ratio]
    penny: pd.DataFrame                # date x [book_marketable, penny_public, ratio]

    def headline(self) -> pd.Series:
        """Last-5-year mean ratios: the numbers that go in the paper's data appendix."""
        last = self.outstanding.index.max() - pd.DateOffset(years=5)
        o = self.outstanding[self.outstanding.index >= last]
        cols = {c: float(o[c].mean()) for c in o.columns if c.startswith("ratio_")}
        i = self.interest[self.interest.index >= last]
        cols["ratio_interest_12m"] = float(i["ratio"].mean()) if len(i) else np.nan
        return pd.Series(cols, name="mean_ratio_last_5y")


class BookValidator:
    def __init__(self, book: BondBook):
        if book.grid is None:
            raise ValueError("validation needs a book built on a MonthlyGrid")
        self.book = book
        self.grid = book.grid

    def outstanding_vs_mspd(self, mspd_summary: pd.DataFrame) -> pd.DataFrame:
        agg = self.book.aggregates(by_kind=True).frame
        m = mspd_summary[mspd_summary["security_type_desc"].eq("Marketable")].copy()
        m["date"] = m["record_date"].dt.to_period("M").dt.to_timestamp()      # month-end statement -> its month
        pub = m.pivot_table(index="date", columns="security_class_desc", values="total_mil_amt", aggfunc="sum")
        out = pd.DataFrame(index=pub.index)
        for kind, cls in CLASS_MAP.items():
            if cls not in pub.columns:
                continue
            # MSPD is end-of-month; the book's column t is start-of-month state, so compare with t+1's start
            book_eom = agg[f"outstanding_{kind}"].shift(-1).reindex(pub.index)
            out[f"book_{kind}"] = book_eom
            out[f"mspd_{kind}"] = pub[cls]
            out[f"ratio_{kind}"] = book_eom / pub[cls]
        out["book_total"] = out[[c for c in out.columns if c.startswith("book_")]].sum(axis=1)
        out["mspd_total"] = pub[[c for c in CLASS_MAP.values() if c in pub.columns]].sum(axis=1)
        out["ratio_total"] = out["book_total"] / out["mspd_total"]
        obs.event(channel="engine", kind="validate.outstanding", months=len(out),
                  ratio_total_last=float(out["ratio_total"].dropna().iloc[-1]) if out["ratio_total"].notna().any() else None)
        return out

    def interest_vs_expense(self, interest_expense: pd.DataFrame, short_rate: Optional[np.ndarray] = None) -> pd.DataFrame:
        agg = self.book.aggregates(short_rate=short_rate, by_kind=False).frame
        ie = interest_expense[interest_expense["expense_catg_desc"].str.contains("PUBLIC ISSUES", case=False, na=False)].copy()
        kind = pd.Series(pd.NA, index=ie.index, dtype="object")
        for sub, k in EXPENSE_MAP.items():
            kind = kind.mask(ie["expense_type_desc"].str.contains(sub, case=False, na=False), k)
        ie["kind"] = kind
        ie = ie.dropna(subset=["kind"])
        ie["date"] = ie["record_date"].dt.to_period("M").dt.to_timestamp()
        pub = ie.groupby("date")["month_expense_amt"].sum() / 1e6                  # $ -> $mm
        book = agg["interest"].reindex(pub.index)
        out = pd.DataFrame({"book_month": book, "published_month": pub})
        out["book_12m"] = out["book_month"].rolling(12).sum()
        out["published_12m"] = out["published_month"].rolling(12).sum()
        out["ratio"] = out["book_12m"] / out["published_12m"]
        return out

    def marketable_vs_penny(self, debt_to_penny: pd.DataFrame) -> pd.DataFrame:
        agg = self.book.aggregates(by_kind=False).frame
        p = debt_to_penny.copy()
        p["date"] = p["record_date"].dt.to_period("M").dt.to_timestamp()
        eom = p.sort_values("record_date").groupby("date")["debt_held_public_amt"].last() / 1e6
        book = agg["outstanding"].shift(-1).reindex(eom.index)
        return pd.DataFrame({"book_marketable": book, "penny_public": eom, "ratio": book / eom})

    def run(self, mspd_summary: pd.DataFrame, interest_expense: pd.DataFrame,
            debt_to_penny: pd.DataFrame, short_rate: Optional[np.ndarray] = None) -> ValidationReport:
        rep = ValidationReport(self.outstanding_vs_mspd(mspd_summary),
                               self.interest_vs_expense(interest_expense, short_rate),
                               self.marketable_vs_penny(debt_to_penny))
        obs.event(channel="engine", kind="validate.headline", **{k: round(v, 4) for k, v in rep.headline().items()})
        return rep
