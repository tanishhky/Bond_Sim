"""Treasury Fiscal Data client: the security-level ledger the whole book is built from.

Endpoints (verified live 2026-09-15 against the API metadata, base URL
``https://api.fiscaldata.treasury.gov/services/api/fiscal_service``):

    /v1/accounting/od/auctions_query      every auction since 1979-11 (11,114 rows,
                                          114 fields): issue/maturity dates, coupon
                                          (``int_rate``), bill discount rate, amounts
    /v1/debt/mspd/mspd_table_3_market     Monthly Statement of the Public Debt,
                                          marketable detail, 2001-01+: per-CUSIP
                                          outstanding (first tranche row carries the
                                          CUSIP total; reopening rows have null)
    /v2/accounting/od/interest_expense    monthly interest expense by security type, 2010-05+
    /v2/accounting/od/avg_interest_rates  monthly average rate by security class, 2001-01+
    /v2/accounting/od/debt_to_penny       daily debt held by public / intragov / total, 1993+

Point-in-time: each loader takes ``as_of`` and keeps only records that were
public then. Auctions are public on ``auction_date``; MSPD is published on the
4th business day after month end (``publication_lag_days`` models that); the
daily/monthly summaries use ``record_date`` plus a short lag. All amounts are
converted to float; the API's string ``"null"`` becomes NaN.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

from .. import obs
from ..decorators import disk_cache, point_in_time

BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
PAGE = 10000                                          # honored by the API (probed)

AUCTION_FIELDS = (
    "record_date,cusip,security_type,security_term,original_security_term,auction_date,"
    "issue_date,maturity_date,dated_date,int_rate,high_yield,high_discnt_rate,price_per100,"
    "total_accepted,soma_accepted,offering_amt,reopening,inflation_index_security,"
    "floating_rate,cash_management_bill_cmb,int_payment_frequency,spread"
)
MSPD_FIELDS = (
    "record_date,security_class1_desc,security_class2_desc,interest_rate_pct,yield_pct,"
    "issue_date,maturity_date,issued_amt,inflation_adj_amt,redeemed_amt,outstanding_amt,src_line_nbr"
)
MSPD_SUMMARY_FIELDS = "record_date,security_type_desc,security_class_desc,debt_held_public_mil_amt,intragov_hold_mil_amt,total_mil_amt"
# security_class2_desc holds a CUSIP on detail rows and a label ("Total Treasury Notes",
# "Total Unmatured Treasury Notes") on subtotal rows; only real CUSIPs count.
_CUSIP_RE = r"^[0-9A-Z]{9}$"


@disk_cache("fiscaldata")
def _fetch_all(endpoint: str, fields: Optional[str], filter_: Optional[str], sort: Optional[str]) -> pd.DataFrame:
    """Paginate an endpoint to exhaustion; return every row as strings."""
    rows, page = [], 1
    with requests.Session() as s:
        while True:
            params = {"page[size]": PAGE, "page[number]": page}
            if fields:
                params["fields"] = fields
            if filter_:
                params["filter"] = filter_
            if sort:
                params["sort"] = sort
            r = s.get(f"{BASE}{endpoint}", params=params, timeout=300)
            r.raise_for_status()
            payload = r.json()
            rows.extend(payload["data"])
            if page >= int(payload["meta"]["total-pages"]):
                break
            page += 1
    obs.event(channel="data", kind="fiscaldata.fetch", endpoint=endpoint, n=len(rows), pages=page)
    return pd.DataFrame(rows)


def _numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("null", np.nan), errors="coerce")
    return df


def _dates(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].replace("null", np.nan), errors="coerce")
    return df


def _yesno(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].eq("Yes")
    return df


class FiscalDataClient:
    """Typed loaders over the raw paginated fetch. Stateless; exists so the
    endpoint knowledge lives in one place."""

    def auctions(self) -> pd.DataFrame:
        df = _fetch_all("/v1/accounting/od/auctions_query", AUCTION_FIELDS, None, "issue_date")
        df = _dates(df, ["record_date", "auction_date", "issue_date", "maturity_date", "dated_date"])
        df = _numeric(df, ["int_rate", "high_yield", "high_discnt_rate", "price_per100",
                           "total_accepted", "soma_accepted", "offering_amt", "spread"])
        df = _yesno(df, ["reopening", "inflation_index_security", "floating_rate", "cash_management_bill_cmb"])
        return df

    def mspd_marketable(self) -> pd.DataFrame:
        df = _fetch_all("/v1/debt/mspd/mspd_table_3_market", MSPD_FIELDS, None, "record_date")
        df = _dates(df, ["record_date", "issue_date", "maturity_date"])
        df = _numeric(df, ["interest_rate_pct", "yield_pct", "issued_amt", "inflation_adj_amt",
                           "redeemed_amt", "outstanding_amt", "src_line_nbr"])
        return df.rename(columns={"security_class1_desc": "security_class", "security_class2_desc": "cusip"})

    def mspd_summary(self) -> pd.DataFrame:
        """MSPD Table 1: outstanding by security class per month ($ millions),
        the cleanest validation target for the reconstructed book."""
        df = _fetch_all("/v1/debt/mspd/mspd_table_1", MSPD_SUMMARY_FIELDS, None, "record_date")
        return _numeric(_dates(df, ["record_date"]), ["debt_held_public_mil_amt", "intragov_hold_mil_amt", "total_mil_amt"])

    def interest_expense(self) -> pd.DataFrame:
        df = _fetch_all("/v2/accounting/od/interest_expense",
                        "record_date,expense_catg_desc,expense_group_desc,expense_type_desc,month_expense_amt,fytd_expense_amt",
                        None, "record_date")
        return _numeric(_dates(df, ["record_date"]), ["month_expense_amt", "fytd_expense_amt"])

    def avg_interest_rates(self) -> pd.DataFrame:
        df = _fetch_all("/v2/accounting/od/avg_interest_rates",
                        "record_date,security_type_desc,security_desc,avg_interest_rate_amt", None, "record_date")
        return _numeric(_dates(df, ["record_date"]), ["avg_interest_rate_amt"])

    def debt_to_penny(self) -> pd.DataFrame:
        df = _fetch_all("/v2/accounting/od/debt_to_penny",
                        "record_date,debt_held_public_amt,intragov_hold_amt,tot_pub_debt_out_amt", None, "record_date")
        return _numeric(_dates(df, ["record_date"]), ["debt_held_public_amt", "intragov_hold_amt", "tot_pub_debt_out_amt"])


# ── point-in-time loaders ───────────────────────────────────────────────────

@point_in_time(date_col="auction_date", release_col="auction_date")
def load_auctions(*, as_of, client: Optional[FiscalDataClient] = None) -> pd.DataFrame:
    """Auction ledger known on ``as_of``: a security is known once auctioned."""
    df = (client or FiscalDataClient()).auctions()
    return df[df["auction_date"] <= pd.Timestamp(as_of)].reset_index(drop=True)


@point_in_time(date_col="record_date", release_col="published")
def load_mspd_marketable(*, as_of, client: Optional[FiscalDataClient] = None,
                         publication_lag_days: int = 6) -> pd.DataFrame:
    """MSPD marketable detail known on ``as_of``. Each month-end statement is
    published on the 4th business day of the next month; ``published`` models
    that with a calendar-day lag. Adds ``cusip_outstanding``: the CUSIP total
    (the first tranche row carries it, reopening rows are null)."""
    df = (client or FiscalDataClient()).mspd_marketable()
    df["published"] = df["record_date"] + pd.Timedelta(days=publication_lag_days)
    df = df[df["published"] <= pd.Timestamp(as_of)].copy()
    tot = df.groupby(["record_date", "cusip"])["outstanding_amt"].transform("max")
    df["cusip_outstanding"] = tot
    return df.reset_index(drop=True)


def _simple_pit(fn_name: str):
    @point_in_time(date_col="record_date", release_col="published")
    def loader(*, as_of, client: Optional[FiscalDataClient] = None, publication_lag_days: int = 6) -> pd.DataFrame:
        df = getattr(client or FiscalDataClient(), fn_name)()
        df["published"] = df["record_date"] + pd.Timedelta(days=publication_lag_days)
        return df[df["published"] <= pd.Timestamp(as_of)].reset_index(drop=True)
    loader.__name__ = f"load_{fn_name}"
    return loader


load_interest_expense = _simple_pit("interest_expense")
load_avg_interest_rates = _simple_pit("avg_interest_rates")
load_debt_to_penny = _simple_pit("debt_to_penny")
load_mspd_summary = _simple_pit("mspd_summary")


def mspd_class_totals(mspd: pd.DataFrame) -> pd.DataFrame:
    """Month x security class outstanding ($ millions) from MSPD detail rows,
    one number per CUSIP per month (subtotal rows and reopening rows dropped),
    plus the published Total Marketable line for a top-level cross-check.
    Should agree with ``mspd_summary`` to rounding; it is the per-CUSIP path."""
    det = mspd[mspd["cusip"].astype(str).str.match(_CUSIP_RE)]
    per_cusip = det.drop_duplicates(["record_date", "cusip"])[["record_date", "security_class", "cusip_outstanding"]]
    by_class = per_cusip.pivot_table(index="record_date", columns="security_class",
                                     values="cusip_outstanding", aggfunc="sum")
    total = (mspd[mspd["security_class"] == "Total Marketable"]
             .set_index("record_date")["outstanding_amt"].rename("Total Marketable (published)"))
    return by_class.join(total, how="left")
