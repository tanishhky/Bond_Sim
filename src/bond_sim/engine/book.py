"""BondBook: every Treasury tranche as a row, every month as a column.

This is Problems 1-3 of the notebook made real. The math is identical:

    Active[i,t]      = (t >  issue[i]) & (t <= maturity[i])
    Outstanding[i,t] = (t >= issue[i]) & (t <  maturity[i])          x face[i]
    CouponFlow[i,t]  = Active[i,t] & coupon_month[i,t]               x face[i] x coupon[i] / (12 / coupon_every)
    Principal[i,t]   = (t == maturity[i])                            x face[i]

Two representations of the same object:

* ``dense_*`` methods materialize the (N, M) boolean/float matrices exactly as
  the README specifies. They are the reference and the tool for per-bond
  attribution. At N ~ 11k tranches and M ~ 920 months a float64 matrix is
  ~80 MB, so they are opt-in.
* ``aggregates()`` never builds (N, M). Each tranche contributes to a known,
  short list of columns (its outstanding window, its coupon months, its
  maturity month), so the column sums are ``np.bincount`` over ~300k
  (row, column) pairs instead of a 10M-cell reduction. ``tests/test_book.py``
  asserts both paths agree to machine precision on the real book.

Instrument handling (documented simplifications, all flagged in the paper):
* Bills and cash-management bills are discount instruments: no coupon; the
  interest (face x discount rate x days/360, the Treasury's bank-discount
  convention) is booked in the maturity month alongside principal.
* Notes and bonds pay ``coupon_every`` (6) months, anchored on the maturity
  month, so the last coupon lands with principal. Odd first coupons and
  accrued interest paid at reopenings are ignored (they net to ~0 across the
  book).
* TIPS are carried at nominal face with their fixed coupon; the inflation
  accrual on principal is NOT modeled here (MSPD's ``inflation_adj_amt`` gives
  its size for the validation report). Documented gap, decision 0004.
* FRNs accrue ``(3m bill rate + spread) / 12`` on face each month; the rate
  path is supplied by the caller (realized history or a simulated path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .. import obs
from ..calendar import MonthlyGrid
from ..config import BookConfig

KINDS = ("bill", "note", "bond", "tips", "frn")


@dataclass(frozen=True)
class Aggregates:
    """Column sums on the grid, $ millions. Index = period 0..M-1."""
    frame: pd.DataFrame

    @property
    def outstanding(self) -> pd.Series:
        return self.frame["outstanding"]

    @property
    def interest(self) -> pd.Series:
        return self.frame["interest"]

    @property
    def principal(self) -> pd.Series:
        return self.frame["principal"]

    @property
    def debt_service(self) -> pd.Series:
        return self.frame["debt_service"]


@dataclass
class BondBook:
    """Vectorized tranche book. All per-tranche arrays have length N.

    issue_period / maturity_period : integer grid indices (may lie off-grid)
    face                            : par, $ millions
    coupon                          : annual rate, decimal (0 for bills/FRN)
    bill_interest                   : $ millions, discount interest paid at maturity
    frn_spread                      : decimal, FRN spread over the 3m bill index
    kind                            : one of KINDS per tranche
    coupon_every                    : months between coupons (6 for Treasuries; 1 in the toy)
    """
    M: int
    issue_period: np.ndarray
    maturity_period: np.ndarray
    face: np.ndarray
    coupon: np.ndarray
    kind: np.ndarray
    bill_interest: np.ndarray
    frn_spread: np.ndarray
    coupon_every: int = 6
    meta: Optional[pd.DataFrame] = None      # original tranche rows (cusip, dates, ...)
    grid: Optional[MonthlyGrid] = None

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def from_toy(cls, issue_period, T_years, coupon, face) -> "BondBook":
        """The notebook's 3-bond toy on a semiannual integer grid: one period per
        coupon, so ``coupon_every=1`` and ``maturity = issue + 2*T``."""
        ip = np.asarray(issue_period, dtype=int)
        mp = ip + 2 * np.asarray(T_years, dtype=int)
        n = len(ip)
        return cls(M=int(mp.max()) + 1, issue_period=ip, maturity_period=mp,
                   face=np.asarray(face, float), coupon=np.asarray(coupon, float),
                   kind=np.array(["note"] * n), bill_interest=np.zeros(n),
                   frn_spread=np.zeros(n), coupon_every=1)

    @classmethod
    def from_auctions(cls, auctions: pd.DataFrame, grid: MonthlyGrid, cfg: BookConfig) -> "BondBook":
        """Map the auction ledger onto the grid. One row per tranche (reopenings
        stay separate: they add face on their own issue date)."""
        a = auctions.copy()
        a = a.dropna(subset=["issue_date", "maturity_date", "total_accepted"])
        kind = np.where(a["security_type"].eq("Bill"), "bill",
                np.where(a["inflation_index_security"], "tips",
                np.where(a["floating_rate"], "frn",
                np.where(a["security_type"].eq("Bond"), "bond", "note"))))
        a["kind"] = kind
        keep = np.ones(len(a), bool)
        if not cfg.include_tips:
            keep &= a["kind"].ne("tips").to_numpy()
        if not cfg.include_frn:
            keep &= a["kind"].ne("frn").to_numpy()
        if not cfg.include_cmb:
            keep &= ~a["cash_management_bill_cmb"].to_numpy()
        a = a[keep].reset_index(drop=True)

        # Reconciled against MSPD issued amounts on 2026-09-15 (note 91282CHY0, bond
        # 912810TK4, bill 912797SA6): total_accepted alone matches to <0.05%;
        # adding soma_accepted overstates issuance. SOMA add-ons are inside total_accepted.
        face = a["total_accepted"].fillna(0).to_numpy() / cfg.face_units
        coupon = np.where(a["kind"].isin(["note", "bond", "tips"]), a["int_rate"].fillna(0) / 100.0, 0.0)
        days = (a["maturity_date"] - a["issue_date"]).dt.days.to_numpy()
        # Bank-discount convention: interest = face x rate x days/360, paid at maturity.
        bill_interest = np.where(a["kind"].eq("bill"),
                                 face * a["high_discnt_rate"].fillna(0).to_numpy() / 100.0 * days / 360.0, 0.0)
        frn_spread = np.where(a["kind"].eq("frn"), a["spread"].fillna(0) / 100.0, 0.0)

        book = cls(M=grid.M,
                   issue_period=grid.index_of(a["issue_date"]),
                   maturity_period=grid.index_of(a["maturity_date"]),
                   face=face, coupon=coupon, kind=a["kind"].to_numpy(),
                   bill_interest=bill_interest, frn_spread=frn_spread,
                   coupon_every=cfg.coupon_months, meta=a, grid=grid)
        obs.event(channel="engine", kind="book.built", n_tranches=book.N,
                  by_kind={k: int((book.kind == k).sum()) for k in KINDS},
                  face_total_mm=float(face.sum()))
        return book

    # ── basic properties ───────────────────────────────────────────────────
    @property
    def N(self) -> int:
        return len(self.face)

    @property
    def periods(self) -> np.ndarray:
        return np.arange(self.M)

    def _t(self) -> np.ndarray:
        """(1, M) period row for broadcasting against (N, 1) tranche columns."""
        return self.periods[None, :]

    # ── dense (N, M) representation: the README's math, verbatim ───────────
    def dense_active(self) -> np.ndarray:
        return (self._t() > self.issue_period[:, None]) & (self._t() <= self.maturity_period[:, None])

    def dense_outstanding_mask(self) -> np.ndarray:
        return (self._t() >= self.issue_period[:, None]) & (self._t() < self.maturity_period[:, None])

    def dense_coupon_month(self) -> np.ndarray:
        """True where a coupon is due: every ``coupon_every`` months counting back from maturity."""
        return ((self.maturity_period[:, None] - self._t()) % self.coupon_every) == 0

    def dense_coupon_flow(self) -> np.ndarray:
        per_coupon = self.face * self.coupon / (12.0 / self.coupon_every) if self.coupon_every != 1 \
            else self.face * self.coupon / 2.0          # toy convention: semiannual grid, C/2 per period
        return (self.dense_active() & self.dense_coupon_month()) * per_coupon[:, None]

    def dense_principal_flow(self) -> np.ndarray:
        return (self._t() == self.maturity_period[:, None]) * self.face[:, None]

    def dense_bill_interest_flow(self) -> np.ndarray:
        return (self._t() == self.maturity_period[:, None]) * self.bill_interest[:, None]

    def dense_outstanding(self) -> np.ndarray:
        return self.dense_outstanding_mask() * self.face[:, None]

    # ── sparse aggregates: same sums, O(pairs) instead of O(N x M) ─────────
    def _per_coupon(self) -> np.ndarray:
        if self.coupon_every == 1:
            return self.face * self.coupon / 2.0
        return self.face * self.coupon / (12.0 / self.coupon_every)

    def _window_pairs(self, lo: np.ndarray, hi: np.ndarray, step: int = 1, anchor: Optional[np.ndarray] = None):
        """(row, col) pairs for every tranche's columns in [lo, hi), clipped to the
        grid, optionally only columns congruent to ``anchor`` modulo ``step``.
        Vectorized with repeat/cumsum, no Python loop over tranches."""
        lo_c = np.clip(lo, 0, self.M)
        hi_c = np.clip(hi, 0, self.M)
        if step > 1:
            # first column >= lo_c that is congruent to anchor (mod step)
            off = (anchor - lo_c) % step
            lo_c = lo_c + off
        counts = np.maximum((hi_c - lo_c + step - 1) // step, 0)
        rows = np.repeat(np.arange(self.N), counts)
        starts = np.repeat(lo_c, counts)
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        cols = starts + within * step
        return rows, cols

    def _bincount(self, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        w = weights[rows]
        if mask is not None:
            w = w * mask[rows]
        return np.bincount(cols, weights=w, minlength=self.M)

    def aggregates(self, short_rate: Optional[np.ndarray] = None, by_kind: bool = True) -> Aggregates:
        """Column sums ($ millions) per period: outstanding, interest (coupon,
        bill discount, FRN accrual), principal, debt service. ``short_rate`` is
        an (M,) annual decimal path of the 3m bill rate for FRN accrual; if
        None, FRN interest is zero and flagged in the frame's attrs."""
        # Outstanding: columns [issue, maturity)
        r_o, c_o = self._window_pairs(self.issue_period, self.maturity_period)
        # Coupons: columns in (issue, maturity], congruent to maturity mod coupon_every
        r_c, c_c = self._window_pairs(self.issue_period + 1, self.maturity_period + 1,
                                      step=self.coupon_every, anchor=self.maturity_period)
        # Maturity column, on grid only
        on = (self.maturity_period >= 0) & (self.maturity_period < self.M)
        r_m = np.arange(self.N)[on]
        c_m = self.maturity_period[on]

        per_coupon = self._per_coupon()
        out = {
            "outstanding": self._bincount(r_o, c_o, self.face),
            "interest_coupon": self._bincount(r_c, c_c, per_coupon),
            "interest_bill": self._bincount(r_m, c_m, self.bill_interest),
            "principal": self._bincount(r_m, c_m, self.face),
        }
        frn = (self.kind == "frn")
        if short_rate is not None and frn.any():
            # monthly accrual on outstanding FRN face: face x (r_3m + spread) / 12
            w = self.face * frn
            base = np.bincount(c_o, weights=w[r_o], minlength=self.M) * np.asarray(short_rate) / 12.0
            spr = np.bincount(c_o, weights=(w * self.frn_spread)[r_o], minlength=self.M) / 12.0
            out["interest_frn"] = base + spr
        else:
            out["interest_frn"] = np.zeros(self.M)
        out["interest"] = out["interest_coupon"] + out["interest_bill"] + out["interest_frn"]
        out["debt_service"] = out["interest"] + out["principal"]
        if by_kind:
            for k in KINDS:
                km = (self.kind == k).astype(float)
                out[f"outstanding_{k}"] = self._bincount(r_o, c_o, self.face, km)
        frame = pd.DataFrame(out, index=pd.RangeIndex(self.M, name="period"))
        if self.grid is not None:
            frame.index = self.grid.dates
            frame.index.name = "date"
        frame.attrs["frn_rate_supplied"] = short_rate is not None
        obs.event(channel="engine", kind="book.aggregates", n_pairs_outstanding=int(len(r_o)),
                  n_pairs_coupon=int(len(r_c)))
        return Aggregates(frame)

    # ── one future cash-flow vector (Problem 5's TotalFutureCF) ────────────
    def total_future_cf(self, as_of_period: int, short_rate: Optional[np.ndarray] = None) -> np.ndarray:
        """Debt service per period for periods > as_of_period, zeros before, from
        tranches already on the book. This is what a scenario matrix prices."""
        cf = self.aggregates(short_rate=short_rate, by_kind=False).frame["debt_service"].to_numpy().copy()
        cf[: as_of_period + 1] = 0.0
        return cf
