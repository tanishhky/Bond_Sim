# 0004: What the bond book models exactly, approximately, and not at all

**Decision (2026-09-15).** The book (`engine/book.py`) is built tranche by
tranche from the auction ledger (11,113 auctions since 1979-11, verified live)
and reconciled against the Monthly Statement of the Public Debt (2001-01+).
The following are the deliberate simplifications, each with the evidence
that sized it.

**Exact.**
* Issuance face = auction `total_accepted`. Reconciled on 2026-09-15 against
  MSPD `issued_amt`: note 91282CHY0 44,000.0 vs 43,999.7; bond 912810TK4
  40,952.9 vs 40,939.9; bill 912797SA6 334,154.4 vs 334,135.4 ($mm). Adding
  `soma_accepted` overstated issuance by 5-6%, so SOMA add-ons are already
  inside `total_accepted`.
* Notes and bonds: fixed coupon `int_rate`, paid every 6 months anchored on
  the maturity month, principal at maturity. Reopenings are separate rows
  that add face on their own issue date.
* Bills and cash-management bills: discount instruments; interest = face x
  `high_discnt_rate` x days/360 (Treasury's bank-discount convention),
  booked in the maturity month with principal.

**Approximate (documented in the paper).**
* Odd first coupons (short or long first periods on back-dated reopenings)
  and the accrued interest buyers pay at reopenings are ignored. They net to
  approximately zero across the book and are second-order to a monthly
  interest total.
* FRNs accrue `(3m bill rate + spread) / 12` on face each month, with the
  rate path supplied by history (DGS3MO) or the simulation. Quarterly
  payment timing is smoothed to monthly accrual.
* TIPS are carried at nominal face with their fixed coupon. The inflation
  accrual on principal is not modeled; MSPD's `inflation_adj_amt` (387,588
  $mm on 2026-08-31, ~1.2% of marketable debt) sizes the omission for the
  validation report. Modeling it requires a CPI path in the simulator, which
  is a documented extension.

**Not modeled (known gaps, quantified by the validation report).**
* Treasury buybacks (2000-2002 and the 2024+ program) retire debt before
  maturity. MSPD `redeemed_amt` records them (bond 912810TK4: 1,300 $mm of
  40,940 issued). The book keeps such tranches outstanding to maturity, so
  it slightly overstates outstanding and interest for bought-back CUSIPs.
  The Fiscal Data buybacks dataset exists and is the fix.
* Non-marketable debt (savings bonds, SLGS, Government Account Series) is
  outside the auction ledger by construction. The thesis is about marketable
  debt held by the public; the debt-to-the-penny series provides the total
  for context.

**Validation (`bond_sim book --validate`).** Monthly reconstructed
outstanding by class vs MSPD per-CUSIP outstanding (2001+); monthly
reconstructed interest vs Fiscal Data interest expense by security type
(2010+); both reported as level and ratio series with the gaps above called
out, never hidden.
