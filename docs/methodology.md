# Methodology (working notes for the paper)

This file is written alongside the code and updated whenever a number
changes. Every figure below can be regenerated from the CLI; the run
directory is named by the configuration hash that produced it.

## 1. Question

How likely is a US sovereign-debt doom loop (debt growth -> higher yields ->
higher interest cost -> faster debt growth), what interrupts it, who is hurt
if nothing does, and for how long. "Doom loop" is operationalized as a
path-level event: debt/GDP or trailing-12-month interest/receipts crossing a
threshold (P-04) within the 30-year window.

## 2. Data

All sources are pulled by API, cached as raw vintage tables, and filtered by
an explicit `as_of` date at every call site.

**Treasury Fiscal Data.** The auction ledger (11,113 auctions 1979-11 to
2026-09) is the primitive: each tranche's issue date, maturity date, coupon
or bill discount rate, and accepted amount. The Monthly Statement of the
Public Debt (2001+), Interest Expense on the Public Debt (2010+), and Debt
to the Penny (1993+) are used only to validate the reconstruction, never to
build it.

**FRED/ALFRED, 39 series.** Full revision histories (decision 0002). The
vintage-coverage table (`bond_sim fetch`) records, per series, the first
true vintage date and the number of pre-vintage observations that carry a
backfilled publication timestamp. Headline facts from that table: GDP
vintages start 1991-12; daily yields 2005-07; monthly MTS fiscal series
2015-11; UNRATE 1960-04; payrolls 1955-06. Any lead/lag claim whose sample
is mostly pre-vintage is labeled as such.

## 3. The bond book

Each auction tranche is a row; each month since 1980-01 is a column
(decision 0001). Outstanding, coupon, bill discount interest, and principal
are boolean-mask arithmetic on that grid (README "Core design"), computed
two ways (dense masks and index-pair bincounts) that tests assert equal.

**Reconciliation, mean ratio book / published, last 5 years (2026-09-15 run):**

| Check | Ratio | Reading |
|---|---|---|
| Bills outstanding vs MSPD Table 1 | 1.008 | face convention correct |
| Notes | 1.005 | correct |
| Bonds | 1.015 | ~1.5% over: buybacks not modeled (decision 0004) |
| FRN | 0.995 | correct |
| TIPS | 0.830 | nominal face; MSPD includes ~17% inflation accrual (documented gap) |
| Total marketable | 0.994 | |
| Interest, 12-month rolling, vs Treasury accrued interest expense | 0.817 | cash coupons vs accrual; TIPS inflation compensation and accrual timing; to be decomposed by security type |

Two facts established by probing rather than assumed: auction
`total_accepted` already includes SOMA add-ons (adding `soma_accepted`
overstated issuance by 5-6% on every CUSIP checked), and MSPD's CUSIP column
also carries subtotal labels, which must be filtered by pattern.

## 4. Dependence structure (decision 0003)

Quarterly panel, differenced. For every cross-group pair (fiscal x rates x
macro x labor): full-sample correlation, 20-quarter rolling correlation and
its range, lead/lag scan over +/- 8 quarters, Granger causality both ways
at the BIC lag with Benjamini-Hochberg control over the whole grid.

**Findings from the 2026-09-15 run** (`runs/<hash>/correlation_report.csv`):

* Almost every thesis pair is regime-dependent: rolling-correlation ranges
  of 1.3-1.7 on a [-1, 1] scale are the norm, not the exception. One
  correlation number per pair would be wrong for all of them.
* Regime timing is pair-specific. Quantile regimes on the twelve widest
  rolling correlations agree with each other at a mean adjusted Rand index
  of 0.11. There is no single market-wide correlation regime to carry into
  the simulator; the VAR keeps one estimated covariance (P-12 stays off).
* Fed holdings of Treasuries (FDHBFRBN) Granger-cause, and are caused by,
  initial claims, unemployment, JOLTS layoffs, and construction and real
  estate employment (all surviving FDR): the QE episodes are recessions.
  Any premium model must not read Fed absorption as a free variable.
* The monthly deficit and initial claims are Granger-linked both ways at a
  12-quarter lag (automatic stabilizers), which is why the primary balance
  is inside the VAR rather than imposed.

## 5. Simulation (decision 0005)

Quarterly VAR in levels on (10y, 3m, nominal growth, trailing-4Q primary
balance, unemployment), estimated from 1985Q1. Levels rather than
differences because a difference-stationary state has no anchor and
random-walks rates and unemployment into their floors over 120 quarters
(observed in the first runs; see decision 0005). The estimated long-run
means and the companion-matrix eigenvalue are printed by every run; anchors
for the primary balance, the long-run 10y, and the natural rate are
explicit overrides (P-03, P-13, P-14), not defaults.

The fiscal risk premium (none / linear / threshold, P-01, P-02) is added on
top of the base 10y from last quarter's debt/GDP; its change is transmitted
to growth, the primary balance, and unemployment through the VAR's own
estimated coefficients on the lagged 10y. Debt accounting is monthly:
legacy interest and principal from the book, new issuance in a pool
refinanced at a bill/coupon blend of the current yields (bill share and
refinancing tenor from the outstanding stock, P-11), deficits financed at
that blend. Policies (status quo, no-layoff mandate, austerity,
monetization, growth-led) are hooks on the same shock draws.

Sector employment paths come from quarterly regressions of sector growth
on the unemployment change and lagged 10y changes, with the sectors'
residual covariance retained. Recovery half-lives are measured on five US
episodes and on every simulated path.

**Results:** see `runs/<hash>/sim_summary.csv` after `bond_sim simulate`.
Numbers are not transcribed here until the placeholder parameters in
`PARAMETERS.md` have been replaced or explicitly defended, because with
placeholders the headline probability is a property of the placeholders.

## 6. What the paper must state as limitations

TIPS inflation accrual, buybacks, odd first coupons, and reopening accrued
interest (decision 0004); constant VAR covariance; receipts/GDP held at its
last-12-month value; the linear premium's slope; no explicit term-premium
or foreign-demand channel (FDHBFIN is in the panel, not the loop); no
inflation variable in the VAR, so nominal and real growth are conflated
and the monetization scenario is not yet credible (P-15).

## 7. First real run (2026-09-15, placeholders in force, K = 2000)

Recorded so the effect of every later parameter decision is measurable
against it. VAR(3), max eigenvalue 0.961; long-run means 10y 4.05%, 3m
3.04%, nominal growth 5.48%, primary balance -1.85% of GDP, unemployment
5.25%. Initial state: debt/GDP 98.0%, 10y 4.83%, u 4.1%, trailing-4Q
primary balance -1.83%, stock average remaining maturity 72 months, bill
share 20%, receipts/GDP 16.6%.

| Policy | P(trigger) | median years to trigger | median 2056 debt/GDP | median 2056 10y | median 2056 u | median interest/GDP |
|---|---|---|---|---|---|---|
| status quo | 0.48 | 15.3 | 106% | 4.44% | 5.1% | 4.0% |
| no-layoff mandate | 0.79 | 10.9 | 209% | 5.27% | 5.1% | 8.2% |
| austerity | 0.00 | | 18% | 3.87% | 5.8% | 1.5% |
| monetization | 0.00 | | 0% | 13.3% | 1.3% | 0.2% |
| growth-led | 0.11 | 11.7 | 58% | 4.88% | 4.3% | 2.6% |

Reading: the status-quo path is a CBO-shaped projection (interest cost
rising toward 4% of GDP as low-coupon debt rolls off) and the trigger
probability is a property of the P-04 thresholds. The no-layoff mandate's
debt path is the placeholder growth penalty (P-06) compounding for thirty
years. Monetization's unemployment floor is the missing inflation variable
(P-15), not a result.
