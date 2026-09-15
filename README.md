# Bond_Sim

**Central question:** how likely is the US actually hitting a sovereign-debt
doom loop (debt growth pushes rates up, higher rates push debt service and
future debt growth up further), what stops it, who gets hurt if it is not
stopped (which industries, which normal people), and how long does that hurt
last?

Bond_Sim started as a bottom-up simulator of US Treasury issuance
(bond-level cash flow schedules, reconstructed ongoing debt stock and
interest payments) and is being extended into the engine behind that
thesis: a controlled Monte Carlo where the discount rate is not handed in
from outside, it is a feedback function of the simulated debt trajectory
itself; correlated shocks to rate-sensitive industries and employment are
baked into every path instead of bolted on after; and a menu of policy
responses, including a no-layoff mandate, are run through the identical
engine so their costs and benefits are actually comparable. The target
output is a real research paper: real data, explicit assumptions, honest
uncertainty, not a demo.

## Core design

Every bond gets its own row. Everything is indexed against one shared
calendar axis (not periods-since-issuance) so cash flows can be summed across
bonds at any given date.

**Per-bond inputs (length N, one row per issue):**
- `issue_period[i]`: calendar period the bond was auctioned
- `T[i]`: original maturity length in years
- `C[i]`: coupon rate set at that bond's own auction (fixed forever)
- `F[i]`: face value issued

**Cash flow matrix, shape (N bonds, M calendar periods), built via broadcasting:**
- `Active[i,t] = (t > issue_period[i]) & (t <= maturity_period[i])`
- `CouponFlow[i,t] = Active[i,t] * (C[i]/2 * F[i])`
- `PrincipalFlow[i,t] = (t == maturity_period[i]) * F[i]`
- `Outstanding[i,t] = (t >= issue_period[i]) & (t < maturity_period[i])`, times `F[i]`

Column sums give the aggregate series:
- `Outstanding[:, t].sum()`: total ongoing debt at date t (stock)
- `CouponFlow[:, t].sum()`: total ongoing interest payment at date t (flow, excludes principal)
- `CouponFlow[:, t].sum() + PrincipalFlow[:, t].sum()`: total debt service at date t

**Static-scenario sensitivity (Phase 2, the toy warm-up):** collapse the
bond dimension into one future cash flow vector, then price it under a
matrix of flat, exogenous what-if rates in a single matmul:
`DF_scenarios (K, M) @ TotalFutureCF (M,)`. This is mechanics practice, not
the real rate model, real rates do not stay flat and do not move
independently of the debt path. The real model is Phase 4 below, where the
rate is endogenous to the simulation instead of handed in.

## Data sources

Layered in as each phase actually starts, not pulled all at once:

- **Treasury auction history** (Phase 3): per-auction issue date, maturity
  date, coupon, high yield, amount accepted, from
  `fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/`.
  T-bills carry no coupon (pure discount, single payment at maturity) and
  need a separate branch from the coupon-bond formula above.
- **Fiscal trajectory** (Phase 4): federal debt held by the public,
  debt/GDP, deficit/GDP, federal outlays and receipts, real GDP. Likely
  FRED plus Treasury/OMB historical tables and CBO baseline projections for
  the forward path. Exact series IDs to confirm once this phase actually
  starts rather than guessed now.
- **Sector employment** (Phase 5): employment by industry (construction,
  real estate, autos, regional banks and small business lending, federal
  contractors) to fit or calibrate rate sensitivity, most likely BLS/FRED
  sector series.
- **Historical crisis recoveries** (Phase 7): NBER recession dates and
  unemployment-duration series, plus at least one cross-country
  sovereign-debt-crisis episode (Greece in the 2010s is the obvious modern
  comparable), as priors for how long a doom-loop-driven downturn actually
  lasts.

## Research plan

Working entirely in `main.ipynb`. LeetCode-style through Phase 3: each step
gets a markdown problem statement (objective, explanation, constraint,
input, expected output), Tanishk codes every solution by hand, no code from
Claude unless explicitly asked. Phase 4 onward is genuine research design,
not a coding exercise: methodology gets written up and agreed before it
gets coded.

**Phase 1, cash flow schedule: DONE.**
Toy 3-bond dataset; `Active`/`CouponFlow`/`PrincipalFlow`/`Outstanding`
masks; column-summed to ongoing debt stock, interest, and total debt
service. Problems 1-3 in `main.ipynb`.

**Phase 2, static-scenario sensitivity: DONE.**
`DF_scenarios (K,M) @ TotalFutureCF (M,)`, three flat what-if rate
scenarios (down/base/up) on the toy debt service vector. Problems 4-5.

**Phase 3, swap dummy for real data: not started.**
1. Pull Treasury auction history from `fiscaldata.treasury.gov`.
2. Map raw fields onto `issue_period`/`T`/`C`/`F`.
3. Branch for T-bills (no coupon, single discount payment).
4. Re-run Phase 1-2's checks against the real debt stock as a sanity check
   before building anything new on top of it.

**Phase 4, endogenous doom-loop rate engine: not started, the load-bearing phase.**
Replace Phase 2's flat exogenous scenarios with a rate that reacts to the
simulated debt path: each period's yield = a base/neutral rate plus a risk
premium driven by the fiscal trajectory (debt/GDP, deficit/GDP, interest
cost/revenue, or similar). Newly issued debt funds that period's deficit
(interest cost plus a primary-deficit draw), so debt compounds through the
same function that is pricing it; that compounding is the actual loop, not
a metaphor. Output is a distribution across many simulated paths: does a
loop take hold, how fast, how bad.

**Open methodology call, needs a decision before this gets coded:** the
exact functional form for the risk-premium feedback. Candidates: (a)
reduced-form, fit a historical or cross-country relationship between
debt/GDP and term premium; (b) a Reinhart-Rogoff-style threshold
calibration; (c) a simpler expert-calibrated slope, explicitly labeled as
an assumption, with a documented sensitivity analysis on that slope.
Recommend starting with (c), the simplest defensible version, and
stress-testing how much the thesis's conclusions actually depend on that
one number, rather than chasing false precision on a parameter nobody can
estimate cleanly.

**Phase 5, correlated labor and industry shocks: not started.**
For every simulated path, translate the realized rate/debt trajectory into
sector-level employment and output impact (construction and real estate,
autos, small business and regional banks, federal contractors exposed to
spending cuts), via fitted historical sensitivities where FRED/BLS data
supports it, literature-calibrated elasticities where it does not,
documented either way. Correlated on purpose: a real doom-loop scenario
hits a rate shock, tighter credit, government spending cuts, and weaker
consumer demand together, drawing sector shocks independently would
understate how bad and how widespread it actually gets. Output is a
sector-by-sector impact series per path, not one aggregate number, so "who
gets hurt" is actually answerable.

**Phase 6, policy counterfactuals: not started.**
Re-run the identical controlled MC engine, same underlying shock draws,
under a menu of policy responses so differences reflect the policy and not
randomness:
1. Status quo, no intervention (the baseline doom-loop path).
2. No-layoff mandate during a defined stress window: model the intended
   effect (unemployment suppressed) against the likely offsetting cost
   (business failures, hiring frozen elsewhere, possible
   labor-hoarding-driven inflation).
3. Fiscal consolidation / austerity: spending cuts and/or tax increases
   that shrink the primary deficit directly.
4. Debt monetization: the Fed absorbs issuance and inflates away the real
   debt burden, traded off against the inflation/currency cost.
5. Growth-led consolidation: higher trend GDP growth (productivity,
   immigration) outrunning the debt, the best case and the hardest to
   engineer on purpose.

**Open methodology call:** exactly how to mechanically model the no-layoff
mandate, a hard floor on sector employment with a cost penalty elsewhere,
or a probabilistic dampener on layoff propensity. Judgment call, not
something derivable from data alone.

**Phase 7, recovery duration: not started.**
Calibrate how fast unemployment and output actually mean-revert after a
downturn using historical analogues (the 2008 recovery, the Volcker-era
disinflation recession, Greece in the 2010s) as priors, apply to every
simulated and policy path to answer "how long does the suffering last," not
just "how bad does it get."

**Phase 8, tests: not started.**
`pytest` cases pinned to the toy dataset's hand-computed numbers (existing
scope), plus deterministic unit tests on the Phase 4-6 mechanics (the
feedback loop's arithmetic, the policy branching logic), not on stochastic
MC outputs directly.

**Phase 9, writeup: not started.**
The actual paper. Real data throughout, every assumption from Phases 4-6
listed explicitly and kept separate from genuine findings, figures, in-repo
`paper/` following the same pattern as RateWalk's working paper. No SSRN
claim or submission until it is actually submitted.

## Status

Phase 1 and Phase 2 are done on the toy dataset (Problems 1-5 in
`main.ipynb`, all executing end to end, outputs verified against
hand-computed values). Phase 3 (real Treasury data) is next.
