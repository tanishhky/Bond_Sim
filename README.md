# Bond_Sim

Bottom-up simulator for the full history of US Treasury issuance: reconstructs
ongoing debt stock and ongoing interest payments from individual bond-level
cash flow schedules, and stress-tests the outstanding stock against simulated
future rate paths.

## Core design

Every bond gets its own row. Everything is indexed against one shared
calendar axis (not periods-since-issuance) so cash flows can be summed across
bonds at any given date.

**Per-bond inputs (length N, one row per issue):**
- `issue_period[i]` — calendar period the bond was auctioned
- `T[i]` — original maturity length in years
- `C[i]` — coupon rate set at that bond's own auction (fixed forever)
- `F[i]` — face value issued

**Cash flow matrix, shape (N bonds, M calendar periods), built via broadcasting:**
- `Active[i,t] = (t > issue_period[i]) & (t <= maturity_period[i])`
- `CouponFlow[i,t] = Active[i,t] * (C[i]/2 * F[i])`
- `PrincipalFlow[i,t] = (t == maturity_period[i]) * F[i]`
- `Outstanding[i,t] = (t >= issue_period[i]) & (t < maturity_period[i])`, times `F[i]`

Column sums give the aggregate series:
- `Outstanding[:, t].sum()` — total ongoing debt at date t (stock)
- `CouponFlow[:, t].sum()` — total ongoing interest payment at date t (flow, excludes principal)
- `CouponFlow[:, t].sum() + PrincipalFlow[:, t].sum()` — total debt service at date t

**Yield impact / sensitivity:** collapse the bond dimension into one future
cash flow vector, then price it under a scenario matrix of simulated/shocked
rate paths in a single matmul: `DF_scenarios (K, M) @ TotalFutureCF (M,)`.
Only expand to a 3D (N, K, M) tensor if per-bond sensitivity attribution is
needed.

## Data source

Per-auction issue date, maturity date, coupon (interest rate), high yield,
and amount accepted: `fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/`.
T-bills in that dataset carry no coupon (pure discount, single payment at
maturity) and need a separate branch from the coupon-bond formula above.

## Status

Scaffolding only. Design and math worked out, implementation not started.
