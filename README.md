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

## Build plan

Working entirely in `main.ipynb`, dummy 3-bond toy dataset first, real data
last. LeetCode-style: each step gets a markdown problem statement (statement,
constraint, input, expected output) in the notebook, then it's coded by hand
against that spec, no code written by the assistant unless explicitly asked.

**Phase 1, cash flow schedule:**
1. Toy bond inputs: `issue_period`, `T`, `C`, `F` — done
2. `maturity_period = issue_period + 2*T` — done
3. Global calendar grid `M = maturity_period.max() + 1`, `periods = arange(M)` — done
4. `Active[i,t]` mask via broadcasting (`periods[None,:]` vs `issue_period[:,None]`/`maturity_period[:,None]`) — **posed as Problem 1 in `main.ipynb`, not yet solved** (current cell is a placeholder nested loop that always assigns `True`)
5. `CouponFlow[i,t]` — not started, will be Problem 2 once Problem 1 is solved
6. `PrincipalFlow[i,t]` — not started, part of Problem 2
7. `Outstanding[i,t]` — not started, part of Problem 2
8. Column sums → ongoing debt / ongoing interest payments / total debt service, hand-check against the toy dataset — not started

**Phase 2, yield impact / sensitivity:**
9. Discount-factor scenario matrix (K rate scenarios × M periods) — not started
10. Collapse bond dimension into one aggregate future cash flow vector — not started
11. `DF_scenarios (K,M) @ TotalFutureCF (M,)` for the sensitivity profile — not started

**Phase 3, swap dummy for real data:**
12. Pull Treasury auction history from `fiscaldata.treasury.gov` — not started
13. Map raw fields onto `issue_period`/`T`/`C`/`F` — not started
14. Branch for T-bills (no coupon, single discount payment) — not started

**Phase 4, tests:**
15. `pytest` cases pinned to the toy dataset's hand-computed numbers — not started

## Status

Phase 1 in progress. `main.ipynb` has the toy dataset, `N`, `maturity_period`,
`M`/`periods` built and verified by hand, and Problem 1 (`Active` mask) posed
and awaiting a correct broadcasted (non-loop) solution.
