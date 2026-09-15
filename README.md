# Bond_Sim

**Central question:** how likely is the US actually hitting a sovereign-debt
doom loop (debt growth pushes rates up, higher rates push debt service and
future debt growth up further), what stops it, who gets hurt if it is not
stopped (which industries, which normal people), and how long does that hurt
last?

Bond_Sim is the research engine behind that thesis. It reconstructs the US
Treasury's marketable debt bond by bond from the auction ledger, pulls every
macro input as point-in-time vintages, estimates how rates, growth, the
primary balance, and unemployment move together, and runs a controlled Monte
Carlo in which the yield reacts to the simulated debt path and five policy
responses are compared on identical shocks. Every artifact carries the hash
of the configuration that produced it.

## Quick start: the notebooks are the working surface

```bash
/usr/bin/python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev,regimes]"
cp .env.example .env               # FRED_API_KEY=...  (free key from fred.stlouisfed.org)
./.venv/bin/python -m pytest        # 23 tests incl. the no-lookahead invariance test
./.venv/bin/jupyter lab notebooks   # open 00_index.ipynb and go phase by phase
```

| Notebook | Phase |
|---|---|
| `00_index` | map, standing rules, the parameter register |
| `01_data_and_vintages` | what was known when: ALFRED vintages, the Treasury ledger, probed facts |
| `02_bond_book` | every tranche on one grid, reconciled to Treasury publications; the effective rate |
| `03_dependence_factors_states` | correlations, regimes, block factors, the latent behavioral chain |
| `04_sustainability` | the identity, not a threshold: pb*, g*, decomposition, feasible pb, trigger on history |
| `05_simulation` | VAR block vs state block on identical debt accounting; distributions; admissibility |
| `06_policies_and_shocks` | five policies x four shocks on common random numbers; act now vs later |
| `07_who_gets_hurt_how_long` | sector employment paths, recovery half-lives |

Each notebook has a parameters cell (`AS_OF`, `K`, `FORCE_REFRESH`); change
it and rerun. Expensive steps are memoized in `runs/notebook_cache/`. The
notebooks are generated from `scripts/build_notebooks.py` so they never
drift from the package API; the mathematics is in `docs/math.md`.

The CLI (`bond_sim fetch|book|analyze|simulate`) remains as an optional
batch wrapper around the same package. `main.ipynb` is the original
hand-coded warm-up (Problems 1-5 on a toy 3-bond book); `tests/test_toy_book.py`
pins its numbers so the engine can never drift from what was hand-verified.

## Architecture

```
src/bond_sim/
  calendar.py        MonthlyGrid: the one axis every array shares (decision 0001)
  config.py          frozen typed config from YAML, content hash on every output
  obs.py             JSONL event spine (durations, counts, status; secrets redacted)
  decorators.py      @timed  @disk_cache  @point_in_time (no-lookahead guard)
  data/
    pit.py           VintageFrame: "what was known on date X" as one boolean mask
    fred.py          FRED/ALFRED client, full vintage history, 39 verified series
    fiscaldata.py    Treasury Fiscal Data: auctions, MSPD, interest expense, debt to the penny
  engine/
    book.py          BondBook: (N tranches x M months); dense masks for attribution,
                     O(pairs) bincount aggregates for speed, proven equal by test
    discount.py      path-dependent discount factors; (K,M) @ (M,) present values
    validate.py      reconciliation vs MSPD Table 1, interest expense, debt to the penny
  analysis/
    correlation.py   full / rolling / regime-split correlation, lead-lag scan,
                     Granger both ways at BIC lag with Benjamini-Hochberg FDR
    regimes.py       quantile or HMM regimes, adjusted-Rand consistency, regime lead/lag
    factors.py       block factors: one interpretable state variable per behavioral block,
                     idiosyncratic residuals kept as the series-level shock
  sim/
    macro.py         quarterly levels VAR (estimated shock covariance), the benchmark block
    states.py        latent Markov chain on the factors: HMM states, within-state VAR(1),
                     residual bootstrap by state, stress hazard, forced transitions, impulses
    macroblock.py    one interface over both blocks so the engine runs on either
    premium.py       endogenous risk premium: none / linear / threshold  (P-01, P-02)
    doomloop.py      monthly debt accounting x quarterly macro block, premium feedback
    sustainability.py the identity: pb*, g*, decomposition, feasible pb, debt limit, trigger
    admissible.py    reject impossible paths, never clip; every rejection reported
    policy.py        status quo, no-layoff mandate, austerity, monetization, growth-led; Delayed
    labor.py         sector employment betas estimated from history
    recovery.py      half-lives from historical episodes and simulated paths
    setup.py         InitialState from data (stock maturity, bill share, receipts/GDP)
  notebook.py        one call per step for the phase notebooks
  cli.py             fetch | book | analyze | simulate | all (optional batch wrapper)
```

## Data (all verified live against the source APIs on 2026-09-15)

| Source | What | Point-in-time rule |
|---|---|---|
| Fiscal Data `/v1/accounting/od/auctions_query` | every auction since 1979-11: issue/maturity dates, coupon, bill discount rate, amounts, TIPS/FRN/CMB flags | known on `auction_date` |
| Fiscal Data `/v1/debt/mspd/mspd_table_1`, `_3_market` | Monthly Statement of the Public Debt, 2001-01+: outstanding by class and by CUSIP | published ~4th business day after month end |
| Fiscal Data `/v2/accounting/od/interest_expense` | monthly accrued interest expense by security type, 2010-05+ | month end + lag |
| Fiscal Data `/v2/accounting/od/debt_to_penny` | daily debt held by the public / intragovernmental / total, 1993+ | record date + lag |
| FRED/ALFRED, 39 series (`data/fred.py::SERIES`) | fiscal stock and flow, GDP, CPI, yields, term premium, holders, unemployment, sector employment, JOLTS, claims | full vintage history; pre-vintage rows backfilled with a conservative publication lag and flagged (decision 0002) |

Established while building (each cost a probe, none was assumed): auction
`total_accepted` already includes SOMA add-ons; MSPD's CUSIP column also holds
subtotal labels; ALFRED caps requests at 2000 vintage dates so daily series
are chunked; ALFRED vintages begin 1991 (GDP) and 2005 (DGS10).

## Research plan and status

| Phase | Content | Status |
|---|---|---|
| 1 | Cash flow schedule on the toy book (Problems 1-3) | done, pinned by tests |
| 2 | Static-scenario sensitivity (Problems 4-5) | done, pinned by tests |
| 3 | Real data: auction ledger -> BondBook on the monthly grid; validation vs MSPD, interest expense, debt to the penny | built; notebook 02 |
| 4 | Doom-loop engine: two macro blocks (levels VAR; block factors + latent chain) on identical monthly debt accounting, pluggable premium, identity-based trigger, admissibility filter | built; notebooks 03-05; premium form and slope are P-01/P-02 |
| 5 | Correlated labor and industry shocks: sector betas on (d_u, d_r10) with residual covariance | built; notebook 07 |
| 6 | Policy counterfactuals and shocks under common random numbers: five policies x four shocks; act now vs later | built; notebook 06; every magnitude is P-06..P-09 |
| 7 | Recovery duration: half-lives from five US episodes and from simulated paths | built; notebook 07 |
| 8 | Tests: toy pins, calendar, no-lookahead invariance, sustainability identity, latent chain, sim smoke | 23 passing |
| 9 | Paper: real data, assumptions separated from findings, in-repo `paper/` | not started |

Everything a human must decide is in `PARAMETERS.md` (P-01..P-18) with its
code location and thesis impact. Design decisions with their evidence are in
`docs/decisions/` (0001 grid, 0002 vintages, 0003 dependence, 0004 instruments,
0005 VAR, 0006 identity trigger, 0007 latent states and admissibility).

## Engineering discipline

* **No look-ahead.** Every loader takes `as_of`; `@point_in_time` asserts
  nothing returned postdates it; `tests/test_no_lookahead.py` proves an
  as_of view is invariant to appended future vintages.
* **No independent parameters.** Shock structure is estimated (VAR residual
  covariance, sector residual covariance), never assumed; every pair of
  inputs gets rolling, regime, lead/lag, and Granger diagnostics.
* **Two representations, one truth.** The bond book's dense (N, M) masks and
  its O(pairs) aggregates are asserted equal in tests.
* **No threshold.** The doom loop is defined by the accounting identity
  (stabilizing primary balance above what is feasible for N periods), not by
  a chosen debt level.
* **Filtered, not clipped.** Impossible paths are rejected and counted;
  admissible paths are never truncated at bounds.
* **Reproducible.** Config hash + seed reproduce every number; policies and
  shocks share the same random numbers.
* **Observable.** Every fetch, fit, and run emits a JSONL event with
  duration, counts, and status; credentials are redacted at the spine.

This is research software, not investment advice.
