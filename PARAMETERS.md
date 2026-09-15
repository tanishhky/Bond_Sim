# Parameters that need a human decision

Everything below is a judgment call, not something the data settles. Each has
a working placeholder so the pipeline runs end to end today; none of the
placeholders should appear in the paper without being replaced or defended.
Code locations are given so a decision is a one-line change.

| ID | Parameter | Placeholder | Where | Why it matters |
|---|---|---|---|---|
| P-01 | Risk-premium slope (bps of 10y yield per 1 pt of debt/GDP above the anchor) | 2.0 | `config/default.yaml` `doomloop.risk_premium.bps_per_pct_debt_gdp` | The single number the doom-loop probability is most sensitive to. Literature range is wide (roughly 2-5 bps per pt in most estimates for advanced economies; higher for emerging markets). The paper should sweep it, not assert it. |
| P-02 | Threshold-model kink and extra slope | 130% debt/GDP, +6 bps/pt | `doomloop.risk_premium.threshold_*` | Only used if `model: threshold`. Encodes the nonlinearity claim (Reinhart-Rogoff style). Needs a cited basis or should stay as a sensitivity case. |
| P-03 | Primary-deficit anchor path | VAR mean (historical average pb) | `sim/macro.py` (VAR intercept) | The VAR mean-reverts pb to its historical average. A CBO-baseline anchor (deficits ~3% of GDP ex-interest) is the alternative and changes the level of every debt path. |
| P-04 | Doom-loop trigger definition | debt/GDP >= 150% or interest/receipts >= 30%; receipts/GDP anchor 17% | `doomloop.loop_trigger_*`, `sim/doomloop.py` `receipts_share` | "How likely is the doom loop" is only meaningful once "doom loop" has a threshold. Both numbers are round placeholders. |
| P-05 | Sector employment betas: estimated vs literature | estimated from history (`labor.beta_source`) | `sim/labor.py` | Estimated betas are honest but noisy for small sectors; literature elasticities are cleaner but imported. Report both. |
| P-06 | No-layoff mandate mechanics | floor at 0 pt/qtr rise in u during stress; growth penalty 0.5 pct pts per pt of prevented unemployment; stress = premium > 0 and u rising over 4 quarters; 24-month window | `sim/policy.py` `NoLayoffMandate`, `policy.no_layoff_*` | This is the policy the thesis was motivated by. Every number here is invented; the mechanism choice (floor vs propensity dampener) is a modeling stance. |
| P-07 | Austerity target and fiscal multiplier | primary surplus target 2% of GDP over a 12-quarter ramp; multiplier 1.0 | `policy.austerity_primary_balance_pct_gdp`, `Austerity.multiplier` | Multiplier size decides whether austerity helps or deepens the loop (the Greek debate). |
| P-08 | Monetization share and inflation cost | Fed absorbs 50% of net issuance; +2 pct pts nominal growth | `policy.monetization_share`, `Monetization.inflation_uplift_pct` | The inflation cost is what makes this a tradeoff rather than a free lunch; it is not derived from anything yet. |
| P-09 | Growth-led uplift | +0.5 pct pts trend growth | `policy.growth_uplift_pct` | Encodes an optimistic scenario; should be tied to a productivity/immigration argument. |
| P-10 | Recovery episodes | Volcker 1981, early 1990s, dot-com 2001, GFC 2008, Covid 2020 | `sim/recovery.py` `EPISODES` | Which historical recoveries are comparable to a fiscal-crisis downturn. Greece needs a non-FRED source. |
| P-11 | Refinancing tenor of the debt stock | estimated: face-weighted average remaining maturity of the outstanding stock at as_of (`sim/setup.py`) | `InitialState.avg_new_maturity_months` (override) | Sets how fast the stock reprices to new yields. Gross-issuance-weighted term (~15 months, bills dominate) was rejected as the wrong object; the stock's remaining maturity is what governs repricing. |
| P-12 | Regime-switching covariance | off | `sim/macro.py` (single covariance) | The 2026-09-15 analysis found pair-specific regimes (mean adjusted Rand 0.11 across the twelve widest pairs), so a single covariance is the evidence-backed default for now. |
| P-13 | Long-run 10y anchor | VAR sample mean 1985-2026 (reported at run time) | `doomloop.r10_anchor_pct` | The levels VAR mean-reverts the base 10y to this. A view that r* has shifted belongs here, explicitly. |
| P-14 | Natural rate of unemployment | VAR sample mean 1985-2026 | `doomloop.u_anchor_pct` | Same logic for unemployment; CBO's NAIRU estimate is the natural candidate. |
| P-15 | Inflation as a state variable | not in the VAR | `sim/macro.py` `VARS` | The VAR is estimated on *nominal* growth with no inflation variable, so the monetization policy's inflation uplift behaves like real growth (unemployment falls to its floor, first real run). Adding CPI inflation (CPIAUCSL is already in the panel) as a sixth state separates real from nominal and is the prerequisite for a credible monetization scenario. |

## How to record a decision

1. Change the value in `config/default.yaml` (or the named code location).
2. Add a line to `docs/decisions/` explaining the source (paper, expert, own
   estimate) and the sensitivity range the paper will report.
3. Rerun `bond_sim simulate`; every artifact carries the config hash so the
   old and new runs stay distinguishable.
