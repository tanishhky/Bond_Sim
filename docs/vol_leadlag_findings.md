# Realized-vol lead-lag findings (Phase 1, 2026-09-21)

`scripts/vol_leadlag_analysis.py`, as_of 2026-07-01, 143 quarters (1990-04-01 to 2025-10-01). CorrelationAnalyzer, quarterly, level transform (all four series are already difference-like), BH-FDR alpha 0.05, lead/lag scan +/- 24 months.

## Question

Does realized rate volatility (physical measure, computed from data already in the pipeline, no new source) lead the model's own fiscal-stress diagnostics (the identity trigger's gap, the HMM's decoded stress state), or does it just move alongside them? A genuine lead would mean the bond market's realized behavior anticipates stress before the accounting identity does; no lead is a real, reportable null, not a failure.

## Headline

Ten-year realized vol genuinely leads the fiscal-identity gap (rvol10->gap, 3-quarter best lag, FDR-significant, p=0.023): the bond market's realized behavior moves ahead of the pure accounting-identity measure of stress, the hypothesis this phase set out to test, and it survives correction. But the relationship with the HMM's own stress-state classification runs the *other* way: stress leads rvol10 (p=0.002), not the reverse (p=0.837), and stress also leads the gap (p=0.017), not the reverse (p=0.548). Read together, the behavioral/financial-conditions regime the state block already estimates looks like the most upstream signal of the three, it moves before both the slow accounting math and realized rate volatility, not after. rvol3m->stress (p=0.041) does NOT survive BH-FDR correction (`granger_a_to_b_sig` False), the same multiple-comparisons lesson the S&P 500 project's retracted "297 pairs" finding already carries: an uncorrected p just under 0.05 in a 12-test grid is not evidence.

## Result

| a      | b      |   full_corr |   best_lag |   best_lag_corr | leads   |   granger_a_to_b_p | granger_a_to_b_sig   |   granger_b_to_a_p | granger_b_to_a_sig   |
|:-------|:-------|------------:|-----------:|----------------:|:--------|-------------------:|:---------------------|-------------------:|:---------------------|
| gap    | stress |      0.3546 |         -3 |          0.4239 | stress  |             0.5482 | False                |             0.0175 | True                 |
| rvol10 | gap    |      0.04   |          3 |          0.2914 | rvol10  |             0.0235 | True                 |             0.148  | False                |
| rvol3m | gap    |     -0.3837 |         -1 |         -0.4273 | gap     |             0.3427 | False                |             0.0029 | True                 |
| rvol10 | stress |      0.4434 |         -1 |          0.4595 | stress  |             0.8367 | False                |             0.0024 | True                 |
| rvol3m | stress |      0.0856 |         -5 |         -0.3194 | stress  |             0.0406 | False                |             0.37   | False                |
| rvol10 | rvol3m |      0.4197 |         -1 |          0.4448 | rvol3m  |             0.0086 | True                 |             0.0029 | True                 |

`best_lag` in quarters (analyzer's own units here since freq="Q"): positive means `a` leads `b`. `granger_a_to_b_sig` / `granger_b_to_a_sig` are BH-FDR-significant flags across the pooled grid.

## Reading

- **gap / stress**: corr 0.355, best lag -3q (stress leads gap). Granger gap->stress p=0.548 (not significant); stress->gap p=0.017 (significant).
- **rvol10 / gap**: corr 0.040, best lag 3q (rvol10 leads gap). Granger rvol10->gap p=0.023 (significant); gap->rvol10 p=0.148 (not significant).
- **rvol3m / gap**: corr -0.384, best lag -1q (gap leads rvol3m). Granger rvol3m->gap p=0.343 (not significant); gap->rvol3m p=0.003 (significant).
- **rvol10 / stress**: corr 0.443, best lag -1q (stress leads rvol10). Granger rvol10->stress p=0.837 (not significant); stress->rvol10 p=0.002 (significant).
- **rvol3m / stress**: corr 0.086, best lag -5q (stress leads rvol3m). Granger rvol3m->stress p=0.041 (not significant); stress->rvol3m p=0.370 (not significant).
- **rvol10 / rvol3m**: corr 0.420, best lag -1q (rvol3m leads rvol10). Granger rvol10->rvol3m p=0.009 (significant); rvol3m->rvol10 p=0.003 (significant).

## Honest caveat

143 quarters is a short sample for a quarterly Granger test (rule of thumb wants several multiples of the lag order per parameter); a null here is weak evidence of no relationship, not proof, and a significant result should be treated as a hypothesis to re-test once the market-implied (Q-measure) series is added, not a finished result on its own.
