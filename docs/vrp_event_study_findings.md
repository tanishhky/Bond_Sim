# Rates variance risk premium: 2003-2020 event study (Phase 2, 2026-09-22)

`scripts/vrp_event_study.py`. VXTYN (TYVIX, final-vintage, discontinued 2020-05-15) minus 20-day realized vol of DGS10, monthly, 2003-01-01 to 2020-05-01 (209 months, 70 quarters after joining to the trigger history and the HMM states). Explicitly bounded and backward-looking; not fed into the live simulation.

## Descriptive: is the rates VRP, like the equity one, mostly a tail-risk premium?

Mean VRP over the full window: **5.363** (implied above realized on average); positive in **100.0%** of months.

| Episode | Window | Mean VRP in window | Full-sample mean |
|---|---|---|---|
| GFC (2007-2009) | 2007-08-01 to 2009-06-01 | 8.138 | 5.363 |
| 2010 flash crash / European debt crisis onset | 2010-04-01 to 2010-06-01 | 6.434 | 5.363 |
| 2011 US debt-ceiling standoff + S&P downgrade | 2011-07-01 to 2011-09-01 | 6.726 | 5.363 |
| 2013 taper tantrum | 2013-05-01 to 2013-09-01 | 5.498 | 5.363 |
| COVID onset | 2020-02-01 to 2020-04-01 | 5.937 | 5.363 |

## Lead-lag against the model's own diagnostics

| a   | b      |   full_corr |   best_lag |   best_lag_corr | leads   |   granger_a_to_b_p | granger_a_to_b_sig   |   granger_b_to_a_p | granger_b_to_a_sig   |
|:----|:-------|------------:|-----------:|----------------:|:--------|-------------------:|:---------------------|-------------------:|:---------------------|
| vrp | stress |      0.6655 |         -1 |          0.6711 | stress  |             0.8462 | False                |             0.0633 | False                |
| gap | stress |      0.6556 |         -4 |          0.755  | stress  |             0.7278 | False                |             0.0008 | True                 |
| vrp | gap    |      0.3229 |          3 |          0.5851 | vrp     |             0.019  | False                |             0.3165 | False                |

## Reading

- **vrp / stress**: corr 0.665, best lag -1q (stress leads vrp). Granger vrp->stress p=0.846 (not significant); stress->vrp p=0.063 (not significant).
- **gap / stress**: corr 0.656, best lag -4q (stress leads gap). Granger gap->stress p=0.728 (not significant); stress->gap p=0.001 (significant).
- **vrp / gap**: corr 0.323, best lag 3q (vrp leads gap). Granger vrp->gap p=0.019 (not significant); gap->vrp p=0.317 (not significant).

## Honest caveats

- 70 quarters is a genuinely short sample, well under Phase 1's 143; treat any Granger result here as a hypothesis, not a finished result, even more so than Phase 1.
- Final-vintage data throughout: VXTYN's ALFRED vintage metadata doesn't reflect true historical publication timing, so this whole script uses the latest values, not an as-of reconstruction. That's fine for a backward-looking event study, it would not be fine for anything claiming to test what was knowable in real time.
- Bounded to 2003-2020 by data availability, not by choice; says nothing about the current period. A current MOVE series (Bloomberg, WRDS/ICE) would be needed to extend this.
