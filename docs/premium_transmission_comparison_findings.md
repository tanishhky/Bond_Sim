# Premium transmission comparison: VAR-own-coefficients vs credit channel

Generated 2026-09-21 by `scripts/premium_transmission_comparison.py`. As-of 2026-09-15, K=2000 paths per variant, identical draws (common random numbers) within each premium slope, policy `status_quo`.

**What this answers.** Decision 0005's `rate_transmission` pushes the fiscal premium through the VAR's own coefficients estimated on 1985-2026 10y moves, a sample with essentially no US sovereign-risk variation (2026-09-17 steelman review, `docs/decisions/steelman-log.md`). P-20 adds a second, literature-calibrated option: Arellano, Bai & Bocola's (2017) sovereign-risk-to-output pass-through, central -0.006 pp growth per 100bp premium, with their own 10%-50% attribution sensitivity run as a low/high band (-0.002 / -0.010). `tests/test_credit_channel.py` proves the two channels differ on a synthetic fixture but deliberately does not assert a magnitude for the simulated gap, since that depends on premium-path and mean-reversion dynamics its single hand-built fixture cannot speak to. This script runs both channels on the real fitted VAR and real initial state.

## The instantaneous coefficients are very different in isolation

|       |   var (rate_transmission) |   credit_central |
|:------|--------------------------:|-----------------:|
| g_nom |                   -0.5645 |          -0.006  |
| pb    |                   -0.275  |           0      |
| u     |                    0.0192 |           0.0005 |

Per 100bp of premium *change* in one quarter, the real fitted VAR's own coefficient on g_nom (-0.564) is about 94x larger in magnitude than the credit channel's central elasticity (-0.006); `tests/test_credit_channel.py`'s synthetic fixture agrees in ordering (its hand-built coefficient is -0.386, same sign, same order of magnitude, also far larger than credit's -0.006), so `rate_transmission` reads as dramatically harsher than the credit channel *instantaneously*, on both the synthetic fixture and real data. What the synthetic fixture's end-to-end test does NOT settle is how much of that instantaneous gap survives into a 30-year simulated outcome, since that depends on how the coefficient interacts with the VAR's own mean reversion and with how large the quarterly premium *change* actually gets along a realistic debt path, not just its sign, which is what the summary below checks on the real fitted VAR.

## Simulated outcomes: default premium slope (P-01 = 2 bps/pt of debt/GDP above 100%)

Premium: linear: 2.0 bps per pt of debt/GDP above 100.0%

|                |   admissible_share |   P(trigger) |   median_years_to_trigger |   median_end_debt_gdp |   median_end_r10 |   median_end_u |   median_end_g_nom |   median_end_pb |
|:---------------|-------------------:|-------------:|--------------------------:|----------------------:|-----------------:|---------------:|-------------------:|----------------:|
| var            |                  1 |        0.965 |                    7.5833 |               147.61  |           5.0896 |         4.9864 |             5.431  |         -2.8325 |
| credit_low     |                  1 |        0.965 |                    7.5833 |               147.257 |           5.0753 |         4.9789 |             5.4217 |         -2.8026 |
| credit_central |                  1 |        0.965 |                    7.5833 |               147.252 |           5.0753 |         4.979  |             5.4216 |         -2.802  |
| credit_high    |                  1 |        0.965 |                    7.5833 |               147.247 |           5.0753 |         4.9791 |             5.4214 |         -2.8015 |

- Growth: var=5.4310pp, credit band=[5.4214, 5.4217]pp, var minus credit_central = +0.0094pp (var is milder on this metric)
- Trigger probability: var=0.9650, credit band=[0.9650, 0.9650], var minus credit_central = +0.0000 (var is milder on this metric)
- Terminal debt/GDP: var=147.6102pp of GDP, credit band=[147.2469, 147.2575]pp of GDP, var minus credit_central = +0.3580pp of GDP (var is harsher on this metric)

At this premium slope the four variants are close to indistinguishable despite the ~94x gap in their instantaneous coefficients: the mechanism is that `rate_transmission` acts on the *change* in premium each quarter, not the level, and under the default 2bps/pt slope that quarterly change stays small (a few basis points) for most of the horizon even as the premium's cumulative level rises; a large coefficient applied to a small, mean-reverting quarterly increment produces a small and mostly transient growth shock each period, most of which the VAR's own autoregressive dynamics unwind before it compounds into the terminal debt/GDP ratio. Under this configuration the debt path is dominated by the mechanical interest-cost/debt-service channel (a higher r10 directly raises debt service, independent of which growth-transmission channel is chosen), not by the growth feedback from either transmission channel.

## Simulated outcomes: steep premium slope (P-01 = 5 bps/pt, the literature's upper bound)

Premium: linear: 5.0 bps per pt of debt/GDP above 100.0%

|                |   admissible_share |   P(trigger) |   median_years_to_trigger |   median_end_debt_gdp |   median_end_r10 |   median_end_u |   median_end_g_nom |   median_end_pb |
|:---------------|-------------------:|-------------:|--------------------------:|----------------------:|-----------------:|---------------:|-------------------:|----------------:|
| var            |                  1 |        0.984 |                    7.5    |               171.223 |           7.7868 |         5.0043 |             5.4643 |         -2.9456 |
| credit_low     |                  1 |        0.983 |                    7.5417 |               169.255 |           7.6813 |         4.979  |             5.4216 |         -2.8011 |
| credit_central |                  1 |        0.983 |                    7.5417 |               169.228 |           7.6797 |         4.9793 |             5.4212 |         -2.7994 |
| credit_high    |                  1 |        0.983 |                    7.5417 |               169.202 |           7.6781 |         4.9796 |             5.4208 |         -2.7983 |

- Growth: var=5.4643pp, credit band=[5.4208, 5.4216]pp, var minus credit_central = +0.0431pp (var is milder on this metric)
- Trigger probability: var=0.9840, credit band=[0.9830, 0.9830], var minus credit_central = +0.0010 (var is harsher on this metric)
- Terminal debt/GDP: var=171.2233pp of GDP, credit band=[169.2021, 169.2545]pp of GDP, var minus credit_central = +1.9950pp of GDP (var is harsher on this metric)

**Reading this.** Neither channel is asserted as *the* correct one: `rate_transmission` is a stated monetary-tightening-shaped assumption (real coefficients, wrong historical regime), and the credit channel is a stated sovereign-risk-shaped assumption built from a different country's crisis (Italy 2011-2013, a currency-union member without its own central bank) and a different shock size than a US debt/GDP path this simulation reaches. The finding that matters for the paper is not which channel wins a horse race in one configuration, but how much the choice of channel matters *relative to* the premium slope itself (P-01): if the two tables above show the channel gap staying small even as the premium slope quintuples, that is evidence the mechanical debt-service channel, not the growth-feedback channel, is what drives this model's debt trajectories, a materially different headline claim than 'the loop hurts the real economy through growth', and the paper should say so explicitly rather than assume the growth channel is doing the work because it is the one with a theoretical story attached.

## Caveats

- The credit-channel elasticity is calibrated to one sovereign-debt crisis (Italy 2011-2013) applied to the US (currency issuer, deepest sovereign-debt market in the world); the transmission *mechanism* (bank funding costs, firm financing premia) is general but the *magnitude* may not transfer.
- The Okun's-law step inside `credit_channel_transmission` (coefficient 0.4, linearized around a 5% unemployment reference) is not itself sourced from the sovereign-risk papers; it is a standard macro rule of thumb converting an output shock to an unemployment response, flagged in the method's own docstring.
- Neither channel models the reverse causality (weak growth widening the primary-balance gap, which independently raises the premium next period); this script isolates the *given a premium path, how does it hit the real economy* question, not the full doom-loop feedback strength.
- The steep-premium run changes only P-01's slope, not the threshold/kink form (P-02) or the anchor; a kinked premium concentrated at high debt/GDP would produce larger quarterly premium changes exactly when the channel choice matters most, and is a natural next sensitivity, not run here.
