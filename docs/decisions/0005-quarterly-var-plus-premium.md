# 0005: A quarterly fiscal-macro VAR with an endogenous risk premium on top

**Decision (2026-09-15).** The simulator's macro block is a VAR(p) estimated
on the real quarterly panel of

    d_r10   change in 10y yield          d_r3m  change in 3m yield
    g_nom   nominal GDP growth (ann.)    pb     primary balance, % of GDP
    d_u     change in unemployment rate

with lag order by BIC (cap 4) and shocks drawn from the estimated residual
covariance (`sim/macro.py`). The fiscal risk premium (`sim/premium.py`) is
computed from last quarter's debt/GDP and added to the 10y yield outside
the VAR. Debt accounting is monthly and bottom-up from the bond book
(`sim/doomloop.py`).

**Why a VAR rather than hand-set coupling coefficients.** The thesis needs
"rates up, growth down, deficit up, unemployment up" to happen with the
magnitudes and lags the US economy has actually shown. A VAR estimates all
of those cross-effects and their shock correlations in one object, with no
number chosen by hand. It is also the standard tool in the fiscal-
sustainability literature, so a reader knows what to expect.

**Why the premium is outside the VAR.** A VAR is linear. The doom-loop
hypothesis is nonlinear: the market's reaction to debt is negligible over
most of history and becomes steep past some level. That nonlinearity is the
thesis's central open question (P-01/P-02), so it is isolated in one
pluggable object with three forms (none / linear / threshold) and the paper
reports results under each. Because the premium enters as a change in
d_r10 and the VAR has lagged d_r10 terms, the premium propagates into
growth, the primary balance, and unemployment through the estimated
coefficients. That propagation is the empirical content of "the loop hurts
the real economy".

**Why quarterly macro, monthly debt.** GDP and the primary balance are
quarterly objects; simulating them monthly would invent intra-quarter
dynamics. Coupons, maturities, and issuance are monthly facts from the
book. The two clocks meet on the monthly grid: macro states are held
constant within a quarter.

**New-issuance pool.** Debt issued after as_of is not tracked tranche by
tranche (there is no auction schedule for 2040). It lives in a pool with an
average coupon that updates as (a) maturing legacy principal and (b) 1/tau
of the pool per month are refinanced at the current issuance yield, and (c)
the deficit is financed at that yield. tau is the average maturity of new
issuance (P-11; estimable from the auction ledger).

**Levels, not differences (revised after the first two real runs).** The
first specification had rates and unemployment in differences. Two
artifacts followed. (1) The d_r10 intercept estimated on 1985-2026 is
negative, the sample's secular decline, so a 30-year simulation
extrapolated rates toward zero and debt/GDP fell (median 10y 2.2% in 2056).
Zeroing that drift fixed the mean but not the second problem: (2) a
difference-stationary state has no level anchor, so over 120 quarters the
10y and the unemployment rate random-walked into their floors and the
medians became artifacts of the clipping (median u 1.4-2.7%; median 10y
lifted to 6.2% by the reflecting barrier at zero). The VAR now has r10,
r3m, and u in levels; the estimated autoregressive coefficients supply
mean reversion to the sample means, which are printed at run time and can
be overridden as explicit anchors (P-03, P-13, P-14). The companion-matrix
eigenvalue is logged to confirm stability.

**Unemployment in logs (added after the admissibility runs).** On the
level of unemployment, the linear VAR produced sub-1% unemployment on
20-25% of 30-year paths (the US minimum on record is 2.5%), which the
admissibility filter then rejected, biasing the surviving sample toward
weaker economies. The VAR now carries log(u) internally and reports levels;
the floor is natural and no path is lost to it. The state block does the
same through a log transform on the labor block's unemployment series.

**Premium transmission.** With base rates in levels, adding the premium into
the r10 equation would let the VAR mean-revert it away, silently assuming
markets forgive debt. Instead the market 10y is base + premium, and the
premium *change* is pushed into the growth, primary-balance, and
unemployment equations through the VAR's own estimated coefficients on the
lagged 10y (`MacroVAR.rate_transmission`). The issuance yield is a
bill/coupon blend at the stock's bill share.

**Premium transmission, two calibrated options (added 2026-09-21, P-20).**
`VARBlock` now takes `transmission="var"` (the paragraph above, kept as
default) or `transmission="credit"`
(`MacroVAR.credit_channel_transmission`), which instead transmits the
premium through Arellano-Bai-Bocola (2017)/Bocola (2016)'s estimated
sovereign-spread-to-output pass-through: central -0.006 pp of annualized
growth per 100bp of premium, with their own attribution sensitivity (10%-
50% of Italy's 2012 output decline) implying a band of roughly -0.002 to
-0.010, converted to unemployment via a standard Okun's-law coefficient
(0.4, not itself sourced from the sovereign-risk papers, flagged
separately in the docstring). This resolves the steelman finding below by
making the assumption explicit and swappable rather than fixing it: the
paper reports both channels, run at identical draws so only the
transmission differs, and does not assert which is harsher a priori. On
the synthetic fixture used in `tests/test_credit_channel.py` the "var"
channel is in fact the harsher one, because that fixture's hand-built
r10->g_nom coefficient (-0.4) is far larger than the credit channel's
-0.006; this is a property of that fixture's calibration, not evidence
about the real fitted VAR, and the real ordering is an empirical question
for the notebook comparison, not something decided here.

**Seasonality.** The MTS deficit is not seasonally adjusted (April is a
surplus month every year). The primary balance in the state is therefore a
trailing-four-quarter ratio, and the initial growth rate is a
trailing-four-quarter average.

**Known limitations.** Constant residual covariance (P-12); no term-premium
channel separate from the fiscal premium; receipts/GDP for the
interest/receipts trigger is the last 12 months of MTS receipts over SAAR
GDP, held fixed over the horizon (P-04). Each is a documented extension
rather than a hidden assumption.

**⚠️ Steelman review, 2026-09-17, see `steelman-log.md` — partially resolved
2026-09-21.** `rate_transmission` pushes the fiscal premium through
coefficients estimated on 1985-2026 10y moves, which were monetary-policy
and term-premium driven, not sovereign-risk driven; the regime being
simulated is absent from the estimation sample (Lucas critique). The
sovereign-risk literature transmits through bank balance sheets and credit
spreads with non-proportional magnitudes (Bocola 2016: +60bp firm financing
premia, output -1.4% at peak; Corsetti & Dedola 2016). "The empirical
content of 'the loop hurts the real economy'" is the empirical content of
"a Fed hike hurts the real economy", relabeled. **Built:** the
Bocola-calibrated credit-channel option above (P-20), reported side by
side rather than replacing `rate_transmission`; and the side-by-side
comparison itself, `scripts/premium_transmission_comparison.py`
(`docs/premium_transmission_comparison_findings.md`), run on the real
fitted VAR (not the synthetic test fixture) at two premium slopes.
**Finding:** the two channels' *instantaneous* coefficients differ by
~94x on the real data (var's g_nom coefficient -0.56 per 100bp of premium
*change* vs credit-central's -0.006); the synthetic test fixture agrees in
this ordering (its hand-built coefficient is -0.386, also far larger than
credit's -0.006), so `rate_transmission` reads as dramatically harsher
*instantaneously* on both. But the *simulated* 30-year outcomes are close
at the default P-01 slope (terminal debt/GDP differs by only 0.36pp of
GDP) because
`rate_transmission` acts on the small quarterly premium *change*, not its
level, and the VAR's own mean reversion unwinds most of each shock before
it compounds. At 2.5x the default P-01 slope the gap widens to ~2.0pp of
GDP but stays modest relative to the ~170pp GDP level itself. Provisional
reading: this model's debt trajectories are driven more by the mechanical
interest-cost/debt-service channel than by either growth-transmission
channel, which is itself a claim the paper should state explicitly rather
than assume. **Still open:** the VAR-vs-state-block transmission-channel
gap noted here (state block uses the Baa loading, P-16) is still
undocumented as its own comparison; the kinked/threshold premium form
(P-02), which concentrates larger quarterly premium changes at high
debt/GDP, has not been run through this comparison and is where channel
choice would plausibly matter most. Which channel is the paper's headline
result, or whether both are reported as a band, is an editorial decision
not yet made.
