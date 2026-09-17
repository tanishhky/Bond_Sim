# 0001: One monthly calendar axis for everything

**Decision (2026-09-15).** Every array in Bond_Sim is indexed on a single
monthly grid (`bond_sim.calendar.MonthlyGrid`): history from 1980-01 through
the as_of month, then a 30-year forward window. Bonds, macro series, and
simulated paths all live on it.

**Why monthly.** It is the finest frequency at which the Monthly Statement of
the Public Debt, the Monthly Treasury Statement, and BLS employment report
natively, so no data is invented to reach it. Auctions are dated to the day
and snap to their issue month; coupons on notes and bonds land in the correct
month because payment months are anchored on the maturity month. Quarterly
GDP is forward-filled onto months (interpolation is available but off by
default: it manufactures an intra-quarter path that was never observed).

**Why 30 years forward.** It matches the CBO long-term outlook window and is
long enough for a slow-burning interest-cost spiral to either take hold or
visibly fail to. The horizon is a config value.

**Consequence.** The macro block of the simulator runs on a quarterly clock
(GDP and the primary balance are quarterly objects) and is expanded onto the
monthly grid piecewise-constant; the debt accounting runs monthly. See
decision 0005.

**Alternatives rejected.** Quarterly grid (loses auction and employment
timing); daily grid (no fiscal data exists at that frequency; 20x the memory
for no information).

**Survives, with one correction (steelman review, 2026-09-17, see
`steelman-log.md`).** "Interpolation manufactures a path that was never
observed" is a false dichotomy: forward-fill also manufactures an unobserved
path (a step at each quarter boundary), and decision 0006 records that step
producing spurious r > g months until growth was smoothed. The grid choice
is fine because every quantity the trigger compares is trailing-12-month
smoothed, which is what makes the fill method invisible to the test. Replace
the "no data is invented" rationale with that argument, and run the one-line
sensitivity (linear or Chow-Lin interpolation of g in place of forward-fill)
to show P(trigger) is unchanged.
