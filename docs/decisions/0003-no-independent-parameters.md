# 0003: No two inputs are independent unless the data says so

**Decision (2026-09-15, Tanishk's standing rule).** Every pair of model
inputs is analyzed for dependence before it is used, and the simulator's
shock structure is estimated, not assumed. Independence is the null that
has to be earned, not the default.

**What the rule produces, per pair of series (`analysis/correlation.py`).**

1. Full-sample correlation of the stationary transform (differences by
   default; levels of debt, rates, and employment are integrated and would
   correlate spuriously).
2. Rolling correlation (60 months). Its range (max minus min) is the
   regime-dependence diagnostic: a range above 0.5 means one number
   misdescribes the relationship.
3. Lead/lag scan: corr(x_t, y_{t+k}) for k in [-24, 24] months. The best lag
   and its sign say who leads.
4. Granger causality in both directions at the BIC-selected VAR order, with
   Benjamini-Hochberg control across the whole grid of tests so the
   "significant" flags survive multiple comparison.

**When the range is wide (`analysis/regimes.py`).** The rolling correlation
(or a chosen driver such as the debt/GDP change) is cut into regimes by
quantile (transparent) or a Gaussian HMM (captures persistence). Then two
questions:

* *Is the regime market-wide?* Adjusted Rand index between the regime label
  sequences derived from different pairs. High agreement means one common
  state variable is driving all of them and the simulator should carry a
  single regime process (P-12: regime-switching covariance).
* *Who switches first?* Cross-correlation of the "in high regime" indicators
  at lags. A consistent positive lead identifies the leading indicator, and
  the simulator's shock ordering must respect it.

**How the simulator honors it (`sim/macro.py`).** The macro block is a VAR
estimated on the quarterly panel; its residual covariance matrix is the
shock correlation, and its coefficient matrices are the cross-variable
responses. Sector employment shocks (`sim/labor.py`) keep their residual
covariance across sectors for the same reason. Policies intervene on the
same shock draws (common random numbers), so a policy comparison never
compares two different dice rolls.

**What this does not do yet.** The VAR covariance is constant. If the regime
analysis finds a market-wide regime, switching the covariance with it is the
next step (P-12), not a default.
