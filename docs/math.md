# Mathematical notes

Everything the notebooks compute, stated once. Notation: $d_t$ debt/GDP,
$r_t$ effective nominal interest rate on the stock, $g_t$ nominal GDP
growth, $pb_t$ primary balance / GDP (surplus positive). Quarterly objects
unless stated; annualized rates are converted with $(1+x)^{1/4}-1$.

## 1. Debt dynamics identity

With $D_t$ the debt stock, $Y_t$ nominal GDP, $I_t$ interest paid and
$PB_t$ the primary balance,

$$D_t = D_{t-1} + I_t - PB_t + SF_t, \qquad I_t = r_t D_{t-1},$$

where $SF_t$ is the stock-flow adjustment (everything that changes the stock
without passing through the deficit: TIPS principal accrual, buybacks,
cash-balance changes, valuation). Dividing by $Y_t = (1+g_t) Y_{t-1}$:

$$\Delta d_t = \underbrace{d_{t-1}\,\frac{r_t - g_t}{1+g_t}}_{\text{snowball}}
\;-\; \underbrace{pb_t}_{\text{primary}} \;+\; \underbrace{sf_t}_{\text{residual}}.$$

The first difference of the ratio is therefore not a free statistic: it is
the sum of three named terms, and the "residual" is measured, not assumed
(it is what the identity does not explain; for the reconstructed book it is
dominated by TIPS accrual and buybacks, decision 0004).

**Effective rate.** $r_t$ is bottom-up: trailing-12-month interest from the
bond book divided by the lagged stock. It moves as low-coupon debt rolls
into new issuance; that repricing, not the level of market yields, is the
interest-cost channel.

## 2. Stabilizing growth and stabilizing primary balance

Setting $\Delta d_t = 0$ and ignoring $sf$:

$$pb^*_t = d_{t-1}\,\frac{r_t - g_t}{1+g_t}, \qquad
g^*_t = \frac{d_{t-1}\, r_t - pb_t}{d_{t-1} + pb_t}.$$

$g^*$ answers "at what rate must nominal GDP grow to hold the ratio, given
today's rate, debt, and primary balance." At $d=1$, $r=4\%$, $pb=-2\%$:
$g^* = 0.06/0.98 = 6.1\%$.

## 3. Higher-order diagnostics

$\Delta^2 d_t = \Delta(\text{snowball}) + \Delta(\text{primary}) + \Delta(sf)$
decomposes acceleration into a widening of $r-g$, a deteriorating balance,
or the level feeding back. The level feedback is the loop: with a fiscal
premium $\pi(d)$ on new issuance, $r$ becomes increasing in $d$, so

$$\frac{\partial \Delta d}{\partial d} = \frac{r(d) - g}{1+g} + d\,\frac{r'(d)}{1+g} > 0$$

past some $d$ even when $r < g$ today. $\Delta^3 d$ is reported and tested
as a leading indicator of the trigger (does its sign change precede
crossings?), never used as the trigger: three differences of quarterly data
are mostly noise.

## 4. Feasible primary balance and the trigger (P-04 replaced)

Two benchmarks for what a government can actually run:

1. **Historical envelope.** The maximum (or a high quantile) of the observed
   US primary balance over the sample.
2. **Fiscal reaction function with fatigue** (Bohn 1998; Ghosh et al. 2013):
   $$pb_t = \alpha + \beta_1 d_{t-1} + \beta_2 d_{t-1}^2 + \beta_3 d_{t-1}^3 + \gamma' z_t + \varepsilon_t,$$
   with $z_t$ cyclical controls (output gap proxy, unemployment). A positive
   $\beta_1$ is the sustainability response; a negative cubic term is
   fatigue. The implied feasible balance at debt $d$ is $\widehat{pb}(d)$,
   and the **debt limit** $d_{\max}$ solves $\widehat{pb}(d) = pb^*(d)$
   with $r = r(d)$: beyond it no reaction the data has ever shown can
   stabilize the ratio.

**Trigger.** A path is in the doom loop from the first period $t$ such that
$pb^*_s > pb^{\text{feas}}_s$ **and** $r_s > \bar g_s$ for all $s \in [t, t+N)$,
where $\bar g$ is growth averaged over the same trailing window as the
effective rate (twelve months), so the two sides of $r - g$ are like-for-like
objects; with raw quarterly growth every one-quarter dip reads as $r > g$.
The second condition isolates the explosive case: with $r < g$ a rising
ratio is a primary-deficit problem that growth is already eroding
($\rho = (1+r)/(1+g) < 1$ in the difference equation $d_t = \rho\, d_{t-1}
- pb_t$); with $r > g$ the stock feeds on itself. No debt level is chosen by
hand; $N$, the envelope quantile, and the $r > g$ requirement are the free
choices and are swept.

**Why not the raw benchmarks.** On US history the single best quarter
(2000: primary surplus 4.4% of GDP at a 40% debt ratio) is too lenient as an
envelope, and the reaction function flags every quarter since 1982 because
the US has run primary deficits at every debt level, so any required
surplus exceeds the "expected" response. The 90th-percentile envelope with
the $r > g$ condition is the default; notebook 04 shows the sweep.

## 5. Block factors

Series are grouped into behavioral blocks $b$ (consumer, corporate,
financial conditions, fiscal, labor, rates, prices). Within a block, each
transformed, standardized series $x_{i,t}$ is decomposed as

$$x_{i,t} = \lambda_i F_{b,t} + u_{i,t},$$

with $F_b$ the first principal component (sign-aligned so the block's anchor
series loads positively) and $u_i$ the idiosyncratic residual. The factor
innovation is the systematic shock; $u_i$ is the imperfect-correlation
shock at the series level, with its covariance retained within the block.

## 6. Latent state chain

The factor vector $F_t \in \mathbb{R}^B$ follows

$$F_t = c_{S_t} + A\,F_{t-1} + e_t, \qquad e_t \sim (0, \Sigma_{S_t}),$$

with $S_t$ a Markov chain on $\{1,\dots,K\}$, transition matrix $P$. Common
$A$ (the data cannot identify state-specific dynamics on 140-170 quarters),
state-specific intercepts and covariances. Estimation is two-step and
transparent: a Gaussian HMM on $F_t$ for $S_t$ and $P$ (Dirichlet-shrunk
toward persistence, the RateWalk lesson on sparse regime data), then OLS for
$A$ and $c_s$ with smoothed state weights, $\Sigma_s$ from residuals by state.
Outcomes are mixtures over states: non-normal by construction, with the
tails the history has.

**Transition covariate (doom-loop channel).** The probability of entering
the stress state depends on financial conditions:
$P(S_{t+1} = \text{stress} \mid S_t, F^{\text{fin}}_t) $ is logistic in the
financial factor with a slope estimated on history. The fiscal premium
shifts the financial factor (mapping estimated from the term premium and
spreads), so a rising premium raises the stress-state hazard. This is the
state-space form of "fiscal stress tightens conditions and raises recession
risk."

**Shocks.** Two kinds, both dated: a forced transition ($S_t := s$ for a
duration) and a factor impulse ($F_{b,t} \mathrel{+}= \delta$ with decay).
Recession, rate spike, foreign-demand withdrawal, and consumer retrenchment
are instances.

**Residual draws.** $e_t$ and $u_{i,t}$ are bootstrapped from the historical
residuals of the current state, so no distributional family is imposed.

## 7. Admissibility, not clipping

Paths are not clipped at bounds. A path is rejected if it violates a hard
feasibility rule (negative nominal GDP; unemployment outside $[1, 35]$;
nominal yields outside $[0, 30]$; inflation outside $[-15, 50]$; debt below
zero). The rejected share and the reason are reported with every run;
results are computed on admissible paths only.

## 8. Common random numbers

Every policy and every shock scenario is run on the same draws (state
innovations, residual bootstrap indices), so a difference between two
outcome distributions is the intervention, not sampling noise.
