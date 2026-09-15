# 0007: Behavioral states from block factors, shocks by bootstrap, paths filtered not clipped

**Decision (2026-09-15, Tanishk's design).** The Gaussian VAR is no longer
the only macro block. A second block, run on identical debt accounting and
identical random numbers, builds the outcome distribution from estimated
behavioral states:

1. **Block factors.** Series are grouped into consumer, corporate/activity,
   financial conditions, fiscal, labor, rates, and prices. Each block is
   reduced to its first principal component, sign-aligned so high means
   expansionary / loose / strong / healthy. The residual of each series
   against its block factor is kept: it is the imperfect-correlation shock
   at the series level, exactly the "deviation in the correlation" that was
   asked for as the uncertainty source.
2. **Latent chain.** A Gaussian HMM on the factor vector (2-4 states by BIC,
   sticky Dirichlet prior on transitions) gives the states and the
   transition matrix; a common VAR(1) with state-specific intercepts gives
   the within-state dynamics; residuals are banked by state and drawn by
   bootstrap. The outcome distribution is a mixture over states with the
   tails history has: non-normal by construction, and no distributional
   family is assumed.
3. **Doom-loop channel.** The hazard of entering the stress state is
   logistic in the financial-conditions factor (slope estimated on
   history). The fiscal premium shifts that factor through the Baa-spread
   loading (P-16). Fiscal stress tightens conditions and raises recession
   risk; that is the state-space form of the loop.
4. **Shocks.** Forced transitions (recession now) and factor impulses (rate
   spike, foreign-demand withdrawal, consumer retrenchment) are dated
   interventions on the same draws. All four are run; none is privileged.
5. **Admissibility.** Macro variables are not clipped, with one structural
   exception: nominal yields are floored at zero inside the macro blocks
   (the effective lower bound: cash exists). Without it the first runs
   rejected 60-90% of paths for a 3m bill dipping below zero at some point
   in 30 years, which is the bound doing its job, not a senseless path.
   After a run, paths that violate hard feasibility (negative GDP,
   unemployment outside 1-35, yields above 30, inflation outside -15 to
   50, negative debt, non-finite values) are rejected and the count per
   rule is reported. Clipping had been creating mass at the bounds;
   rejection keeps the distribution honest and makes the model's failure
   modes visible. The VAR block's rejection rate for sub-1% unemployment is
   itself reported as a property of a linear Gaussian VAR.
6. **Idiosyncratic persistence.** A series' residual against its block
   factor is not white noise (unemployment does not jump a point a quarter
   around the labor factor). Each idiosyncratic residual follows its own
   estimated AR(1) with bootstrapped innovations, so reconstructed targets
   keep the persistence and the innovation size the data show. Drawing
   residuals independently each quarter had produced thinner tails than
   the VAR, the opposite of the model's purpose.

**Why two blocks.** The VAR is the transparent benchmark every reader
knows; the state model is the thesis's own construction. Running both on
the same draws lets the paper say how much of any result is the
distributional assumption. The bond book, premium, policies, and trigger
are shared.

**Data constraint, stated.** The common sample of all seven blocks starts
1990Q1 (VIX and the SLOOS tightening series begin then): ~146 quarters. That
supports 2-3 latent states with a persistence prior, not a per-block joint
chain. Results are reported at 2 and 3 states (P-17).
