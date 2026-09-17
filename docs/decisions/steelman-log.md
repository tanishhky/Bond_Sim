# Steelman log

Running record of `/steelman` reviews run against Bond_Sim design choices and
thesis claims (dual-lens quant + economic review, adversarial pass, grounded
in the Quant Wiki corpus at `GitHub/quant-wiki`). Reverse-chronological. Each
entry that critiques a specific decision doc gets a one-line pointer added to
that doc too, so the flag is visible without knowing to check this file.

---

## 2026-09-17 (Fable audit pass): decisions 0001, 0002, 0003, 0004, 0005, 0007

Six entries, one pass. Verdicts: 0001 Survives; 0002, 0003, 0005, 0007 Needs revision; 0004 Needs revision scoped to the monetization counterfactual. Wiki coverage was thin for 0001 and 0002 (the corpus has no temporal-disaggregation or real-time-data econometrics), so those objections are reasoned independently and say so. Code facts cited below were checked in the repo, not assumed.

### 0001-monthly-grid: SURVIVES (rationale needs a correction)

**Claim:** Indexing everything on a monthly grid, with quarterly GDP and primary balance forward-filled piecewise-constant onto months, does not invent information, whereas interpolation would manufacture an unobserved intra-quarter path; monthly is the right resolution because it is the finest native frequency of the fiscal data.

**Quant lens:** No direct wiki hit. Independently: forward-fill is also an unobserved path (a step function with a jump at each quarter boundary), and the temporal-disaggregation literature (Chow-Lin, Denton, Litterman) exists because step filling introduces autocorrelated measurement error. In the trigger, `r_eff` is genuinely monthly (bottom-up from the book) while `g` is stale for up to two months inside each quarter. The neutralizer is decision 0006's later fix: `g` enters the trigger as a trailing-12-month mean, which damps the step to a fraction of its size (at quarter ends it is exactly the four-quarter mean). The nearest wiki grounding, Barrett (2018), models r-g at low frequency with spectral methods; the sustainability literature operates coarser than monthly, so the grid is finer than anything it is compared against.

**Economic lens:** Not an economics question. Gerling et al. (2017) date fiscal crises at annual resolution, so a 12-month persistence window is already at the coarse end of what the literature calls an episode.

**Tension:** none identified.

**Strongest objection:** "Interpolation manufactures a path that was never observed" is a false dichotomy; forward-fill manufactures one too. If any quantity in the r>g test used raw monthly `g`, the step would create spurious r>g months at quarter boundaries. Decision 0006 records exactly that happening ("with raw quarterly growth the probability swung from 0.99 to 0.01 between 6- and 24-month persistence because every one-quarter dip read as r > g"). The objection is not hypothetical; it bit once and was fixed by smoothing, not by the grid choice.

**Verdict: Survives.** Every quantity the trigger compares is trailing-12-month smoothed, which makes the forward-fill choice invisible to the test. Two edits for the writeup: (1) drop the "no data is invented" rationale and replace it with the smoothing argument; (2) run a one-line sensitivity (linear or Chow-Lin interpolation of `g` in place of forward-fill) and report that P(trigger) is unchanged, which is the actual proof.

**Citations:** Barrett (2018); Gerling, Medas, Poghosyan, Farah-Yacoub & Xu (2017); decision 0006's own record.

### 0002-full-alfred-vintages: NEEDS REVISION

**Claim:** Pulling full ALFRED real-time vintages (with a conservative backfill for pre-vintage rows) makes every as-of analysis, including lead/lag and regime claims and the historical trigger evaluation, answer "did anyone actually know this at the time?" with a yes.

**Quant lens:** No real-time-data econometrics in the wiki (no Croushore-Stark). Independently: vintage discipline is necessary for lead/lag claims (the doc's own 2009 GDP example, revised five times) and the no-lookahead invariance test is the right architectural guard. But the guard covers the data layer only. `sim/sustainability.py::evaluate_history(q, feasible, ...)` takes a plain quarterly DataFrame; there is no `as_of` argument and no vintage handling anywhere in that module, and `FeasiblePB.evaluate` runs on the same frame. Whatever panel is passed in (in practice the latest vintage) is what the historical backtest sees.

**Economic lens:** Borio, Disyatat & Juselius (2016) is the closest hit and cuts against the decision's completeness: real-time estimates of the cyclically-adjusted balance were systematically optimistic before 2008 because the financial-cycle component of output was read as trend. That is a feasible-primary-balance-in-real-time problem: the envelope quantile taken from today's vintage of the 2000 surplus (4.4%) is not what a 2005 backtester saw, and an expanding-window envelope would exclude later observations by construction. Mauro et al. (2013) run Bohn-style tests on a final-vintage 200-year panel and do not claim real-time validity, which is the honest alternative.

**Tension:** the data infrastructure is vintage-exact; the trigger's historical evaluation and its feasibility benchmark are not. "The reviewer must get a yes" holds for one layer and is implicitly claimed for both.

**Strongest objection:** The thesis's flagship historical fact, "the default definition flags only 2009Q2-Q3 since 1980", is a final-vintage result. On real-time data, 2009Q2's GDP, primary balance, and effective rate were different, and the envelope would be a quantile of pb history visible at the time. The real-time trigger may flag different quarters, or flag 2009 later than Q2. Until `evaluate_history` runs on as-of vintages with an as-of envelope, the "which quarters would have flagged" table is a hindsight table, and a referee who knows the Borio point will ask.

**Verdict: Needs revision.** Concrete fix: give `evaluate_history` an as-of path that (a) builds the quarterly frame from `VintageFrame.as_of(t)` for each t and (b) fits `FeasiblePB` on the pb history visible at t (expanding window). Report the real-time flagged quarters next to the final-vintage ones; the difference is itself a thesis result. Also state that 30-year forward simulations are vintage-independent (today's vintage is the right initial condition), so the strict pull buys credibility for the backtest and the lead/lag claims, which is where it must actually be used.

**Citations:** Borio, Disyatat & Juselius (2016); Mauro, Romeu, Binder & Zaman (2013); Reinhart & Rogoff (2010) on subperiod re-estimation under structural change.

### 0003-no-independent-parameters: NEEDS REVISION

**Claim:** Estimating the shock covariance from VAR residuals, rather than assuming independence, captures the dependence between inputs; a constant covariance is acceptable until the regime analysis finds a market-wide regime (a shared state variable across pairs), at which point P-12 switches it.

**Quant lens:** The model's own diagnostic already answered the first question: rolling-correlation ranges of 1.3-1.7 on the thesis pairs (the 2026-09-15 analysis run recorded in methodology.md), i.e. correlations that change sign across 60-month windows. A single full-sample covariance is the calm-weighted average of two different dependence structures. The rule "switch covariance only if regimes synchronize across pairs" conflates two claims: (i) there is one common latent state (ARI high) and (ii) dependence is time-varying at all. The measured ARI of 0.11 rejects (i) and says nothing against (ii). Vallarino (2026) estimates sovereign-stress dependence on 24-month rolling co-movement networks precisely because the structure is time-varying without a single common driver, and finds stress synchronization is non-random (placebo p<0.001).

**Economic lens:** Cole, Neuhann & Ordonez (2016) give the mechanism: belief linkages make co-movement jump during stress independently of real linkages, so stress covariance is structurally different from calm covariance, not a scaled version of it. Vallarino's "large regional episodes are not a mechanical consequence of volatile countries" is the empirical version.

**Tension:** the state block (0007) fixes this for itself (state-specific covariances, residuals bootstrapped by state); the VAR block does not. The decision treats the VAR's constant covariance as a pending extension; the objection is that the pending extension is the quantity of interest.

**Strongest objection:** The trigger is a tail event driven by the joint move "rates up, growth down, deficit up". With calm-weighted covariance the VAR block under-generates exactly that joint move, so its P(trigger) is biased low relative to the world its own residuals show in stress windows. The two blocks then differ (0.60 vs 0.58 at present) for a reason the paper attributes to "distributional assumption" when part of it is the covariance regime.

**Verdict: Needs revision.** Minimal fix without P-12's full machinery: estimate the VAR residual covariance on NBER-recession quarters (or on the state block's 44 stress-state quarters) and re-run the VAR block with it as a sensitivity; report P(trigger) under calm, full-sample, and stress covariances. Rewrite the last paragraph of 0003: the constant covariance is a known downward bias on tail co-movement, not a neutral default awaiting evidence.

**Citations:** Vallarino (2026); Cole, Neuhann & Ordonez (2016); the 2026-09-15 analysis run (rolling ranges 1.3-1.7, mean ARI 0.11).

### 0004-instrument-simplifications: NEEDS REVISION (scoped to the monetization counterfactual)

**Claim:** Ignoring TIPS inflation accrual, buybacks, odd first coupons, and reopening accrued interest is second-order for a monthly interest total and for the trigger; each omission is sized by the validation report.

**Quant lens:** For the status-quo, austerity, and growth-led scenarios the claim holds: TIPS reconcile at 0.83 of MSPD (the missing 17% of the TIPS class is the accrual), buybacks are ~3% of one bond's face, odd coupons net to zero. But the sizing quoted in the decision (accrual = 1.2% of total marketable debt) is the wrong denominator for the scenario where it matters: under an inflation-driven counterfactual the relevant quantity is the TIPS face share of marketable debt (several times 1.2%; the ledger can compute it exactly), because that share is not inflated away, its principal grows one-for-one with CPI.

**Economic lens:** BIS Papers 65 (Threat of Fiscal Dominance, citing Barro 2003 and Siu 2004): indexed debt immunizes the budget from inflation shocks; nominal debt is what provides the state-contingent "insurance" monetization exploits. Furman & Summers (2020) carry the amount of debt inflated away each year as a first-order term in the r-g accounting. Elmendorf & Mankiw (1998) on the price-level channel to real debt. All three: the nominal/indexed split determines how much a given inflation path reduces the real burden.

**Tension:** none between lenses; both say the omission is scenario-dependent.

**Strongest objection:** Once the monetization counterfactual operates through an inflation path (which P-15 is meant to enable; check how `sim/policy.py` implements it today), a book that carries TIPS at nominal face with a fixed coupon inflates away TIPS principal as if it were nominal. That biases monetization toward "works better than it does" by the TIPS face share, on top of the already-flagged P-15 gap. Two documented limitations compound in the one counterfactual where each is first-order, and the current note "monetization results are not credible until P-15" attributes the non-credibility to P-15 alone.

**Verdict: Needs revision, narrowly.** Scope the "second-order" claim: true for scenarios without an inflation channel, false for monetization. When P-15 lands, accrue TIPS principal on the simulated CPI path (the doc already names this extension); until then, carve the TIPS face share out of the inflation-erosion mechanics or report the monetization P(trigger) with that share as a stated bound.

**Citations:** BIS Papers No 65 (2012); Furman & Summers (2020); Elmendorf & Mankiw (1998); methodology.md validation table (TIPS 0.830).

### 0005-quarterly-var-plus-premium: NEEDS REVISION

**Claim:** A levels VAR estimated on 1985-2026 with a fiscal risk premium added outside it captures the doom-loop transmission; pushing the premium change into growth, primary balance, and unemployment through the VAR's estimated coefficients on the lagged 10y (`MacroVAR.rate_transmission`) is "the empirical content of 'the loop hurts the real economy'".

**Quant lens:** `rate_transmission` (macro.py:201) sums the estimated coefficient matrices over lags, takes the r10 column, and zeroes the rate rows: the real-economy response to a sovereign-risk premium is, by construction, the average historical response to a 10y move. On 1985-2026, 10y variation was driven by monetary policy, term premia and flight-to-quality, not by US sovereign risk, which barely varied. That is the Lucas critique in its cleanest form: the regime being simulated is absent from the estimation sample. Eichengreen, Menuet & Donnat (2026), in the corpus, run a Primiceri time-varying-parameter VAR specifically to test whether fiscal responses are regime-dependent; the tool to check the constant-coefficient assumption is already in the library. Riblier (2023) identifies fiscal shocks by narrative and timing restrictions rather than reduced-form coefficients, for the same reason.

**Economic lens:** Bocola (2016) models the sovereign-risk channel explicitly: news of future default tightens bank funding constraints and raises firms' financing premia (+60bp at 2011Q4 for Italy, risk channel ~45% of the effect), output down 1.4% annualized at peak. Perez (2015, via Mitchener & Trebesch 2021) adds a liquidity channel. Corsetti & Dedola (2016) make the transmission depend on the central bank's balance-sheet stance. None of these transmit sovereign risk through "the historical response to a 10y move"; they transmit through bank balance sheets and credit spreads, with magnitudes not proportional to the Treasury yield change.

**Tension:** the state block (0007) transmits the premium through the Baa-spread loading on the financial factor (P-16), which is closer to Bocola's channel. The two blocks therefore embody two transmission theories, and the paper currently reads their difference as "distributional assumption" (0007, "Why two blocks") when it is also a transmission-channel difference.

**Strongest objection:** "That propagation is the empirical content of 'the loop hurts the real economy'" is the empirical content of "a Fed hike hurts the real economy", relabeled. The mean-reversion anchors compound it: r10 and u revert to 1985-2026 means, a sample containing no doom loop, so the VAR's estimated dynamics pull the economy back toward the no-loop equilibrium while the imposed premium pushes it away; P(trigger) is then partly a race between an estimated force and a stated one.

**Verdict: Needs revision.** (1) Reword the decision and the paper: the VAR block transmits the premium as if it were a monetary tightening, a stated assumption, not empirical content of the loop. (2) Add a second transmission option (credit-channel mapping calibrated to Bocola's pass-through, output loss per 100bp of sovereign spread) reported side by side with `rate_transmission`. (3) State that the VAR-vs-state comparison differs in transmission channel as well as distribution, so the 0.60 vs 0.58 gap is not over-read as a distributional result.

**Citations:** Bocola (2016); Corsetti & Dedola (2016); Mitchener & Trebesch (2021) on Perez (2015); Eichengreen, Menuet & Donnat (2026); Riblier (2023).

### 0007-latent-states-and-admissibility: NEEDS REVISION

**Claim:** A 2-3 state Gaussian HMM on block factors with a common VAR(1), state-specific intercepts, and residuals bootstrapped by state gives an outcome distribution with the tails history has; the hazard of entering the stress state is logistic in the financial-conditions factor with a slope estimated on history, which is the state-space form of the doom loop; rejecting infeasible paths rather than clipping keeps the distribution honest.

**Quant lens:** PARAMETERS.md P-19 records that on 1990-2026 the unrestricted hazard slope is negative (the three stress entries, 2001, 2007Q4, 2020, were preceded by loose conditions), a ≥0 sign restriction is imposed, the channel is off by default, and the table says "three events cannot identify a hazard ... a value here is a stated view, not an estimate". Decision 0007 item 3 says "slope estimated on history". The two documents disagree; the parameter table is the correct one. JFR-rg Part II (Wakimoto 2026) addresses exactly this: with few regime transitions the honest output is a set-valued regime classification, "conservative by design ... rather than forcing point precision when the data cannot support it". Separately, 44 stress-state quarters supply the residual bank for 120 simulated quarters, so the stress tail is the empirical tail of 44 draws.

**Economic lens:** Conesa & Kehoe (2015) fix the panic probability at 4% and call it arbitrary; Aguiar et al. (2016) treat the sunspot probability as a calibration target. The sovereign-debt literature does not estimate crisis-entry hazards from three domestic episodes; it states a value and sweeps it, which is what notebook 05's grid does. Gourinchas & Obstfeld (2011) drop post-crisis observations to avoid post-crisis bias in crisis-prediction regressions, relevant to a 2-state fit whose stress state is substantially aftermath (2009-2012, 2021-2023) rather than entry.

**Tension:** the state block was built to carry the doom-loop channel the VAR cannot; the data turned that channel off. The paper must not present the state block's P(trigger) as containing an estimated loop when the loop parameter is a prior set to zero.

**Strongest objection (admissibility):** rejection is a selection rule, so the reported number is P(trigger | admissible). Under placeholders admissibility is 99.5-100% and the truncation is negligible for the status quo, but the rate-spike shock triggers on all paths (2026-09-15 run) and the paths most likely to breach yields>30 or inflation>50 are doom-loop paths. Methodology says every rejection is counted; it does not say P(trigger and rejected) is reported per scenario, which is the number that bounds the truncation. Clipping puts mass at the bound; rejection puts a hole in the tail. Both distort; the decision presents only one as a distortion.

**Verdict: Needs revision.** (1) Change 0007 item 3 to match P-19: the hazard slope is a stated view with a ≥0 restriction, off by default, swept in notebook 05; history cannot identify it. (2) Report the stress state as a set (2- and 3-state fits side by side, already P-17) and state that the stress residual bank is 44 quarters. (3) For every scenario, report "triggered and rejected" alongside admissibility so P(trigger) is bracketed between P(trigger | admissible) and that plus P(rejected). (4) Check the Gourinchas-Obstfeld convention: if the stress state is meant to capture entry, test whether excluding aftermath quarters changes the hazard sign.

**Citations:** PARAMETERS.md P-19; Wakimoto (2026, JFR-rg Part II); Conesa & Kehoe (2015); Aguiar, Chatterjee, Cole & Stangebye (2016); Gourinchas & Obstfeld (2011).

**Follow-ups done (2026-09-17):** (1) decision 0007 item 3 reworded to match P-19; (3) `SimResult.summary()` now reports `P(trigger_rejected)`, `P(trigger)_lo`, `P(trigger)_hi` and `summary_table` shows them, so every scenario's P(trigger) is bracketed rather than conditional on survival. (2) and (4) remain open.

---

## 2026-09-17: Trigger persistence (P-04 / 0006-identity-trigger), NEEDS REVISION

**Claim tested:** requiring the accounting-identity trigger (pb* infeasible
AND r>g) to persist for N consecutive months before firing the crisis
mechanism, rather than firing on a single-month breach, correctly filters out
transient, noise-driven breaches that don't reflect genuine fiscal
unsustainability.

**Quant lens.** Supported, but narrowly. Stochastic control theory (Ferrari &
Rodosthenous 2018, regime-switching debt-to-GDP control as a Dynkin game)
formally justifies a "wait region" before acting under switching costs,
structurally analogous to a persistence requirement. This is an optimal
*policy-intervention* result, not a *crisis-detection* result, the analogy
supports "when should the government act," not automatically "when does a
crisis occur."

**Economic lens.** The quantitative rollover-crisis literature (Aguiar,
Chatterjee, Cole & Stangebye 2016 Handbook survey) states directly that the
self-fulfilling mechanism, while requiring fundamentals bad enough for
default to be possible, "allows relatively wide latitude in the timing of a
sovereign debt crisis" — actual timing is investor-belief-driven, not tied to
how long the fundamental trigger has persisted. Cole, Neuhann & Ordonez
(2016) add a contagion channel: a belief shift can arrive from inference
about a *different* country's default, with no connection to your own
domestic trigger's duration at all.

**Tension.** The quant lens and economic lens are answering different
questions with different clocks. Quant: waiting is optimal when you control
the timing (a policy choice). Economic: the actual crisis-firing event is
controlled by investor beliefs, not gated by your persistence window.

**Strongest objection.** The N-month filter conflates two distinct
questions: (1) when does the bad equilibrium become *reachable*
(fundamentals-gated — filtering noise here is legitimate), vs (2) when does
the crisis actually *fire* (belief/coordination-gated, with "wide latitude"
relative to fundamentals per the literature). 0006's own framing
("how likely is the doom loop") reads as answering question 2 using a
mechanism that's actually built to answer question 1.

**Verdict: Needs revision**, not wrong, incomplete. `trigger_persistence_months`
is well-justified as defining when the bad equilibrium becomes reachable, and
should be described that way. Whether the thesis/model also uses it as a
proxy for *actual* crisis timing is the open question, worth checking
directly: do either of the two macro blocks (levels VAR / block-factor latent
chain) carry any belief- or coordination-shock layer on top of the identity
trigger, or does firing happen purely once persistence is satisfied? If the
latter, that's a real, citable gap between what the model does and what the
self-fulfilling-crisis literature says actually drives timing, worth
addressing explicitly in the thesis (as a stated scope limitation, or as an
extension) rather than left implicit.

**Citations:** Aguiar, Chatterjee, Cole & Stangebye (2016); Cole, Neuhann &
Ordonez (2016); Ferrari & Rodosthenous (2018).
