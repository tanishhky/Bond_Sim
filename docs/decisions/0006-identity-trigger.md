# 0006: The doom loop is defined by the accounting identity, not a debt level

**Decision (2026-09-15, Tanishk's design).** The fixed debt/GDP and
interest/receipts thresholds (old P-04) are demoted to a comparison column.
A path is in the doom loop from the first period at which the
debt-stabilizing primary balance exceeds the feasible primary balance for N
consecutive periods:

    pb*_t = d_{t-1} (r_t - g_t) / (1 + g_t)     >  pb_feasible    for N periods

with `r` the bottom-up effective rate from the bond book (trailing interest
over the stock), `g` nominal growth, and `d` the ratio. The stabilizing
growth rate `g* = (d r - pb) / (d + pb)` is reported alongside as the
"what would it take" number. Full derivation in `docs/math.md` 1-4.

**Why.** "How likely is the doom loop" is only as good as the definition of
the loop. Any chosen level (150%? 200%?) is an opinion. The identity says
exactly when a ratio is on an explosive path (r > g and the primary balance
cannot close the gap) and when it is not, and it says it for each path
separately using that path's own rates and growth. Japan at 250% with r < g
is not in a loop; a country at 90% with r - g = 4% and a structural deficit
is.

**Feasible primary balance.** Two benchmarks, both from data:
1. the historical envelope: the maximum (or a high quantile) of the observed
   US primary balance;
2. a fiscal reaction function with fatigue (cubic in lagged debt, cyclical
   control), whose implied response at the simulated debt level is what
   history says a government does. It also yields a debt limit, the level
   beyond which no historically observed response stabilizes the ratio,
   once the premium makes `r` a function of `d`.

**⚠️ Benchmark choice, open (real-time backtest, 2026-09-17, see
`steelman-log.md` 0002 entry and `docs/realtime_trigger_backtest*.md`).**
Evaluated in real time on vintage data, the envelope flags 25% of quarters
while today's vintage flags 0% (its level is set by the 2000 surplus), the
reaction function is vintage-robust but fires through the 1996-2000
surpluses, and a third benchmark added the same day, the actual trailing
primary balance (`which="actual"` in `evaluate_history`, "explosive under
current policy"), is vintage-robust over 1986-2026 and fires only in
1986-1994 and 2009-2010. **Adopted the same day (Tanishk's call):** `actual` is the default
benchmark (`config/default.yaml`, PARAMETERS.md P-04), `reaction` defines
the debt limit, `envelope` is a sensitivity. `evaluate_paths` carries the
`actual` branch: on simulated paths the feasible balance is the path's own
trailing primary balance, so P(trigger) reads "probability the simulated
economy is on an explosive path under the stance it is actually running".
Results in `docs/methodology.md` section 7 predate this switch and must be
re-run.

**Derivatives.** The first difference is decomposed into snowball, primary,
and stock-flow residual every period. The second difference is decomposed
the same way to say what is accelerating the ratio. The third difference is
computed and tested as a leading indicator (share of triggered paths on which
it was positive beforehand) but never used to trigger: three differences of
quarterly data are noise.

**Free parameters.** N (persistence, default 12 months) and the envelope
quantile (default 1.0). Both are swept in the notebook; nothing else is
chosen.

**Validation.** The trigger is applied to history first (`evaluate_history`)
so the notebook shows which quarters since 1980 would have flagged and
whether they correspond to anything real.

**⚠️ Open question (steelman review, 2026-09-17, see `steelman-log.md`).**
`trigger_persistence_months` is well-supported as defining when the bad
equilibrium becomes *reachable* (a fundamentals question). Whether it's also
being used as a proxy for when a crisis actually *fires* is a distinct claim
the self-fulfilling-crisis literature doesn't support, real crisis timing is
described as belief/coordination-driven with "wide latitude" relative to
fundamentals (Aguiar-Chatterjee-Cole-Stangebye 2016). Check whether either
macro block carries a belief-shock layer on top of the identity trigger, or
whether firing is purely persistence-gated, before treating the trigger's
output as a crisis-timing prediction rather than a reachability condition.
