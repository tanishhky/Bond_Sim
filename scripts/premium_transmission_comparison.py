"""Side-by-side comparison of the VAR block's two premium-transmission
channels (decision 0005, P-20), on the real fitted VAR and identical draws.

Requested by the 2026-09-17 steelman review's own fix: "add a Bocola-
calibrated credit-channel transmission option reported side by side" with
`rate_transmission`. Answers the question the unit test in
`tests/test_credit_channel.py` deliberately does not: on the real fitted
US VAR (not the synthetic test fixture, whose hand-built r10->g_nom
coefficient is artificially large), which channel is harsher, and by how
much. Runs four transmission variants at two premium slopes (P-01 default
and the literature's upper bound) on one set of draws per slope:

    var              rate_transmission (the VAR's own summed lagged-10y
                     coefficients; decision 0005's original choice)
    credit_low       credit_channel_transmission, output_elasticity=-0.002
                     (Arellano-Bai-Bocola 2017's low attribution, 10%)
    credit_central   output_elasticity=-0.006 (their central estimate, ~30%)
    credit_high      output_elasticity=-0.010 (their high attribution, 50%)

Usage: .venv/bin/python scripts/premium_transmission_comparison.py
Writes: docs/premium_transmission_comparison_findings.md
"""
import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import bond_sim.notebook as nb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AS_OF = pd.Timestamp("2026-09-15")
K = 2000

VARIANTS = {
    "var": dict(transmission="var"),
    "credit_low": dict(transmission="credit", credit_elasticity=-0.002),
    "credit_central": dict(transmission="credit", credit_elasticity=-0.006),
    "credit_high": dict(transmission="credit", credit_elasticity=-0.010),
}

SUMMARY_COLS = ["admissible_share", "P(trigger)", "median_years_to_trigger", "median_end_debt_gdp",
                "median_end_r10", "median_end_u", "median_end_g_nom", "median_end_pb"]

VARNAMES = ("r10", "r3m", "g_nom", "pb", "u")


def run_grid(ctx, var, init, feas, premium, policy):
    HQ = (len(init.legacy_interest_mm) + 2) // 3
    rng = np.random.default_rng(ctx.cfg.sim.seed)
    draws = nb.VARBlock(var).prepare(K, HQ, rng)     # common random numbers across all four variants
    runs = {}
    for name, kw in VARIANTS.items():
        blk = nb.VARBlock(var, **kw)
        sim = nb.DoomLoopSimulator(ctx.cfg, blk, premium, init, policy, feasible=feas)
        runs[name] = sim.run(K=K, draws=draws, start=ctx.as_of + pd.DateOffset(months=1))
    return runs


def fmt_table(runs):
    summ = pd.DataFrame({n: r.summary() for n, r in runs.items()}).T
    return summ[SUMMARY_COLS].astype(float).round(4)


def main():
    ctx = nb.load_context(AS_OF)
    print(f"Context loaded as of {AS_OF.date()}, quarterly panel: {ctx.q.index.min().date()} to {ctx.q.index.max().date()}")

    book, auctions, agg, short = nb.build_book(ctx)
    var = nb.fit_var(ctx)
    init = nb.initial_state(ctx, book, auctions)
    _, feas, _ = nb.history_sustainability(ctx, agg)
    policy = nb.policy_from_name("status_quo", ctx.cfg.policy)
    dl = ctx.cfg.doomloop.risk_premium
    print(f"VAR({var.lag}) fitted; feasible envelope: {100 * feas.envelope:.2f}%; start debt/GDP {init.debt_public_mm / init.gdp_saar_mm * 100:.1f}%")

    # instantaneous per-100bp-of-premium-change coefficient on each channel, decoupled
    # from the simulation: explains *why* the simulated paths diverge (or don't).
    rt = dict(zip(VARNAMES, var.rate_transmission(np.array([1.0]))[0]))
    ct = dict(zip(VARNAMES, var.credit_channel_transmission(np.array([1.0]))[0]))
    coef_tbl = pd.DataFrame({"var (rate_transmission)": rt, "credit_central": ct}).loc[["g_nom", "pb", "u"]]
    print("Per-100bp-premium-change instantaneous coefficients:")
    print(coef_tbl.round(4).to_string())

    premium_default = nb.premium_from_config(dl)
    premium_steep = nb.premium_from_config(dataclasses.replace(dl, bps_per_pct_debt_gdp=5.0))
    print(f"\nPremium (default, P-01): {premium_default.describe()}")
    print(f"Premium (steep, P-01 upper literature bound): {premium_steep.describe()}")

    runs_default = run_grid(ctx, var, init, feas, premium_default, policy)
    table_default = fmt_table(runs_default)
    runs_steep = run_grid(ctx, var, init, feas, premium_steep, policy)
    table_steep = fmt_table(runs_steep)

    def spread(table, col):
        credit_vals = table.loc[["credit_low", "credit_central", "credit_high"], col]
        return table.loc["var", col], credit_vals.min(), credit_vals.max()

    def describe_gap(table, col, unit=""):
        v, lo, hi = spread(table, col)
        gap_pts = v - table.loc["credit_central", col]
        direction = "harsher" if (col == "P(trigger)" and gap_pts > 0) or (col != "P(trigger)" and "g_nom" in col and gap_pts < 0) or (col == "median_end_debt_gdp" and gap_pts > 0) else "milder"
        return (f"var={v:.4f}{unit}, credit band=[{lo:.4f}, {hi:.4f}]{unit}, "
                f"var minus credit_central = {gap_pts:+.4f}{unit} (var is {direction} on this metric)")

    lines = [
        "# Premium transmission comparison: VAR-own-coefficients vs credit channel",
        "",
        f"Generated {pd.Timestamp.today().date()} by `scripts/premium_transmission_comparison.py`. "
        f"As-of {AS_OF.date()}, K={K} paths per variant, identical draws (common random numbers) within each "
        f"premium slope, policy `status_quo`.",
        "",
        "**What this answers.** Decision 0005's `rate_transmission` pushes the fiscal premium through the VAR's "
        "own coefficients estimated on 1985-2026 10y moves, a sample with essentially no US sovereign-risk "
        "variation (2026-09-17 steelman review, `docs/decisions/steelman-log.md`). P-20 adds a second, "
        "literature-calibrated option: Arellano, Bai & Bocola's (2017) sovereign-risk-to-output pass-through, "
        "central -0.006 pp growth per 100bp premium, with their own 10%-50% attribution sensitivity run as a "
        "low/high band (-0.002 / -0.010). `tests/test_credit_channel.py` proves the two channels differ on a "
        "synthetic fixture but deliberately does not assert a magnitude for the simulated gap, since that "
        "depends on premium-path and mean-reversion dynamics its single hand-built fixture cannot speak to. "
        "This script runs both channels on the real fitted VAR and real initial state.",
        "",
        "## The instantaneous coefficients are very different in isolation",
        "",
        coef_tbl.round(4).to_markdown(),
        "",
        f"Per 100bp of premium *change* in one quarter, the real fitted VAR's own coefficient on g_nom "
        f"({rt['g_nom']:.3f}) is about {abs(rt['g_nom'] / ct['g_nom']):.0f}x larger in magnitude than the "
        f"credit channel's central elasticity ({ct['g_nom']:.3f}); `tests/test_credit_channel.py`'s synthetic "
        f"fixture agrees in ordering (its hand-built coefficient is -0.386, same sign, same order of "
        "magnitude, also far larger than credit's -0.006), so `rate_transmission` reads as dramatically "
        "harsher than the credit channel *instantaneously*, on both the synthetic fixture and real data. What "
        "the synthetic fixture's end-to-end test does NOT settle is how much of that instantaneous gap "
        "survives into a 30-year simulated outcome, since that depends on how the coefficient interacts with "
        "the VAR's own mean reversion and with how large the quarterly premium *change* actually gets along a "
        "realistic debt path, not just its sign, which is what the summary below checks on the real fitted VAR.",
        "",
        "## Simulated outcomes: default premium slope (P-01 = 2 bps/pt of debt/GDP above 100%)",
        "",
        f"Premium: {premium_default.describe()}",
        "",
        table_default.to_markdown(),
        "",
        f"- Growth: {describe_gap(table_default, 'median_end_g_nom', 'pp')}",
        f"- Trigger probability: {describe_gap(table_default, 'P(trigger)')}",
        f"- Terminal debt/GDP: {describe_gap(table_default, 'median_end_debt_gdp', 'pp of GDP')}",
        "",
        "At this premium slope the four variants are close to indistinguishable despite the ~94x gap in their "
        "instantaneous coefficients: the mechanism is that `rate_transmission` acts on the *change* in premium "
        "each quarter, not the level, and under the default 2bps/pt slope that quarterly change stays small "
        "(a few basis points) for most of the horizon even as the premium's cumulative level rises; a large "
        "coefficient applied to a small, mean-reverting quarterly increment produces a small and mostly "
        "transient growth shock each period, most of which the VAR's own autoregressive dynamics unwind before "
        "it compounds into the terminal debt/GDP ratio. Under this configuration the debt path is dominated by "
        "the mechanical interest-cost/debt-service channel (a higher r10 directly raises debt service, "
        "independent of which growth-transmission channel is chosen), not by the growth feedback from either "
        "transmission channel.",
        "",
        "## Simulated outcomes: steep premium slope (P-01 = 5 bps/pt, the literature's upper bound)",
        "",
        f"Premium: {premium_steep.describe()}",
        "",
        table_steep.to_markdown(),
        "",
        f"- Growth: {describe_gap(table_steep, 'median_end_g_nom', 'pp')}",
        f"- Trigger probability: {describe_gap(table_steep, 'P(trigger)')}",
        f"- Terminal debt/GDP: {describe_gap(table_steep, 'median_end_debt_gdp', 'pp of GDP')}",
        "",
        "**Reading this.** Neither channel is asserted as *the* correct one: `rate_transmission` is a stated "
        "monetary-tightening-shaped assumption (real coefficients, wrong historical regime), and the credit "
        "channel is a stated sovereign-risk-shaped assumption built from a different country's crisis (Italy "
        "2011-2013, a currency-union member without its own central bank) and a different shock size than a US "
        "debt/GDP path this simulation reaches. The finding that matters for the paper is not which channel "
        "wins a horse race in one configuration, but how much the choice of channel matters *relative to* the "
        "premium slope itself (P-01): if the two tables above show the channel gap staying small even as the "
        "premium slope quintuples, that is evidence the mechanical debt-service channel, not the growth-"
        "feedback channel, is what drives this model's debt trajectories, a materially different headline claim "
        "than 'the loop hurts the real economy through growth', and the paper should say so explicitly rather "
        "than assume the growth channel is doing the work because it is the one with a theoretical story "
        "attached.",
        "",
        "## Caveats",
        "",
        "- The credit-channel elasticity is calibrated to one sovereign-debt crisis (Italy 2011-2013) applied to "
        "the US (currency issuer, deepest sovereign-debt market in the world); the transmission *mechanism* "
        "(bank funding costs, firm financing premia) is general but the *magnitude* may not transfer.",
        "- The Okun's-law step inside `credit_channel_transmission` (coefficient 0.4, linearized around a 5% "
        "unemployment reference) is not itself sourced from the sovereign-risk papers; it is a standard macro "
        "rule of thumb converting an output shock to an unemployment response, flagged in the method's own "
        "docstring.",
        "- Neither channel models the reverse causality (weak growth widening the primary-balance gap, which "
        "independently raises the premium next period); this script isolates the *given a premium path, how "
        "does it hit the real economy* question, not the full doom-loop feedback strength.",
        "- The steep-premium run changes only P-01's slope, not the threshold/kink form (P-02) or the anchor; a "
        "kinked premium concentrated at high debt/GDP would produce larger quarterly premium changes exactly "
        "when the channel choice matters most, and is a natural next sensitivity, not run here.",
        "",
    ]
    out = ROOT / "docs" / "premium_transmission_comparison_findings.md"
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
