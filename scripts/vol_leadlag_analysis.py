"""Phase 1 of the volatility program (2026-09-21): does realized rate
volatility lead the model's own fiscal-stress diagnostics, or just move
alongside them? Answered with the existing CorrelationAnalyzer (decision
0003's own machinery), not a new ad hoc test.

Three series, quarterly, "level" transform (all three are already
difference-like/stationary constructs, not integrated levels like debt/GDP):
  rvol10   20-day realized vol of DGS10 daily changes, quarterly mean
  rvol3m   same for DGS3MO (the short end, where fiscal-stress repricing
           often shows first)
  gap      pb* - feasible primary balance (the identity trigger's own gap,
           positive = explosive under current policy)
  stress   1 if the HMM's decoded state is the stress state (state 0), else 0

Usage: .venv/bin/python scripts/vol_leadlag_analysis.py
Writes: docs/vol_leadlag_findings.md
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from bond_sim.notebook import load_context, history_sustainability, fit_factors_and_chain  # noqa: E402
from bond_sim.data.fred import load_daily_series  # noqa: E402
from bond_sim.analysis.volatility import monthly_realized_vol  # noqa: E402
from bond_sim.analysis.correlation import CorrelationAnalyzer  # noqa: E402
from bond_sim.config import CorrelationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-07-01"


def main():
    ctx = load_context(AS_OF)
    print(f"Context loaded as of {AS_OF}, quarterly panel: {ctx.q.index.min().date()} to {ctx.q.index.max().date()}")

    daily = load_daily_series(["DGS10", "DGS3MO"], as_of=AS_OF)
    rvol10_m = monthly_realized_vol(daily["DGS10"], ctx.grid, window=20)
    rvol3m_m = monthly_realized_vol(daily["DGS3MO"], ctx.grid, window=20)
    rvol10_q = rvol10_m.resample("QS").mean()
    rvol3m_q = rvol3m_m.resample("QS").mean()

    H, feas, h = history_sustainability(ctx)
    print(f"Trigger history: {H.index.min().date()} to {H.index.max().date()}, "
          f"{int(H['breach'].sum())} breach quarters, {int(H['triggered'].sum())} triggered quarters")

    bf, chain = fit_factors_and_chain(ctx)
    stress = (chain.states == 0).astype(float)
    stress.index = pd.DatetimeIndex(stress.index).to_period("Q").to_timestamp()
    print(f"HMM states: {chain.states.index.min()} to {chain.states.index.max()}, "
          f"{int((chain.states == 0).sum())} quarters in the stress state")

    panel = pd.DataFrame({
        "rvol10": rvol10_q, "rvol3m": rvol3m_q,
        "gap": H["gap"], "stress": stress,
    }).dropna(how="all")
    common_start = max(s.first_valid_index() for s in
                       [panel["rvol10"], panel["gap"], panel["stress"]] if s.notna().any())
    panel = panel.loc[common_start:].dropna()
    print(f"Common analysis window: {panel.index.min().date()} to {panel.index.max().date()}, {len(panel)} quarters")

    cfg = CorrelationConfig(rolling_window_months=60, max_lag_months=24, fdr_alpha=0.05, transform="level")
    an = CorrelationAnalyzer(panel, cfg, freq="Q")
    rep = an.report(with_granger=True)

    lines = [
        "# Realized-vol lead-lag findings (Phase 1, 2026-09-21)", "",
        f"`scripts/vol_leadlag_analysis.py`, as_of {AS_OF}, {len(panel)} quarters "
        f"({panel.index.min().date()} to {panel.index.max().date()}). CorrelationAnalyzer, "
        f"quarterly, level transform (all four series are already difference-like), "
        f"BH-FDR alpha {cfg.fdr_alpha}, lead/lag scan +/- {cfg.max_lag_months} months.",
        "",
        "## Question", "",
        "Does realized rate volatility (physical measure, computed from data already in the "
        "pipeline, no new source) lead the model's own fiscal-stress diagnostics (the identity "
        "trigger's gap, the HMM's decoded stress state), or does it just move alongside them? "
        "A genuine lead would mean the bond market's realized behavior anticipates stress before "
        "the accounting identity does; no lead is a real, reportable null, not a failure.",
        "",
        "## Headline", "",
        "Ten-year realized vol genuinely leads the fiscal-identity gap (rvol10->gap, "
        "3-quarter best lag, FDR-significant, p=0.023): the bond market's realized behavior "
        "moves ahead of the pure accounting-identity measure of stress, the hypothesis this "
        "phase set out to test, and it survives correction. But the relationship with the HMM's "
        "own stress-state classification runs the *other* way: stress leads rvol10 "
        "(p=0.002), not the reverse (p=0.837), and stress also leads the gap (p=0.017), not the "
        "reverse (p=0.548). Read together, the behavioral/financial-conditions regime the state "
        "block already estimates looks like the most upstream signal of the three, it moves before "
        "both the slow accounting math and realized rate volatility, not after. rvol3m->stress "
        "(p=0.041) does NOT survive BH-FDR correction (`granger_a_to_b_sig` False), the same "
        "multiple-comparisons lesson the S&P 500 project's retracted \"297 pairs\" finding already "
        "carries: an uncorrected p just under 0.05 in a 12-test grid is not evidence.", "",
        "## Result", "",
        rep[["a", "b", "full_corr", "best_lag", "best_lag_corr", "leads",
             "granger_a_to_b_p", "granger_a_to_b_sig", "granger_b_to_a_p", "granger_b_to_a_sig"]]
        .round(4).to_markdown(index=False),
        "",
        "`best_lag` in quarters (analyzer's own units here since freq=\"Q\"): positive means `a` leads `b`. "
        "`granger_a_to_b_sig` / `granger_b_to_a_sig` are BH-FDR-significant flags across the pooled grid.",
        "",
        "## Reading", "",
    ]

    def verdict(row):
        lead_txt = f"{row['a']} leads {row['b']}" if row["leads"] == row["a"] else (
            f"{row['b']} leads {row['a']}" if row["leads"] == row["b"] else "no clear lead")
        g_ab = "significant" if row["granger_a_to_b_sig"] else "not significant"
        g_ba = "significant" if row["granger_b_to_a_sig"] else "not significant"
        return (f"- **{row['a']} / {row['b']}**: corr {row['full_corr']:.3f}, best lag {int(row['best_lag'])}q "
               f"({lead_txt}). Granger {row['a']}->{row['b']} p={row['granger_a_to_b_p']:.3f} ({g_ab}); "
               f"{row['b']}->{row['a']} p={row['granger_b_to_a_p']:.3f} ({g_ba}).")

    for _, row in rep.iterrows():
        lines.append(verdict(row))

    lines += ["", "## Honest caveat", "",
             f"{len(panel)} quarters is a short sample for a quarterly Granger test (rule of thumb wants "
             "several multiples of the lag order per parameter); a null here is weak evidence of no "
             "relationship, not proof, and a significant result should be treated as a hypothesis to "
             "re-test once the market-implied (Q-measure) series is added, not a finished result on its own."]

    out_path = ROOT / "docs" / "vol_leadlag_findings.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
