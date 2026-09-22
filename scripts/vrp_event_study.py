"""Phase 2 of the volatility program (2026-09-22): the rates variance risk
premium, 2003-2020, bounded to VXTYN's real history (CBOE's TYVIX,
discontinued 2020-05-15, no current replacement exists on FRED, see
analysis/volatility.py's module docstring). Final-vintage data, not a
real-time as-of view, explicitly: this is a backward-looking event study,
never an input to the live simulation pipeline.

VRP = VXTYN (implied) - rvol10 (realized), the same construction as
VolEdge's equity variance risk premium, applied to rates. Question: is the
rates VRP, like the equity one, mostly compensation for tail risk (positive
on average, spikes further in genuine crises) rather than a free signal, and
does it lead the model's own fiscal-stress diagnostics any better than
realized vol alone did in Phase 1.

Usage: .venv/bin/python scripts/vrp_event_study.py
Writes: docs/vrp_event_study_findings.md
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from bond_sim.notebook import load_context, history_sustainability, fit_factors_and_chain  # noqa: E402
from bond_sim.data.fred import load_daily_series  # noqa: E402
from bond_sim.analysis.volatility import monthly_realized_vol, variance_risk_premium  # noqa: E402
from bond_sim.analysis.correlation import CorrelationAnalyzer  # noqa: E402
from bond_sim.config import CorrelationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AS_OF = "2026-07-01"   # the live model's as_of; VXTYN itself only runs through 2020-05-15
VXTYN_END = "2020-05-15"

KNOWN_EPISODES = [
    ("2007-08-01", "2009-06-01", "GFC (2007-2009)"),
    ("2010-04-01", "2010-06-01", "2010 flash crash / European debt crisis onset"),
    ("2011-07-01", "2011-09-01", "2011 US debt-ceiling standoff + S&P downgrade"),
    ("2013-05-01", "2013-09-01", "2013 taper tantrum"),
    ("2020-02-01", "2020-04-01", "COVID onset"),
]


def main():
    ctx = load_context(AS_OF)

    daily = load_daily_series(["DGS10", "VXTYN"], as_of=AS_OF)
    vxtyn = daily["VXTYN"]
    print(f"VXTYN: {vxtyn.index.min().date()} to {vxtyn.index.max().date()}, {len(vxtyn)} obs "
          f"(final-vintage, this series' ALFRED metadata does not carry genuine historical timing)")

    rvol10_m = monthly_realized_vol(daily["DGS10"], ctx.grid, window=20)
    vxtyn_m = vxtyn.resample("MS").mean()

    vrp_m = variance_risk_premium(vxtyn_m, rvol10_m)
    print(f"VRP overlap window: {vrp_m.index.min().date()} to {vrp_m.index.max().date()}, {len(vrp_m)} months")

    H, feas, h = history_sustainability(ctx)
    bf, chain = fit_factors_and_chain(ctx)
    stress = (chain.states == 0).astype(float)
    stress.index = pd.DatetimeIndex(stress.index).to_period("Q").to_timestamp()

    vrp_q = vrp_m.resample("QS").mean()
    panel = pd.DataFrame({"vrp": vrp_q, "gap": H["gap"], "stress": stress}).dropna()
    print(f"Quarterly analysis panel: {panel.index.min().date()} to {panel.index.max().date()}, {len(panel)} quarters")

    cfg = CorrelationConfig(rolling_window_months=40, max_lag_months=16, fdr_alpha=0.05, transform="level")
    an = CorrelationAnalyzer(panel, cfg, freq="Q")
    rep = an.report(with_granger=True)

    # descriptive: mean/positive-share of VRP overall and around known stress episodes
    overall_mean = float(vrp_m.mean())
    overall_pos_share = float((vrp_m > 0).mean())
    episode_rows = []
    for start, end, label in KNOWN_EPISODES:
        window = vrp_m.loc[start:end]
        if window.notna().sum() == 0:
            episode_rows.append((label, start, end, None, None))
            continue
        episode_rows.append((label, start, end, float(window.mean()), float(vrp_m.mean())))

    lines = [
        "# Rates variance risk premium: 2003-2020 event study (Phase 2, 2026-09-22)", "",
        f"`scripts/vrp_event_study.py`. VXTYN (TYVIX, final-vintage, discontinued {VXTYN_END}) minus "
        f"20-day realized vol of DGS10, monthly, {vrp_m.index.min().date()} to {vrp_m.index.max().date()} "
        f"({len(vrp_m)} months, {len(panel)} quarters after joining to the trigger history and the HMM "
        "states). Explicitly bounded and backward-looking; not fed into the live simulation.", "",
        "## Descriptive: is the rates VRP, like the equity one, mostly a tail-risk premium?", "",
        f"Mean VRP over the full window: **{overall_mean:.3f}** (implied above realized on average); "
        f"positive in **{overall_pos_share:.1%}** of months.", "",
        "| Episode | Window | Mean VRP in window | Full-sample mean |",
        "|---|---|---|---|",
    ]
    for label, start, end, m, full in episode_rows:
        if m is None:
            lines.append(f"| {label} | {start} to {end} | no VXTYN coverage | |")
        else:
            lines.append(f"| {label} | {start} to {end} | {m:.3f} | {full:.3f} |")

    lines += ["", "## Lead-lag against the model's own diagnostics", "",
             rep[["a", "b", "full_corr", "best_lag", "best_lag_corr", "leads",
                  "granger_a_to_b_p", "granger_a_to_b_sig", "granger_b_to_a_p", "granger_b_to_a_sig"]]
             .round(4).to_markdown(index=False), "",
             "## Reading", ""]

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

    lines += ["", "## Honest caveats", "",
             f"- {len(panel)} quarters is a genuinely short sample, well under Phase 1's 143; treat any "
             "Granger result here as a hypothesis, not a finished result, even more so than Phase 1.",
             "- Final-vintage data throughout: VXTYN's ALFRED vintage metadata doesn't reflect true "
             "historical publication timing, so this whole script uses the latest values, not an as-of "
             "reconstruction. That's fine for a backward-looking event study, it would not be fine for "
             "anything claiming to test what was knowable in real time.",
             "- Bounded to 2003-2020 by data availability, not by choice; says nothing about the current "
             "period. A current MOVE series (Bloomberg, WRDS/ICE) would be needed to extend this."]

    out_path = ROOT / "docs" / "vrp_event_study_findings.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
