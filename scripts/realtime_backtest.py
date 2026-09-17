"""Real-time vs hindsight trigger backtest (decision 0002 follow-up).

For each quarter start, rebuilds the context as of that date (vintage macro
panel, book through that date), fits the feasible envelope on what was known
then, and records what the trigger said. Compares with the hindsight table
(today's vintage over all of history). Writes:

    data_cache/realtime_backtest_rt.csv      one row per as_of date
    data_cache/realtime_backtest_cmp.csv     aligned real-time vs hindsight flags
    docs/realtime_trigger_backtest.md        the summary the paper needs

Usage: .venv/bin/python scripts/realtime_backtest.py [start] [end] [benchmark]
       benchmark: envelope | reaction | min (default: the config's feasible_benchmark).
       A non-default benchmark writes to *_<benchmark>.md / .csv instead. The reaction
       function is re-estimated on each as-of history, so it needs a minimum sample
       (40 quarters) before its rows start.
"""
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from bond_sim.notebook import cfgmod, load_context, history_frame, history_sustainability  # noqa: E402
from bond_sim.sim import evaluate_history_realtime, compare_realtime_final  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
START = sys.argv[1] if len(sys.argv) > 1 else "1986-01-01"
END = sys.argv[2] if len(sys.argv) > 2 else "2026-07-01"
FIRST_TRUE_GDP_VINTAGE = "1991-12-04"   # ALFRED starts tracking GDP here; earlier as_of rows use backfilled vintages


def main():
    cfg = cfgmod.load()
    dl = cfg.doomloop
    which = sys.argv[3] if len(sys.argv) > 3 else dl.feasible_benchmark
    suffix = "" if which == dl.feasible_benchmark else f"_{which}"
    min_obs = 2 if which == "envelope" else 40
    dates = pd.date_range(START, END, freq="QS")
    t0 = time.time()
    n = [0]

    def history_fn(t):
        n[0] += 1
        if n[0] % 8 == 1:
            print(f"  as_of {t.date()}  ({n[0]}/{len(dates)}, {time.time() - t0:.0f}s)", flush=True)
        return history_frame(load_context(t))

    rt = evaluate_history_realtime(history_fn, dates, n_quarters=4, which=which,
                                   require_r_gt_g=dl.trigger_require_r_gt_g,
                                   growth_smoothing_quarters=max(dl.trigger_growth_smoothing_months // 3, 1),
                                   quantile=dl.feasible_quantile, fit_reaction=(which != "envelope"), min_obs=min_obs)
    H_final, feas_final, h_final = history_sustainability(load_context(dates[-1]), n_quarters=4, which=which)
    cmp = compare_realtime_final(rt, H_final)

    (ROOT / "data_cache").mkdir(exist_ok=True)
    rt.to_csv(ROOT / "data_cache" / f"realtime_backtest_rt{suffix}.csv")
    cmp.to_csv(ROOT / "data_cache" / f"realtime_backtest_cmp{suffix}.csv")

    rt["lag_quarters"] = ((rt.index.year - rt["date"].dt.year) * 4 + (rt.index.quarter - rt["date"].dt.quarter))
    pre = rt.index < pd.Timestamp(FIRST_TRUE_GDP_VINTAGE)
    dis = cmp[~cmp["agree"]]
    trig_rt = rt.index[rt["triggered"]]
    trig_fn = H_final.index[H_final["triggered"]]

    lines = [
        "# Real-time vs hindsight trigger backtest",
        "",
        f"Generated {pd.Timestamp.today().date()} by `scripts/realtime_backtest.py`. Trigger: "
        f"benchmark `{which}`, quantile {dl.feasible_quantile}, r > g required = {dl.trigger_require_r_gt_g}, "
        f"persistence 4 quarters, growth smoothing {max(dl.trigger_growth_smoothing_months // 3, 1)} quarters. "
        f"Config hash from the run context. {len(rt)} as-of dates ({rt.index[0].date()} to {rt.index[-1].date()}), "
        f"{time.time() - t0:.0f}s.",
        "",
        "**What real time means here.** At each as-of date the macro panel is the ALFRED vintage current on that "
        "date, the book runs through that date, the feasible envelope is the quantile of the primary-balance "
        "history visible then (expanding window), and the growth smoothing uses only past quarters. The hindsight "
        "table is `evaluate_history` on today's vintage. Before "
        f"{FIRST_TRUE_GDP_VINTAGE} (first true GDP vintage) the real-time rows use backfilled vintages "
        "(first-vintage value, conservative publication lag; decision 0002), so they are approximate: "
        f"{int(pre.sum())} of {len(rt)} rows.",
        "",
        "## Headline",
        "",
        f"- Hindsight table (today's vintage) triggered quarters: {len(trig_fn)} "
        f"({', '.join(str(d.date()) for d in trig_fn[:12])}{', ...' if len(trig_fn) > 12 else ''})",
        f"- Real-time table triggered as-of dates: {len(trig_rt)} "
        f"({', '.join(str(d.date()) for d in trig_rt[:12])}{', ...' if len(trig_rt) > 12 else ''})",
        f"- Agreement on the aligned observation dates: {cmp['agree'].mean():.1%} of {len(cmp)}; "
        f"disagreements: {len(dis)}",
        f"- Median reporting lag between the as-of date and the last quarter the trigger could see: "
        f"{rt['lag_quarters'].median():.0f} quarter(s) (min {rt['lag_quarters'].min()}, max {rt['lag_quarters'].max()})",
        f"- Real-time feasible envelope ranged {rt['envelope_asof'].min():.3f} to {rt['envelope_asof'].max():.3f} "
        f"(decimal of GDP); hindsight envelope {feas_final.envelope:.3f}",
        "",
        "## Disagreements (observation date, real-time vs hindsight)",
        "",
    ]
    if len(dis):
        tbl = dis[["triggered_realtime", "triggered_final", "feasible_realtime", "feasible_final"]].copy()
        tbl.index = [d.date() for d in tbl.index]
        lines.append(tbl.round(4).to_markdown())
    else:
        lines.append("None.")
    lines += ["", "## Real-time rows that triggered", ""]
    if len(trig_rt):
        t = rt.loc[rt["triggered"], ["date", "n_obs", "envelope_asof", "pb_star", "feasible", "gap", "lag_quarters"]].copy()
        t["date"] = t["date"].dt.date
        t.index = [d.date() for d in t.index]
        lines.append(t.round(4).to_markdown())
    else:
        lines.append("None.")
    (ROOT / "docs" / f"realtime_trigger_backtest{suffix}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nwritten: docs/realtime_trigger_backtest{suffix}.md, data_cache/realtime_backtest_*{suffix}.csv")


if __name__ == "__main__":
    main()
