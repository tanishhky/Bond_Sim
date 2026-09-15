"""Command line: the pipeline in five verbs, every artifact stamped with the config hash.

    bond_sim fetch      pull and cache all data; print the vintage-coverage table
    bond_sim book       build the bond book, validate against Treasury publications
    bond_sim analyze    correlation / regime / lead-lag / Granger report on the thesis pairs
    bond_sim simulate   fit the macro VAR, run every policy under common random numbers,
                        map paths to sector employment, measure recovery half-lives
    bond_sim all        the four above in order

Artifacts land in runs/<config_hash>/ as CSV/NPY plus a JSONL event log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfgmod
from . import obs
from .calendar import MonthlyGrid

# Thesis pairs: fiscal state x rates x labor. Same-group pairs are informative but
# drown the cross-group signal, so they are excluded unless --all-pairs is given.
THESIS_GROUPS = {
    "fiscal": ["FYGFGDQ188S", "MTSDS133FMS", "A091RC1Q027SBEA", "FDHBFIN", "FDHBFRBN"],
    "rates": ["DGS10", "DGS3MO", "THREEFYTP10"],
    "macro": ["GDP", "CPIAUCSL"],
    "labor": ["UNRATE", "JTSLDL", "ICSA", "USCONS", "CES5553000001", "CES3133600101",
              "CES5552200001", "CES9091000001"],
}
SIM_SERIES = ["DGS10", "DGS3MO", "GDP", "MTSDS133FMS", "MTSR133FMS", "A091RC1Q027SBEA", "UNRATE"]
SIM_PATH_VARS = ("debt_gdp", "r10", "r10_base", "premium", "u", "g_nom", "pb", "interest_12m_gdp", "interest_12m_receipts", "pool_coupon")


def _grid(cfg, as_of):
    return MonthlyGrid.from_history_plus_horizon(cfg.data.history_start, as_of, cfg.calendar.horizon_years)


def _as_of(cfg, arg):
    return pd.Timestamp(arg or cfg.data.as_of or pd.Timestamp.today().normalize())


def _outdir(cfg, root: Path) -> Path:
    d = root / cfg.content_hash()
    d.mkdir(parents=True, exist_ok=True)
    obs.configure(log_dir=d / "logs")
    return d


def _quarterly_panel(w: pd.DataFrame, ids) -> pd.DataFrame:
    """Mixed monthly/quarterly panel -> quarterly (flows summed, everything else averaged)."""
    from .data import SERIES
    flows = [s for s in ids if SERIES[s].agg == "sum" and s in w.columns]
    wq = w.resample("QS").mean()
    if flows:
        wq[flows] = w[flows].resample("QS").sum(min_count=3)
    return wq


# ── fetch ───────────────────────────────────────────────────────────────────

def cmd_fetch(cfg, args):
    from .data import FredClient, SERIES, load_auctions, load_mspd_summary, load_interest_expense, load_debt_to_penny
    as_of = _as_of(cfg, args.as_of)
    out = _outdir(cfg, Path(args.runs))
    cov = FredClient().vintage_coverage(list(SERIES))
    cov.to_csv(out / "vintage_coverage.csv")
    counts = {"auctions": len(load_auctions(as_of=as_of)), "mspd_summary": len(load_mspd_summary(as_of=as_of)),
              "interest_expense": len(load_interest_expense(as_of=as_of)), "debt_to_penny": len(load_debt_to_penny(as_of=as_of))}
    print(cov[["freq", "first_obs", "first_true_vintage", "n_obs", "n_backfilled_obs"]].to_string())
    print(f"\nrows: {counts}\nartifacts: {out}")


# ── book ────────────────────────────────────────────────────────────────────

def cmd_book(cfg, args):
    from .data import load_auctions, load_mspd_summary, load_interest_expense, load_debt_to_penny, load_macro_panel
    from .data.fred import panel_wide
    from .engine.book import BondBook
    from .engine.validate import BookValidator
    from .sim.setup import realized_short_rate
    as_of = _as_of(cfg, args.as_of)
    grid = _grid(cfg, as_of)
    out = _outdir(cfg, Path(args.runs))
    auctions = load_auctions(as_of=as_of)
    book = BondBook.from_auctions(auctions, grid, cfg.book)
    panel = panel_wide(load_macro_panel(["DGS3MO"], as_of=as_of, grid=grid), grid)
    short = realized_short_rate(panel, grid)
    agg = book.aggregates(short_rate=short).frame
    agg.to_csv(out / "book_aggregates.csv")
    print(f"book: {book.N} tranches on {grid}")
    print(agg.loc[agg.index <= as_of].tail(3)[["outstanding", "interest", "principal", "debt_service"]].round(0).to_string())
    if args.validate:
        rep = BookValidator(book).run(load_mspd_summary(as_of=as_of), load_interest_expense(as_of=as_of),
                                      load_debt_to_penny(as_of=as_of), short_rate=short)
        rep.outstanding.to_csv(out / "validate_outstanding.csv")
        rep.interest.to_csv(out / "validate_interest.csv")
        rep.penny.to_csv(out / "validate_penny.csv")
        print("\nvalidation, mean ratio book/published over the last 5 years:")
        print(rep.headline().round(4).to_string())
    print(f"artifacts: {out}")


# ── analyze ─────────────────────────────────────────────────────────────────

def cmd_analyze(cfg, args):
    from .analysis import CorrelationAnalyzer, RegimeDetector
    from .data import load_macro_panel
    from .data.fred import panel_wide
    as_of = _as_of(cfg, args.as_of)
    grid = _grid(cfg, as_of)
    out = _outdir(cfg, Path(args.runs))
    if args.series:
        ids, group_of = args.series, None
    else:
        ids = [s for g in THESIS_GROUPS.values() for s in g]
        group_of = {s: g for g, ss in THESIS_GROUPS.items() for s in ss} if args.thesis else None
    w = panel_wide(load_macro_panel(ids, as_of=as_of, grid=grid), grid)
    wq = _quarterly_panel(w.loc[w.index <= as_of], ids)
    an = CorrelationAnalyzer(wq, cfg.correlation, freq="Q")
    rep = an.report(with_granger=not args.no_granger, group_of=group_of)
    rep.to_csv(out / "correlation_report.csv", index=False)
    an.full_matrix().to_csv(out / "correlation_matrix.csv")
    print(rep.head(25).round(3).to_string(index=False))
    det = RegimeDetector(cfg.correlation)
    top = rep[rep["regime_dependent"]].head(12)
    labels = {f"{r.a}|{r.b}": det.label(an.rolling(r.a, r.b).dropna()) for r in top.itertuples()}
    if len(labels) >= 2:
        cons = det.consistency(labels)
        cons.to_csv(out / "regime_consistency_ari.csv")
        off = cons.to_numpy()[~np.eye(len(cons), dtype=bool)]
        print(f"\nregime consistency (mean pairwise ARI over {len(labels)} regime-dependent pairs): {np.nanmean(off):.3f}")
        print("  (near 0: regimes are pair-specific, keep one covariance; near 1: one market-wide regime, switch it)")
    print(f"artifacts: {out}")


# ── simulate ────────────────────────────────────────────────────────────────

def _quarterly_changes(monthly: np.ndarray, x0: float) -> np.ndarray:
    """(K, H) monthly levels -> (K, HQ) end-of-quarter changes, first change vs x0."""
    eoq = monthly[:, 2::3]
    prev = np.concatenate([np.full((monthly.shape[0], 1), x0), eoq[:, :-1]], axis=1)
    return eoq - prev


def _diagnostics(res) -> dict:
    d = {f"rejected_{k}": int(v) for k, v in res.rejections.items()}
    d["admissible_share"] = float(res.admissible.mean())
    return d


def cmd_simulate(cfg, args):
    from .data import load_auctions, load_mspd_summary, load_macro_panel
    from .data.fred import panel_wide
    from .engine.book import BondBook
    from .sim import (DoomLoopSimulator, LaborModel, MacroVAR, RecoveryModel, VARBlock, premium_from_config,
                      policy_from_name)
    from .sim.macro import build_quarterly_state
    from .sim.setup import build_initial_state
    as_of = _as_of(cfg, args.as_of)
    grid = _grid(cfg, as_of)
    out = _outdir(cfg, Path(args.runs))

    # 1. Data and the estimated macro block
    ids = SIM_SERIES + list(cfg.labor.sectors)
    w = panel_wide(load_macro_panel(ids, as_of=as_of, grid=grid), grid)
    w = w.loc[w.index <= as_of]
    q = build_quarterly_state(w)
    var = MacroVAR(max_lag=cfg.doomloop.var_max_lag).fit(q, start=args.var_start or cfg.doomloop.var_start)
    print(f"VAR({var.lag}) on {len(var.sample)} quarters {var.sample.index.min().date()}..{var.sample.index.max().date()}")
    print("residual correlation:\n" + var.resid_corr().round(2).to_string())
    print(f"estimated long-run means: {var.long_run_means().round(3).to_dict()}  |  max eigenvalue {var.max_eigenvalue():.3f}")
    means = {k: v for k, v in (("pb", cfg.doomloop.pb_anchor_pct_gdp), ("r10", cfg.doomloop.r10_anchor_pct),
                               ("u", cfg.doomloop.u_anchor_pct)) if v is not None}
    if means:
        var.set_long_run_means(means)
        print("imposed long-run means:   " + repr(var.long_run_means().round(3).to_dict()))

    # 2. The book, the starting point, the premium
    auctions = load_auctions(as_of=as_of)
    book = BondBook.from_auctions(auctions, grid, cfg.book)
    init = build_initial_state(book, w, q, load_mspd_summary(as_of=as_of), auctions, grid, as_of)
    print(f"initial: debt/GDP {100*init.debt_public_mm/init.gdp_saar_mm:.1f}%  10y {init.r10_pct:.2f}  3m {init.r3m_pct:.2f}  "
          f"u {init.u_pct:.1f}  pb(4q) {init.pb_pct_gdp:.2f}% GDP  g_nom(4q) {init.g_nom_pct:.2f}%  "
          f"stock avg remaining maturity {init.avg_new_maturity_months:.0f}m  bill share {100*init.bill_share:.0f}%  "
          f"receipts/GDP {init.receipts_gdp_pct:.1f}%")
    premium = premium_from_config(cfg.doomloop.risk_premium)

    # 3. Sector betas from history (Phase 5) and historical recoveries (Phase 7)
    labor = LaborModel(list(cfg.labor.sectors)).fit(w, q)
    labor.table().to_csv(out / "labor_betas.csv")
    print("\nsector betas (quarterly %-growth on d_u and lagged d_r10):\n" + labor.table().round(3).to_string())
    rec = RecoveryModel()
    hist_rec = rec.historical(w["UNRATE"].dropna())
    hist_rec.to_csv(out / "recovery_historical.csv")
    print("\nhistorical recovery half-lives (months from unemployment peak):\n" + hist_rec[["peak_date", "peak_u", "trough_u", "half_life_months"]].to_string())

    # 4. Every policy on the same shocks
    K = args.paths or cfg.sim.n_paths
    rng = np.random.default_rng(cfg.sim.seed)
    HQ = (len(init.legacy_interest_mm) + 2) // 3
    block = VARBlock(var)
    draws = block.prepare(K, HQ, rng)                        # common random numbers across policies
    rows, diag = [], {}
    for name in (args.policies or cfg.policy.menu):
        sim = DoomLoopSimulator(cfg, VARBlock(var), premium, init, policy_from_name(name, cfg.policy))
        res = sim.run(K=K, draws=draws, start=as_of + pd.DateOffset(months=1))
        for v in SIM_PATH_VARS:
            res.quantiles(v).to_csv(out / f"sim_{name}_{v}.csv")
        np.save(out / f"sim_{name}_trigger_month.npy", res.trigger_month)
        # sector employment paths from the simulated (d_u, d_r10), same rng stream per policy
        d_u = _quarterly_changes(res.paths["u"], init.u_pct)
        d_r = _quarterly_changes(res.paths["r10"], init.r10_pct)
        sectors = labor.simulate(d_u, d_r, np.random.default_rng(cfg.sim.seed + 1))
        sector_end = {s: float(np.median(v[:, -1])) for s, v in sectors.items()}
        sector_trough = {s: float(np.median(v.min(axis=1))) for s, v in sectors.items()}
        pd.DataFrame({"median_end_index": sector_end, "median_trough_index": sector_trough}).to_csv(out / f"sim_{name}_sectors.csv")
        hl = rec.simulated(res.paths["u"], init.u_pct)
        s = res.summary()
        s["median_recovery_half_life_months"] = float(hl.median()) if hl.notna().any() else np.nan
        s["share_paths_with_downturn"] = float(hl.notna().mean())
        s["worst_sector_median_trough"] = min(sector_trough, key=sector_trough.get)
        rows.append(s)
        diag[name] = _diagnostics(res)
    summary = pd.DataFrame(rows).set_index("policy")
    summary.to_csv(out / "sim_summary.csv")
    with open(out / "sim_meta.json", "w") as fh:
        json.dump({"as_of": str(as_of.date()), "K": K, "premium": premium.describe(), "var_lag": var.lag,
                   "long_run_means": var.long_run_means().round(4).to_dict(), "config_hash": cfg.content_hash(),
                   "init": {k: v for k, v in init.__dict__.items() if not isinstance(v, np.ndarray)},
                   "diagnostics": diag}, fh, indent=2, default=str)
    cols = ["P(trigger)", "median_years_to_trigger", "median_end_debt_gdp", "median_end_r10", "median_end_u",
            "median_end_interest_12m_gdp", "median_recovery_half_life_months", "worst_sector_median_trough"]
    print("\n" + summary[cols].round(3).to_string())
    print(f"\nadmissibility: {diag}")
    print(f"artifacts: {out}")


def main(argv=None) -> int:
    import warnings
    # numpy on macOS Accelerate emits spurious "encountered in matmul" RuntimeWarnings for
    # perfectly finite products; every run checks finiteness explicitly (see _diagnostics).
    warnings.filterwarnings("ignore", message=".*encountered in matmul")
    warnings.filterwarnings("ignore", message=".*OpenSSL.*")
    p = argparse.ArgumentParser(prog="bond_sim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--as-of", dest="as_of", default=None)
    p.add_argument("--runs", default="runs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    b = sub.add_parser("book"); b.add_argument("--validate", action="store_true")
    a = sub.add_parser("analyze"); a.add_argument("--series", nargs="*"); a.add_argument("--no-granger", action="store_true")
    a.add_argument("--all-pairs", dest="thesis", action="store_false", help="include same-group pairs")
    s = sub.add_parser("simulate"); s.add_argument("--paths", type=int); s.add_argument("--policies", nargs="*"); s.add_argument("--var-start", default=None)
    sub.add_parser("all")
    args = p.parse_args(argv)
    cfg = cfgmod.load(args.config)
    if args.cmd == "all":
        for name in ("fetch", "book", "analyze", "simulate"):
            ns = argparse.Namespace(**vars(args), validate=True, series=None, no_granger=False, paths=None,
                                    policies=None, var_start=None, thesis=True)
            globals()[f"cmd_{name}"](cfg, ns)
        return 0
    globals()[f"cmd_{args.cmd}"](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
