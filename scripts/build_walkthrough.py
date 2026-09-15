"""Generate notebooks/01_pipeline_walkthrough.ipynb from code, so the narrative
notebook is regenerable and never drifts from the package API.

    ./.venv/bin/python scripts/build_walkthrough.py
    ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_pipeline_walkthrough.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))       # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))         # noqa: E731

md("""
# Bond_Sim: pipeline walkthrough

**Question.** How likely is a US sovereign-debt doom loop (debt growth pushes yields up, higher yields push
interest cost and future debt growth up), what interrupts it, who is hurt if nothing does, and for how long.

This notebook runs every phase of the engine on real data and explains what each step does and why it is
built the way it is. Each section maps to a module in `src/bond_sim/` and to a decision record in `docs/decisions/`.
Numbers that are placeholders awaiting a human call are listed in `PARAMETERS.md`; nothing in this notebook
should be read as a finding until those are replaced or defended.

Phases 1-2 (the hand-coded toy book, Problems 1-5) live in `main.ipynb`; `tests/test_toy_book.py` pins their
numbers so the engine below can never drift from what was hand-verified.
""")

code("""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
from bond_sim.config import load
from bond_sim.calendar import MonthlyGrid

AS_OF = pd.Timestamp("2026-09-15")
cfg = load()
grid = MonthlyGrid.from_history_plus_horizon(cfg.data.history_start, AS_OF, cfg.calendar.horizon_years)
print(grid, "| config hash", cfg.content_hash())
""")

md("""
## 1. Point-in-time data (decision 0002)

Every macro series is stored as its **full revision history** from ALFRED: one row per observation per
vintage window. "What was known on date X" is then one boolean mask (`VintageFrame.as_of`), and the
`@point_in_time` decorator asserts that nothing a loader returns postdates its `as_of`.

Below: GDP for 2009Q3 was first published at 14,301.5 and has been revised repeatedly since. An analysis
dated November 2009 sees the first number; today's FRED shows the last one. Lead/lag and regime claims
in the paper are made on the former.
""")

code("""
from bond_sim.data import FredClient, SERIES
fc = FredClient()
gdp = fc.fetch_vintages("GDP")
display(gdp.revisions("GDP", "2009-07-01").head(6))
print("known on 2009-11-15:", gdp.as_of("2009-11-15").set_index("date").loc["2009-07-01", "value"],
      "| latest:", gdp.latest().set_index("date").loc["2009-07-01", "value"])
""")

md("""
ALFRED only started tracking most series in the 1990s-2000s. Observations older than a series' first vintage
carry a **backfilled** publication timestamp (observation date plus a conservative typical lag) and are flagged,
so the paper can state exactly which point-in-time results rest on true vintages.
""")

code("""
cov = fc.vintage_coverage(["GDP", "DGS10", "UNRATE", "MTSDS133FMS", "PAYEMS", "FYGFGDQ188S"])
display(cov[["freq", "first_obs", "first_true_vintage", "n_obs", "n_backfilled_obs", "pub_lag_days_assumed"]])
""")

md("""
## 2. The bond book (Phase 3, decisions 0001 and 0004)

The auction ledger (11,113 auctions since 1979) is the primitive: every tranche is a row, every month since
1980 a column. Outstanding, coupons, bill discount interest, and principal are boolean-mask arithmetic on that
grid, exactly the formulas from Problems 1-3, computed two ways (dense masks and index-pair `bincount`)
that the tests assert equal.
""")

code("""
from bond_sim.data import load_auctions, load_macro_panel, load_mspd_summary, load_interest_expense, load_debt_to_penny
from bond_sim.data.fred import panel_wide
from bond_sim.engine.book import BondBook
from bond_sim.engine.validate import BookValidator
from bond_sim.sim.setup import realized_short_rate

auctions = load_auctions(as_of=AS_OF)
book = BondBook.from_auctions(auctions, grid, cfg.book)
w3m = panel_wide(load_macro_panel(["DGS3MO"], as_of=AS_OF, grid=grid), grid)
short = realized_short_rate(w3m, grid)
agg = book.aggregates(short_rate=short).frame
print(f"{book.N} tranches; by kind:", {k: int((book.kind == k).sum()) for k in ("bill", "note", "bond", "tips", "frn")})
hist = agg.loc[agg.index <= AS_OF]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
(hist[[f"outstanding_{k}" for k in ("bill", "note", "bond", "tips", "frn")]] / 1e6).plot.area(ax=ax[0], linewidth=0)
ax[0].set_title("Reconstructed marketable debt outstanding ($tn)"); ax[0].set_xlabel("")
(hist["interest"].rolling(12).sum() / 1e6).plot(ax=ax[1]); ax[1].set_title("Reconstructed interest, trailing 12m ($tn)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
""")

md("""
**Validation, not calibration.** The book is compared to what the Treasury publishes and the gaps are reported,
never fitted away: bills/notes/bonds/FRN reconcile to within ~1.5% of MSPD Table 1; TIPS sit at ~0.83 because
the book carries nominal face without the inflation accrual (decision 0004); the 12-month interest ratio against
Treasury's accrued interest expense is the next gap to decompose by security type.
""")

code("""
rep = BookValidator(book).run(load_mspd_summary(as_of=AS_OF), load_interest_expense(as_of=AS_OF),
                              load_debt_to_penny(as_of=AS_OF), short_rate=short)
display(rep.headline().round(4).to_frame())
ax = rep.outstanding[[c for c in rep.outstanding.columns if c.startswith("ratio_")]].plot(figsize=(11, 3.5), ylim=(0.6, 1.2))
ax.axhline(1, color="k", lw=0.8); ax.set_title("Book / MSPD outstanding by class"); plt.show()
""")

md("""
## 3. Nothing is independent (decision 0003)

For every cross-group pair (fiscal x rates x macro x labor) on the quarterly panel: full-sample correlation,
rolling correlation and its range, a lead/lag scan, and Granger causality both ways with Benjamini-Hochberg
control over the whole grid. A wide rolling range means the relationship is regime-dependent and must not be
summarized by one number; the regime detector then asks whether those regimes coincide across pairs.
""")

code("""
from bond_sim.analysis import CorrelationAnalyzer, RegimeDetector
from bond_sim.cli import THESIS_GROUPS, _quarterly_panel
ids = [s for g in THESIS_GROUPS.values() for s in g]
group_of = {s: g for g, ss in THESIS_GROUPS.items() for s in ss}
w = panel_wide(load_macro_panel(ids, as_of=AS_OF, grid=grid), grid)
wq = _quarterly_panel(w.loc[w.index <= AS_OF], ids)
an = CorrelationAnalyzer(wq, cfg.correlation, freq="Q")
rep_c = an.report(with_granger=True, group_of=group_of)
display(rep_c.head(12)[["a", "b", "full_corr", "rolling_min", "rolling_max", "best_lag", "leads",
                        "granger_a_to_b_sig", "granger_b_to_a_sig"]].round(3))
""")

code("""
pair = ("FYGFGDQ188S", "DGS10")     # debt/GDP vs the 10y: the relationship the whole thesis turns on
r = an.analyze_pair(*pair)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
r.rolling.plot(ax=ax[0]); ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title(f"20-quarter rolling corr: d({pair[0]}) vs d({pair[1]})")
r.lead_lag.plot.bar(ax=ax[1]); ax[1].set_title(f"corr(d{pair[0]}_t, d{pair[1]}_(t+lag)); +lag = {pair[0]} leads")
plt.tight_layout(); plt.show()
det = RegimeDetector(cfg.correlation)
top = rep_c[rep_c["regime_dependent"]].head(12)
labels = {f"{x.a}|{x.b}": det.label(an.rolling(x.a, x.b).dropna()) for x in top.itertuples()}
cons = det.consistency(labels)
off = cons.to_numpy()[~np.eye(len(cons), dtype=bool)]
print(f"mean pairwise adjusted Rand index across {len(labels)} regime-dependent pairs: {np.nanmean(off):.3f} "
      "(near 0 = pair-specific regimes; near 1 = one market-wide regime)")
""")

md("""
## 4. The doom loop (Phase 4, decision 0005)

A quarterly VAR in levels on (10y, 3m, nominal growth, trailing-4Q primary balance, unemployment) supplies the
estimated cross-variable dynamics and the shock covariance. The fiscal premium (`sim/premium.py`, the thesis's
central open parameter P-01) is added on top of the base 10y from last quarter's debt/GDP, and its change is
transmitted to growth, the primary balance, and unemployment through the VAR's own estimated coefficients on the
lagged 10y. Debt accounting is monthly: the legacy book's exact schedules, a refinancing pool for new issuance,
deficits financed at the current bill/coupon blend.
""")

code("""
from bond_sim.sim import MacroVAR, DoomLoopSimulator, premium_from_config, policy_from_name, LaborModel, RecoveryModel
from bond_sim.sim.macro import build_quarterly_state
from bond_sim.sim.setup import build_initial_state
from bond_sim.cli import SIM_SERIES
ws = panel_wide(load_macro_panel(SIM_SERIES + list(cfg.labor.sectors), as_of=AS_OF, grid=grid), grid)
ws = ws.loc[ws.index <= AS_OF]
q = build_quarterly_state(ws)
var = MacroVAR(max_lag=cfg.doomloop.var_max_lag).fit(q, start=cfg.doomloop.var_start)
print(f"VAR({var.lag}) on {len(var.sample)} quarters; max eigenvalue {var.max_eigenvalue():.3f} (stable if < 1)")
display(var.resid_corr().round(2)); display(var.long_run_means().round(2).to_frame("estimated long-run mean"))
init = build_initial_state(book, ws, q, load_mspd_summary(as_of=AS_OF), auctions, grid, AS_OF)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in init.__dict__.items() if not isinstance(v, np.ndarray)})
""")

md("""
**Policies on identical dice.** Every policy is a set of hooks on the same shock draws (common random numbers), so
the difference between two outcome distributions is the policy, not the randomness. Every magnitude inside the
policies is a placeholder (P-06 to P-09); the mechanics are what is being demonstrated here.
""")

code("""
K = 1000
premium = premium_from_config(cfg.doomloop.risk_premium)
print("premium:", premium.describe())
HQ = (len(init.legacy_interest_mm) + 2) // 3
shocks = var.draw_shocks(K, HQ, np.random.default_rng(cfg.sim.seed))
results = {}
for name in cfg.policy.menu:
    sim = DoomLoopSimulator(cfg, var, premium, init, policy_from_name(name, cfg.policy))
    results[name] = sim.run(K=K, shocks=shocks, start=AS_OF + pd.DateOffset(months=1))
summary = pd.DataFrame([r.summary() for r in results.values()]).set_index("policy")
display(summary[["P(trigger)", "median_years_to_trigger", "median_end_debt_gdp", "median_end_r10", "median_end_u",
                 "median_end_interest_12m_gdp"]].round(2))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, v, title in zip(axes, ("debt_gdp", "r10", "u"), ("Debt / GDP (%)", "10y yield (%)", "Unemployment (%)")):
    for name, res in results.items():
        qq = res.quantiles(v)
        ax.plot(qq.index, qq["q50"], label=name)
        if name == "status_quo":
            ax.fill_between(qq.index, qq["q05"], qq["q95"], alpha=0.15)
    ax.set_title(f"{title}: medians by policy (band = status quo 5-95%)")
axes[0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("""
## 5. Who gets hurt, and for how long (Phases 5 and 7)

Sector employment growth is regressed on the unemployment change and lagged 10y changes; the residual covariance
across sectors is kept so sector shocks are drawn together. Recovery half-lives are measured on five US
episodes and on every simulated path (months from the unemployment peak until half the gap is closed).
""")

code("""
labor = LaborModel(list(cfg.labor.sectors)).fit(ws, q)
display(labor.table().round(3))
rec = RecoveryModel()
display(rec.historical(ws["UNRATE"].dropna())[["peak_date", "peak_u", "trough_u", "half_life_months"]])
hl = {name: rec.simulated(res.paths["u"], init.u_pct) for name, res in results.items()}
display(pd.DataFrame({name: {"median_half_life_months": h.median(), "share_paths_with_downturn": h.notna().mean()}
                      for name, h in hl.items()}).T.round(2))
""")

md("""
## 6. What is still a placeholder

`PARAMETERS.md` lists every number a human must decide (premium slope and form, trigger thresholds, the
no-layoff mandate's mechanics and cost, the austerity multiplier, monetization's inflation cost, the growth
uplift, anchors for the primary balance, the long-run 10y and the natural rate). Until those are set, the
probabilities above are properties of the placeholders, and the paper says so.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_pipeline_walkthrough.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
