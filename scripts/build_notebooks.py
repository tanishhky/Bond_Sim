"""Generate the phase notebooks (notebooks/00..07) from code.

The notebooks are the working surface: every documented choice, experiment,
and outcome appears where it was made, with the explanation next to the
code. The implementation lives in the package so tests can reach it and the
notebooks stay readable. Regenerate after any API change:

    ./.venv/bin/python scripts/build_notebooks.py
    for n in notebooks/0*.ipynb; do ./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=3600 "$n"; done
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True)

HEADER = """
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
import bond_sim.notebook as nb
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 80)
plt.rcParams["figure.dpi"] = 110

# ── parameters: change and rerun ─────────────────────────────────────────────
AS_OF = pd.Timestamp("2026-09-15")     # every loader filters on this date (point-in-time contract)
K = {K}                                 # Monte Carlo paths (raise for the paper)
FORCE_REFRESH = False                  # True re-downloads every series and rebuilds cached steps
os.environ["BOND_SIM_CACHE"] = "0" if FORCE_REFRESH else "1"
import bond_sim.config as bcfg
CFG_HASH = bcfg.load().content_hash()   # every cached step is keyed by the configuration that produced it
ctx = nb.cached(f"ctx_{AS_OF.date()}_{CFG_HASH}", lambda: nb.load_context(AS_OF), refresh=FORCE_REFRESH)
print(ctx.grid, "| config hash", CFG_HASH, "| series:", ctx.w.shape[1])
"""


def notebook(name: str, cells, K: int = 1000):
    nb = nbf.v4.new_notebook()
    out = []
    for kind, src in cells:
        src = src.strip().replace("{K}", str(K))
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb["cells"] = out
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
    path = OUT / f"{name}.ipynb"
    nbf.write(nb, path)
    print("wrote", path)


# ═══════════════════════════════════════════════════════════════════════════
notebook("00_index", [
("md", """
# Bond_Sim: the doom-loop thesis, one notebook per phase

**Question.** How likely is a US sovereign-debt doom loop (debt growth pushes yields up, higher yields push interest
cost and future debt growth up), what interrupts it, who is hurt if nothing does, and for how long.

**How to work here.** Each phase notebook has a parameters cell at the top (`AS_OF`, `K`, `FORCE_REFRESH`). Change it and
rerun; every cell below recomputes from the point-in-time data. Expensive steps are memoized in `runs/notebook_cache/`
(delete it or set `FORCE_REFRESH = True` to rebuild). The code the cells call lives in `src/bond_sim/` where the tests
can reach it; the mathematics is in `docs/math.md`; every design decision and the evidence for it is in `docs/decisions/`.

| Phase | Notebook | What it establishes |
|---|---|---|
| 1 | `01_data_and_vintages` | what was known when: full ALFRED vintages, the Treasury ledger, the facts that had to be probed |
| 2 | `02_bond_book` | every Treasury tranche on one monthly grid, reconciled to what the Treasury publishes; the bottom-up effective rate |
| 3 | `03_dependence_factors_states` | nothing is independent: correlations, regimes, block factors, the latent behavioral chain |
| 4 | `04_sustainability` | the identity, not a threshold: pb*, g*, the decomposition of first/second/third differences, feasible pb, the trigger on history |
| 5 | `05_simulation` | two macro blocks on identical debt accounting; distributions, admissibility, trigger probabilities |
| 6 | `06_policies_and_shocks` | five policies x four shocks on common random numbers; act now vs act later |
| 7 | `07_who_gets_hurt_how_long` | sector employment paths and recovery half-lives |

**Standing rules.** No look-ahead (every loader takes `as_of`, the invariance test proves it). No independent parameters
(shock structure is estimated). Paths are filtered by feasibility, never clipped. Every policy and shock runs on the same
random numbers. Every artifact carries the configuration hash.
"""),
("code", """
from pathlib import Path
from IPython.display import Markdown, display
display(Markdown(Path("../PARAMETERS.md").read_text()))
"""),
("code", """
import bond_sim.config as c
cfg = c.load(); print("config hash:", cfg.content_hash())
for name in ("data", "calendar", "book", "correlation", "sim", "doomloop", "states", "labor", "policy", "recovery"):
    print(f"{name:12s}", getattr(cfg, name))
"""),
], K=1000)

# ═══════════════════════════════════════════════════════════════════════════
notebook("01_data_and_vintages", [
("md", """
# Phase 1: what was known, and when

Every macro series enters as its **full revision history** from ALFRED: one row per observation per vintage window.
"What did the world know on date X" is one boolean mask (`VintageFrame.as_of`), so no downstream number can be built on
a revision that did not yet exist. The Treasury side (auctions, the Monthly Statement of the Public Debt, interest
expense, debt to the penny) is dated by publication and filtered the same way.

Decision record: `docs/decisions/0002-full-alfred-vintages.md`.
"""),
("code", HEADER),
("md", """
## 1.1 The registry

Every series id below was resolved against the FRED API when it was added (title, frequency, units, range recorded
from the response). `pub_lag_days` is the conservative publication lag used only for observations older than the
series' first ALFRED vintage.
"""),
("code", """
from bond_sim.data import SERIES
reg = pd.DataFrame([{"id": s.id, "group": s.group, "freq": s.freq, "units": s.units, "pub_lag_days": s.pub_lag_days,
                     "title": s.title} for s in SERIES.values()]).set_index("id")
display(reg.groupby("group").size().rename("n_series").to_frame().T)
display(reg.sort_values(["group", "id"]))
"""),
("md", """
## 1.2 Revisions are large and slow

GDP for 2009Q3 was first published at 14,301.5 and revised five times. An analysis dated November 2009 must see the
first number. Below, the same series as it was known on four different dates.
"""),
("code", """
from bond_sim.data import FredClient
fc = FredClient()
gdp = fc.fetch_vintages("GDP")
display(gdp.revisions("GDP", "2009-07-01"))
fig, ax = plt.subplots(figsize=(11, 4))
for d in ("2009-11-15", "2010-11-15", "2012-11-15", str(AS_OF.date())):
    v = gdp.as_of(d).set_index("date")["value"]
    v.loc["2007":"2012"].plot(ax=ax, label=f"as known on {d}")
ax.set_title("Nominal GDP 2007-2012, four vintages"); ax.legend(); ax.set_xlabel(""); plt.show()
"""),
("md", """
## 1.3 Where true vintages start, and what is backfilled

ALFRED tracks GDP from 1991-12, daily yields from 2005-07, the monthly Treasury Statement from 2015-11. Older
observations carry the first vintage's date, which is not when they were published; those rows are re-stamped with
`date + pub_lag_days` and flagged `backfilled`. The table is the paper's data appendix.
"""),
("code", """
cov = nb.cached("vintage_coverage", lambda: fc.vintage_coverage(list(SERIES)), refresh=FORCE_REFRESH)
cov["backfilled_share"] = cov["n_backfilled_obs"] / cov["n_obs"]
display(cov[["freq", "first_obs", "first_true_vintage", "n_obs", "n_vintage_rows", "backfilled_share", "pub_lag_days_assumed"]].round(3))
ax = cov["backfilled_share"].sort_values().plot.barh(figsize=(8, 12)); ax.set_title("Share of observations before the first true vintage"); plt.show()
"""),
("md", """
## 1.4 The Treasury ledger

The auction dataset is the primitive the bond book is built from: 11,113 auctions since 1979-11 with issue and maturity
dates, coupon (notes/bonds), discount rate (bills), and accepted amounts. Two facts that had to be established by probing
rather than assumed: `total_accepted` already includes SOMA add-ons (adding `soma_accepted` overstated issuance on every
CUSIP checked), and the MSPD detail table's CUSIP column also carries subtotal labels.
"""),
("code", """
from bond_sim.data import load_auctions
au = load_auctions(as_of=AS_OF)
au["kind"] = np.where(au.security_type.eq("Bill"), "bill", np.where(au.inflation_index_security, "tips",
             np.where(au.floating_rate, "frn", np.where(au.security_type.eq("Bond"), "bond", "note"))))
au["year"] = au.issue_date.dt.year
print(f"{len(au):,} auctions, {au.cusip.nunique():,} CUSIPs, {au.issue_date.min().date()} .. {au.issue_date.max().date()}")
display(au.groupby("kind").agg(auctions=("cusip", "size"), cusips=("cusip", "nunique"),
                                 gross_issuance_tn=("total_accepted", lambda s: s.sum() / 1e12)).round(2))
gross = au.pivot_table(index="year", columns="kind", values="total_accepted", aggfunc="sum") / 1e12
ax = gross.plot.area(figsize=(11, 4), linewidth=0); ax.set_title("Gross marketable issuance by year ($tn)"); ax.set_xlabel(""); plt.show()
"""),
("md", """
## 1.5 The contract, tested

`tests/test_no_lookahead.py` builds a vintage table, snapshots an `as_of` view, appends later vintages and later
observations, and asserts the view is unchanged. `@point_in_time` re-checks every loader's output at runtime.
"""),
("code", """
from bond_sim.data.pit import VintageFrame
v_now, v_later = gdp.as_of("2009-11-15"), gdp.as_of("2009-11-15")     # same call after more data would land: identical by construction
print("rows known on 2009-11-15:", len(v_now), "| latest observation then:", v_now.date.max().date())
"""),
])

# ═══════════════════════════════════════════════════════════════════════════
notebook("02_bond_book", [
("md", """
# Phase 2: the bond book

Every auction tranche is a row; every month since 1980 is a column. The four masks from the hand-coded toy
(`main.ipynb`, Problems 1-3) are the whole engine:

    Active[i,t]      = (t >  issue[i]) & (t <= maturity[i])
    Outstanding[i,t] = (t >= issue[i]) & (t <  maturity[i])   x face
    CouponFlow[i,t]  = Active & coupon_month                  x face x coupon / 2
    Principal[i,t]   = (t == maturity[i])                      x face

computed two ways (dense masks for attribution, index-pair `bincount` for speed) that `tests/test_toy_book.py` asserts
equal. Instrument simplifications and their measured size: `docs/decisions/0004-instrument-simplifications.md`.
"""),
("code", HEADER),
("code", """
book, auctions, agg, short = nb.cached(f"book_{AS_OF.date()}_{CFG_HASH}", lambda: nb.build_book(ctx), refresh=FORCE_REFRESH)
print(f"{book.N:,} tranches on {ctx.grid}; by kind:", {k: int((book.kind == k).sum()) for k in ("bill", "note", "bond", "tips", "frn")})
hist = agg.loc[agg.index <= AS_OF]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
(hist[[f"outstanding_{k}" for k in ("bill", "note", "bond", "tips", "frn")]] / 1e6).plot.area(ax=ax[0], linewidth=0)
ax[0].set_title("Reconstructed marketable debt outstanding ($tn)"); ax[0].set_xlabel("")
(hist["interest"].rolling(12).sum() / 1e6).plot(ax=ax[1]); ax[1].set_title("Reconstructed interest, trailing 12 months ($tn)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
"""),
("md", """
## 2.1 The maturity structure

How fast the stock reprices to new yields is set by its remaining maturity and its bill share. Both are read off the
book (they feed the simulator's refinancing pool, P-11), and the next five years of maturing principal is the
"maturity wall" every debt-management discussion is about.
"""),
("code", """
prof = nb.book_profile(book, pd.date_range("1985-01-01", AS_OF, freq="YS").append(pd.DatetimeIndex([AS_OF])))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
prof["wam_months"].plot(ax=ax[0]); ax[0].set_title("Weighted average remaining maturity (months)")
(100 * prof["bill_share"]).plot(ax=ax[1]); ax[1].set_title("Bill share of the stock (%)")
fut = agg.loc[(agg.index > AS_OF) & (agg.index <= AS_OF + pd.DateOffset(years=5)), "principal"] / 1e6
fut.resample("QS").sum().plot.bar(ax=ax[2], width=0.9); ax[2].set_title("Maturing principal, next 5 years ($tn per quarter)")
ax[2].set_xticklabels([d.strftime("%Y-%m") for d in fut.resample("QS").sum().index], rotation=90, fontsize=7)
for a in ax: a.set_xlabel("")
plt.tight_layout(); plt.show()
display(prof.tail(3).round(2))
"""),
("md", """
## 2.2 Validation, not calibration

The book is compared with what the Treasury publishes and the gaps are reported, never fitted away:
* outstanding by class vs MSPD Table 1 (bills, notes, bonds, FRN within ~1.5%; TIPS at ~0.83 because the book carries
  nominal face without the inflation accrual);
* twelve-month interest vs Treasury's accrued interest expense (cash coupons vs accrual, TIPS inflation compensation);
* marketable total vs debt to the penny.
"""),
("code", """
from bond_sim.data import load_mspd_summary, load_interest_expense, load_debt_to_penny
from bond_sim.engine.validate import BookValidator
rep = nb.cached(f"validation_{AS_OF.date()}_{CFG_HASH}", lambda: BookValidator(book).run(load_mspd_summary(as_of=AS_OF),
      load_interest_expense(as_of=AS_OF), load_debt_to_penny(as_of=AS_OF), short_rate=short), refresh=FORCE_REFRESH)
display(rep.headline().round(4).to_frame("mean ratio book / published, last 5y"))
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
rep.outstanding[[c for c in rep.outstanding.columns if c.startswith("ratio_")]].plot(ax=ax[0], ylim=(0.6, 1.2)); ax[0].axhline(1, color="k", lw=0.8)
ax[0].set_title("Book / MSPD outstanding by class"); ax[0].set_xlabel("")
rep.interest["ratio"].plot(ax=ax[1], ylim=(0.5, 1.2)); ax[1].axhline(1, color="k", lw=0.8); ax[1].set_title("Book / published interest, 12m rolling"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
"""),
("md", """
## 2.3 The bottom-up effective interest rate

The identity in Phase 4 needs `r`, the rate the stock actually pays: trailing-12-month interest divided by the stock a
year earlier. It moves as low-coupon debt rolls into new issuance, which is the interest-cost channel of the loop.
Cross-check: the Treasury's own "Average Interest Rates on U.S. Treasury Securities" (Total Marketable).
"""),
("code", """
from bond_sim.data import load_avg_interest_rates
r_eff = nb.effective_rate(agg, AS_OF)
air = load_avg_interest_rates(as_of=AS_OF)
tm = air[air.security_desc.str.contains("Total Marketable", case=False, na=False)].set_index("record_date")["avg_interest_rate_amt"]
tm.index = tm.index.to_period("M").to_timestamp()
fig, ax = plt.subplots(figsize=(11, 4))
(100 * r_eff).loc["2001":].plot(ax=ax, label="book: trailing-12m interest / stock")
tm.plot(ax=ax, label="Treasury: average interest rate, total marketable")
ax.set_title("Effective interest rate on marketable debt (%)"); ax.legend(); ax.set_xlabel(""); plt.show()
print("latest:", f"book {100*r_eff.iloc[-1]:.2f}%  treasury {tm.iloc[-1]:.2f}%")
"""),
("md", """
## 2.4 Scenario discounting on the real book (Problem 5 at scale)

The present value of all future debt service under flat yield scenarios: the same `DF (K,M) @ TotalFutureCF (M,)`
collapse as the toy, now on 11,000 tranches. The simulator in Phase 5 replaces flat scenarios with simulated paths.
"""),
("code", """
from bond_sim.engine.discount import flat_scenario_factors, present_value
cf = book.total_future_cf(ctx.t0, short_rate=short)
rates = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08])
pv = present_value(flat_scenario_factors(rates, book.M, periods_per_year=12), cf) / 1e6
display(pd.DataFrame({"flat yield": rates, "PV of legacy debt service ($tn)": pv.round(2)}).set_index("flat yield"))
"""),
])

# ═══════════════════════════════════════════════════════════════════════════
notebook("03_dependence_factors_states", [
("md", """
# Phase 3: nothing is independent

Three layers, each building on the last (`docs/decisions/0003`, `0007`; `docs/math.md` 5-6):

1. **Pairwise dependence.** For every cross-group pair (fiscal x rates x macro x labor): full-sample and rolling
   correlation, lead/lag scan, Granger causality both ways with Benjamini-Hochberg control. A wide rolling range means
   one number misdescribes the pair.
2. **Block factors.** Highly correlated series act as one: each behavioral block (consumer, corporate, financial
   conditions, fiscal, labor, rates, prices) is reduced to a single interpretable factor. The residual of each series
   against its factor is the imperfect-correlation shock.
3. **Latent states.** A Markov chain on the factor vector, estimated as a hidden Markov model, with within-state
   dynamics and bootstrapped residuals. Outcomes become mixtures over states, not normals.
"""),
("code", HEADER),
("md", "## 3.1 Pairwise dependence on the thesis pairs"),
("code", """
from bond_sim.analysis import CorrelationAnalyzer, RegimeDetector
from bond_sim.cli import THESIS_GROUPS, _quarterly_panel
ids = [s for g in THESIS_GROUPS.values() for s in g]
group_of = {s: g for g, ss in THESIS_GROUPS.items() for s in ss}
wq = _quarterly_panel(ctx.w[[c for c in ids if c in ctx.w.columns]], ids)
an = CorrelationAnalyzer(wq, ctx.cfg.correlation, freq="Q")
rep = nb.cached(f"corr_report_{AS_OF.date()}_{CFG_HASH}", lambda: an.report(with_granger=True, group_of=group_of), refresh=FORCE_REFRESH)
display(rep.head(15)[["a", "b", "n", "full_corr", "rolling_min", "rolling_max", "best_lag", "leads", "granger_a_to_b_sig", "granger_b_to_a_sig"]].round(3))
print(f"{len(rep)} cross-group pairs; regime-dependent (rolling range > 0.5): {int(rep.regime_dependent.sum())}")
"""),
("code", """
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for row, pair in zip(axes, [("FYGFGDQ188S", "DGS10"), ("FYGFGDQ188S", "THREEFYTP10")]):
    r = an.analyze_pair(*pair, with_granger=False)
    r.rolling.plot(ax=row[0]); row[0].axhline(0, color="k", lw=0.8); row[0].set_title(f"20q rolling corr: d({pair[0]}) vs d({pair[1]})"); row[0].set_xlabel("")
    r.lead_lag.plot.bar(ax=row[1]); row[1].set_title(f"corr(d{pair[0]}_t, d{pair[1]}_(t+lag)); +lag = debt leads")
plt.tight_layout(); plt.show()
det = RegimeDetector(ctx.cfg.correlation)
top = rep[rep.regime_dependent].head(12)
labels = {f"{x.a}|{x.b}": det.label(an.rolling(x.a, x.b).dropna()) for x in top.itertuples()}
cons = det.consistency(labels); off = cons.to_numpy()[~np.eye(len(cons), dtype=bool)]
print(f"regime timing consistency across the 12 widest pairs, mean adjusted Rand index: {np.nanmean(off):.3f}  (near 0: pair-specific; near 1: one market regime)")
"""),
("md", """
## 3.2 Block factors

One factor per block, sign-aligned so that high means expansionary / loose / strong / healthy. The loadings say which
series carry the block; the explained variance says how much of the block is one thing. Sample: the common overlap of all
seven blocks (VIX and the SLOOS series begin 1990).
"""),
("code", """
bf, chain = nb.cached(f"factors_chain_{AS_OF.date()}_{CFG_HASH}", lambda: nb.fit_factors_and_chain(ctx), refresh=FORCE_REFRESH)
display(bf.summary())
load = pd.DataFrame({b: {c: round(float(l), 2) for c, l in zip(f.columns, f.loadings)} for b, f in bf.fits.items()})
display(load.fillna("").T)
F = bf.factors()
fig, axes = plt.subplots(len(F.columns), 1, figsize=(13, 2.0 * len(F.columns)), sharex=True)
for ax, b in zip(axes, F.columns):
    F[b].plot(ax=ax, lw=0.9); ax.axhline(0, color="k", lw=0.5); ax.set_ylabel(b)
    for t0, t1 in [("1990-07-01", "1991-03-01"), ("2001-03-01", "2001-11-01"), ("2007-12-01", "2009-06-01"), ("2020-02-01", "2020-04-01")]:
        ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color="grey", alpha=0.15)
axes[0].set_title("Block factors (grey: NBER recessions)"); plt.tight_layout(); plt.show()
display(F.corr().round(2))
"""),
("md", """
## 3.3 The latent chain

A Gaussian HMM on the factor vector; the state count by BIC over 2-4; a sticky Dirichlet prior on the transition
matrix (sparse regime data, the RateWalk lesson); state 0 is always the weakest by construction. Within each state a
common VAR(1) with state-specific intercepts gives the dynamics; residuals are banked by state for bootstrapping.
The stress hazard is logistic in the financial-conditions factor: the doom-loop channel in state space.
"""),
("code", """
display(pd.DataFrame(chain.selection).T.rename_axis("candidate states"))
print(f"chosen: {chain.k} states (lowest BIC among candidates whose every state has >= {chain.min_state_occupancy} quarters; factors winsorized at +/-{chain.winsor_sd} sd for estimation)")
print(f"hazard beta (stress entry on the financial factor): {chain.hazard_beta:.3f} (unrestricted estimate {chain.hazard_beta_raw:.3f}; negative estimates disable the channel)   max eigenvalue of A: {np.abs(np.linalg.eigvals(chain.A)).max():.3f}")
display(pd.DataFrame(chain.P, index=[f"from {s}" for s in range(chain.k)], columns=[f"to {s}" for s in range(chain.k)]).round(3))
display(pd.DataFrame(chain.means, columns=chain.blocks, index=[f"state {s}" for s in range(chain.k)]).round(2))
fig, ax = plt.subplots(figsize=(13, 3))
chain.posterior.plot.area(ax=ax, linewidth=0, alpha=0.8); ax.set_title("Smoothed state probabilities"); ax.set_xlabel(""); plt.show()
print("state counts:", np.bincount(chain.states, minlength=chain.k).tolist())
print("quarters in state 0:", [d.strftime("%Y-%m") for d in chain.states[chain.states == 0].index])
"""),
("md", """
## 3.4 Why not a normal distribution

The residuals the simulator bootstraps from are not Gaussian: skewness and excess kurtosis by state, against zero for
a normal. This is the empirical basis for drawing from history instead of from a fitted family.
"""),
("code", """
from scipy import stats
rows = []
for s, bank in chain.resid_bank.items():
    for j, b in enumerate(chain.blocks):
        rows.append({"state": s, "block": b, "n": len(bank), "skew": stats.skew(bank[:, j]), "excess_kurtosis": stats.kurtosis(bank[:, j])})
display(pd.DataFrame(rows).pivot(index="block", columns="state", values=["skew", "excess_kurtosis"]).round(2))
draws = chain.draw(2000, 120, np.random.default_rng(0))
sim = chain.simulate(2000, 120, draws, bf.last_factors(), chain.initial_state_probs())
occ = chain.occupancy(sim["S"])
ax = occ.plot.area(figsize=(11, 3), linewidth=0, alpha=0.8); ax.set_title("Unconditional state occupancy over 30 years, 2000 paths"); ax.set_xlabel("quarter"); plt.show()
"""),
])

# ═══════════════════════════════════════════════════════════════════════════
notebook("04_sustainability", [
("md", """
# Phase 4: the identity, not a threshold

The ratio $d$ obeys $\\Delta d_t = d_{t-1}\\frac{r_t-g_t}{1+g_t} - pb_t + sf_t$ (snowball, primary, stock-flow residual).
Setting $\\Delta d = 0$ gives the debt-stabilizing primary balance $pb^*$ and the debt-stabilizing growth rate
$g^* = (d\\,r - pb)/(d + pb)$. A path is in the doom loop when $pb^*$ exceeds what is feasible for N consecutive
periods. No debt level is chosen. Derivations: `docs/math.md` 1-4; decision `docs/decisions/0006`.

$r$ is the bottom-up effective rate from Phase 2. Second differences are decomposed to say what is accelerating the
ratio; third differences are tested as a leading indicator only.
"""),
("code", HEADER),
("code", """
book, auctions, agg, short = nb.cached(f"book_{AS_OF.date()}_{CFG_HASH}", lambda: nb.build_book(ctx), refresh=FORCE_REFRESH)
from bond_sim.sim import decompose, stabilizing_growth, stabilizing_primary_balance, FeasiblePB, evaluate_history, debt_limit
from bond_sim.sim.premium import LinearPremium, ThresholdPremium, NoPremium, premium_from_config
h = nb.history_frame(ctx, agg)
dec = decompose(h["d"], h["r_eff"], h["g_nom"], h["pb"])
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
(100 * dec[["snowball", "primary", "residual"]]).plot.bar(ax=ax[0], stacked=True, width=1.0, linewidth=0)
ax[0].plot(range(len(dec)), 100 * dec["delta_d"], color="k", lw=1.2, label="delta d"); ax[0].legend(fontsize=8)
ax[0].set_title("Quarterly change in debt/GDP (pct pts): snowball vs primary vs stock-flow residual"); ax[0].set_xticks([])
(100 * dec["r_minus_g"]).plot(ax=ax[1]); ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("r minus g (pct pts, annualized)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
"""),
("md", """
## 4.1 What would it take: $pb^*$ and $g^*$ against what actually happened
"""),
("code", """
d_lag = h["d"].shift(1)
h["pb_star"] = stabilizing_primary_balance(d_lag, h["r_eff"], h["g_nom"])
h["g_star"] = stabilizing_growth(d_lag, h["r_eff"], h["pb"])
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
(100 * h[["pb", "pb_star"]]).plot(ax=ax[0]); ax[0].set_title("Primary balance: actual vs debt-stabilizing (% GDP)"); ax[0].set_xlabel("")
(100 * h[["g_nom", "g_star"]]).rolling(4).mean().plot(ax=ax[1]); ax[1].set_title("Nominal growth: actual vs debt-stabilizing (4q avg, %)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
print(f"latest: pb {100*h.pb.iloc[-1]:.2f}%  pb* {100*h.pb_star.iloc[-1]:.2f}%  |  g {100*h.g_nom.iloc[-4:].mean():.2f}%  g* {100*h.g_star.iloc[-1]:.2f}%  |  r_eff {100*h.r_eff.iloc[-1]:.2f}%  d {100*h.d.iloc[-1]:.0f}%")
"""),
("md", """
## 4.2 What is feasible

Two benchmarks from the data: the historical envelope (maximum observed primary balance) and a fiscal reaction function
with fatigue (cubic in lagged debt, unemployment as the cyclical control). A negative cubic term means the response
weakens at high debt. With a premium that makes $r$ increase in $d$, the reaction function implies a debt limit.
"""),
("code", """
feas = FeasiblePB.from_history(h["pb"], h["d"], h["u"], quantile=ctx.cfg.doomloop.feasible_quantile)
fr = feas.reaction
print(f"envelope (max primary balance): {100*feas.envelope:.2f}% of GDP")
print(f"reaction function: pb = {fr.coef[0]:.4f} + {fr.coef[1]:.4f} d + {fr.coef[2]:.4f} d^2 + {fr.coef[3]:.4f} d^3 + {fr.coef[4]:.4f} (u - mean)   R2 {fr.r2:.2f}, n {fr.n}, fatigue: {fr.fatigue}")
grid_d = np.linspace(0.3, 2.5, 100)
r_now, g_now = float(h.r_eff.iloc[-1]), float(h.g_nom.iloc[-8:].mean())
prem_models = [("no premium", NoPremium()), ("linear premium (P-01 placeholder)", premium_from_config(ctx.cfg.doomloop.risk_premium)),
               ("threshold premium (P-02 placeholder)", ThresholdPremium(ctx.cfg.doomloop.risk_premium.bps_per_pct_debt_gdp, ctx.cfg.doomloop.risk_premium.anchor_debt_gdp_pct,
                                                                         ctx.cfg.doomloop.risk_premium.threshold_debt_gdp_pct, ctx.cfg.doomloop.risk_premium.threshold_extra_bps_per_pct))]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(100 * grid_d, 100 * feas.evaluate(grid_d, which="reaction"), label="feasible pb (reaction function)")
ax.axhline(100 * feas.envelope, color="grey", ls="--", label=f"feasible pb (envelope, q{ctx.cfg.doomloop.feasible_quantile})")
for name, prem in prem_models:
    r_of_d = lambda x, p=prem: r_now + float(p.premium(np.array([100 * x]))[0]) / 100.0
    pbs = 100 * stabilizing_primary_balance(grid_d, np.array([r_of_d(x) for x in grid_d]), np.full_like(grid_d, g_now))
    ax.plot(100 * grid_d, pbs, label=f"pb* needed, {name}")
ax.set_xlabel("debt / GDP (%)"); ax.set_ylabel("% of GDP"); ax.set_ylim(-6, 8); ax.legend(fontsize=8)
ax.set_title(f"Required vs feasible primary balance at today's r_eff {100*r_now:.2f}% and recent g {100*g_now:.2f}%"); plt.show()
"""),
("md", """
**The debt limit depends on r minus g, not on debt alone.** With r below g (today: effective rate 3.3% against
nominal growth above 5%), every debt level is stabilizable by a modest deficit and the reaction function implies no
meaningful limit. The limit only appears once the base rate exceeds growth, so it is reported as a function of an
assumed r minus g spread (0, +1, +2 points on top of today's growth) and of the premium model, against both feasibility
benchmarks. That table is the paper's answer to "how much room is there", conditional on the one number nobody knows.
"""),
("code", """
rows = []
for rg in (0.0, 0.01, 0.02):
    for name, prem in prem_models:
        r_of_d = lambda x, p=prem, rg=rg: g_now + rg + float(p.premium(np.array([100 * x]))[0]) / 100.0
        for which in ("actual", "envelope", "reaction"):
            feas_w = FeasiblePB(envelope=feas.envelope, reaction=feas.reaction if which == "reaction" else None)
            if which == "envelope":
                pbs = stabilizing_primary_balance(grid_d, np.array([r_of_d(x) for x in grid_d]), np.full_like(grid_d, g_now))
                over = np.nonzero(pbs > feas.envelope)[0]
                lim = float(grid_d[over[0]]) if len(over) else None
            else:
                lim = debt_limit(feas_w, r_of_d, g_now)
            rows.append({"r minus g (pct pts)": 100 * rg, "premium": name, "benchmark": which,
                         "debt limit (% GDP)": None if lim is None else round(100 * lim)})
display(pd.DataFrame(rows).pivot(index=["r minus g (pct pts)", "premium"], columns="benchmark", values="debt limit (% GDP)"))
print(f"today: debt/GDP {100*h.d.iloc[-1]:.0f}%, r_eff {100*r_now:.2f}%, recent nominal growth {100*g_now:.2f}% (r minus g = {100*(r_now-g_now):+.2f} pts)")
"""),
("md", """
## 4.3 The trigger on history

Which quarters since 1980 would have flagged, under each persistence window and benchmark? A definition that flags
nothing is useless; one that flags every quarter is worse. This is the calibration exercise for N and the quantile,
the only two free numbers.
"""),
("code", """
rows = []
for quantile in (1.0, 0.9, 0.75):
    feas_q = FeasiblePB.from_history(h["pb"], h["d"], h["u"], quantile=quantile, fit_reaction=False)
    for require in (True, False):
        for n in (2, 4, 8):
            H = evaluate_history(h, feas_q, n_quarters=n, which="envelope", require_r_gt_g=require)
            flagged = H.index[H["triggered"].fillna(False)]
            rows.append({"envelope quantile": quantile, "feasible pb %GDP": round(100 * feas_q.envelope, 2), "require r>g": require, "N quarters": n,
                         "breach quarters": int(H["breach"].sum()), "triggered quarters": len(flagged),
                         "first": flagged.min().date() if len(flagged) else None, "last": flagged.max().date() if len(flagged) else None})
for require in (True, False):
    H = evaluate_history(h, feas, n_quarters=4, which="reaction", require_r_gt_g=require)
    flagged = H.index[H["triggered"].fillna(False)]
    rows.append({"envelope quantile": "reaction fn", "feasible pb %GDP": None, "require r>g": require, "N quarters": 4,
                 "breach quarters": int(H["breach"].sum()), "triggered quarters": len(flagged),
                 "first": flagged.min().date() if len(flagged) else None, "last": flagged.max().date() if len(flagged) else None})
display(pd.DataFrame(rows))
H = evaluate_history(h, feas, n_quarters=4, which=ctx.cfg.doomloop.feasible_benchmark, require_r_gt_g=ctx.cfg.doomloop.trigger_require_r_gt_g)
fig, ax = plt.subplots(figsize=(13, 4))
(100 * H["gap"]).plot(ax=ax, label="pb* minus feasible pb (pct pts of GDP)"); ax.axhline(0, color="k", lw=0.8)
for d in H.index[H["triggered"].fillna(False)]:
    ax.axvspan(d, d + pd.DateOffset(months=3), color="red", alpha=0.15)
ax.set_title("Sustainability gap on history (red: triggered under N=4)"); ax.legend(); ax.set_xlabel(""); plt.show()
"""),
("md", """
## 4.4 Second and third differences

Acceleration decomposed into the snowball and the primary balance, and the third difference as a candidate early warning:
how often was it positive in the two years before a breach began, versus in general?
"""),
("code", """
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
(100 * dec[["delta2_snowball", "delta2_primary"]]).rolling(4).mean().plot(ax=ax[0]); ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_title("Acceleration of debt/GDP: snowball vs primary (4q avg)"); ax[0].set_xlabel("")
(100 * dec["delta3_d"]).rolling(4).mean().plot(ax=ax[1]); ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("Third difference (4q avg)"); ax[1].set_xlabel("")
plt.tight_layout(); plt.show()
breach_start = H["breach"] & ~H["breach"].shift(1, fill_value=False)
pre = pd.Series(False, index=H.index)
for d in H.index[breach_start]:
    pre.loc[(H.index < d) & (H.index >= d - pd.DateOffset(years=2))] = True
d3 = dec["delta3_d"].reindex(H.index)
print(f"share of quarters with positive third difference: overall {float((d3 > 0).mean()):.2f}, in the 2 years before a breach begins {float((d3[pre] > 0).mean()):.2f}  (n pre-breach quarters: {int(pre.sum())})")
"""),
])

# ═══════════════════════════════════════════════════════════════════════════
notebook("05_simulation", [
("md", """
# Phase 5: the simulation

Two macro blocks run on identical monthly debt accounting and identical random numbers:

* **VAR block**: the estimated levels VAR (Gaussian shocks), the transparent benchmark (`docs/decisions/0005`).
* **State block**: block factors and the latent chain with bootstrapped residuals (`0007`), the thesis's own construction.

The fiscal premium is added on top of the base 10y from last quarter's debt/GDP and transmitted to the real economy
each block's way. Paths are filtered by feasibility, never clipped (every rejection is reported). The trigger is the
identity-based one from Phase 4; the old threshold is shown only for comparison.
"""),
("code", HEADER),
("code", """
from bond_sim.sim import NoPremium, LinearPremium, evaluate_paths
book, auctions, agg, short = nb.cached(f"book_{AS_OF.date()}_{CFG_HASH}", lambda: nb.build_book(ctx), refresh=FORCE_REFRESH)
bf, chain = nb.cached(f"factors_chain_{AS_OF.date()}_{CFG_HASH}", lambda: nb.fit_factors_and_chain(ctx), refresh=FORCE_REFRESH)
var = nb.fit_var(ctx)
init = nb.initial_state(ctx, book, auctions)
H_, feas, h = nb.history_sustainability(ctx, agg)
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in init.__dict__.items() if not isinstance(v, np.ndarray)})
print(f"VAR({var.lag}) long-run means: {var.long_run_means().round(2).to_dict()}   |   chain: {chain.describe()}")
print("premium:", nb.premium_from_config(ctx.cfg.doomloop.risk_premium).describe(), "  feasible pb envelope:", f"{100*feas.envelope:.2f}%")
"""),
("md", "## 5.1 Status quo under both blocks"),
("code", """
runs = {}
for name in ("VAR", "states"):
    factory = (lambda shocks, n=name: nb.make_blocks(ctx, var, bf, chain, shocks)[n])
    runs[name] = nb.run_scenarios(ctx, init, feas, factory, ["status_quo"], K)[("none", "status_quo")]
summ = pd.DataFrame({n: r.summary() for n, r in runs.items()}).T
display(summ[["block", "admissible_share", "P(trigger)", "P(trigger_threshold)", "median_years_to_trigger", "median_end_debt_gdp", "median_end_r10", "median_end_u", "median_end_g_nom", "median_end_pb", "median_end_interest_12m_gdp"]].round(3))
display(pd.DataFrame({n: r.rejections for n, r in runs.items()}))
"""),
("code", """
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, v, title in zip(axes.ravel(), ("debt_gdp", "r10", "u", "interest_12m_gdp", "pb", "g_nom"),
                        ("Debt / GDP (%)", "10y yield incl. premium (%)", "Unemployment (%)", "Interest / GDP, trailing 12m (%)", "Primary balance (% GDP)", "Nominal growth (%)")):
    for n, r in runs.items():
        nb.fan(ax, r, v, n)
    ax.set_title(f"{title}: median and 5-95% band"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
("md", """
## 5.2 The trigger: probability, timing, and what growth would be needed

`P(trigger)` is the share of admissible paths on which the debt-stabilizing primary balance exceeds the feasible one for
12 consecutive months. The threshold version (150% debt/GDP or 30% interest/receipts) is a different, cruder question.
"""),
("code", """
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for n, r in runs.items():
    ttt = r.time_to_trigger_years()
    ax[0].hist(ttt, bins=30, alpha=0.5, label=f"{n}: P={r.trigger_probability():.2f}")
    gs = r.sustainability
    ax[1].plot(r.dates, 100 * np.nanmedian(gs.g_star[r.admissible], axis=0), label=f"{n}: median g* (needed)")
    ax[1].plot(r.dates, np.nanmedian(r.paths["g_nom"][r.admissible], axis=0), ls="--", label=f"{n}: median g (simulated)")
    ax[2].plot(r.dates, 100 * np.nanmedian(gs.gap[r.admissible], axis=0), label=n)
ax[0].set_title("Years to trigger (identity), triggered paths"); ax[0].legend(fontsize=8)
ax[1].set_title("Debt-stabilizing vs simulated nominal growth (%)"); ax[1].legend(fontsize=7)
ax[2].axhline(0, color="k", lw=0.8); ax[2].set_title("Median gap: pb* minus feasible pb (pct pts GDP)"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""),
("md", """
## 5.3 Shape of the outcome distribution

The state block exists to produce the tails history has. Terminal debt/GDP: skewness and excess kurtosis under each
block (a normal has zero of both), plus state occupancy over the horizon.
"""),
("code", """
display(pd.DataFrame({n: nb.shape_stats(r, "debt_gdp") for n, r in runs.items()}).round(2))
r = runs["states"]
occ = chain.occupancy(r.paths["state"][r.admissible][:, 2::3].astype(int))
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
occ.plot.area(ax=ax[0], linewidth=0, alpha=0.8); ax[0].set_title("State occupancy along the simulation (admissible paths)"); ax[0].set_xlabel("quarter")
for n, rr in runs.items():
    x = rr.paths["debt_gdp"][rr.admissible, -1]; ax[1].hist(x[np.isfinite(x)], bins=60, alpha=0.5, density=True, label=n)
ax[1].set_title("Terminal debt/GDP (%), density"); ax[1].legend(); plt.tight_layout(); plt.show()
"""),
("md", """
## 5.4 The two placeholders the state block's loop rests on

The state block's doom loop runs through P-01 (premium slope, bps of 10y per point of debt/GDP) and P-16 (how much a
point of premium tightens financial conditions, default: like a point of Baa spread). Neither is estimated here. The
grid below reruns the status quo on the same draws across both, so the reader sees how much of the probability is the
mechanism and how much is the placeholder. The VAR block only depends on P-01.
"""),
("code", """
from bond_sim.sim import LinearPremium, StateBlock
rows = []
HQ = (len(init.legacy_interest_mm) + 2) // 3
draws_s = StateBlock(bf, chain).prepare(K, HQ, np.random.default_rng(ctx.cfg.sim.seed))
k_default = StateBlock(bf, chain).k_prem
for slope in (0.0, 1.0, 2.0, 5.0):
    for scale in (0.0, 0.25, 0.5, 1.0):
        blk = StateBlock(bf, chain, premium_to_financial=k_default * scale)
        r = nb.DoomLoopSimulator(ctx.cfg, blk, LinearPremium(slope, ctx.cfg.doomloop.risk_premium.anchor_debt_gdp_pct), init,
                                 nb.policy_from_name("status_quo", ctx.cfg.policy), feasible=feas).run(K=K, draws=draws_s, start=AS_OF + pd.DateOffset(months=1))
        ok = r.admissible
        rows.append({"premium bps/pt (P-01)": slope, "premium->conditions x default (P-16)": scale, "admissible": float(ok.mean()),
                     "P(trigger)": r.trigger_probability(), "median_end_debt_gdp": float(np.median(r.paths["debt_gdp"][ok, -1])),
                     "share_in_stress_state_2056": float((r.paths["state"][ok, -1] == 0).mean())})
grid = pd.DataFrame(rows)
display(grid.pivot(index="premium bps/pt (P-01)", columns="premium->conditions x default (P-16)", values="P(trigger)").round(3))
display(grid.pivot(index="premium bps/pt (P-01)", columns="premium->conditions x default (P-16)", values="share_in_stress_state_2056").round(3))
print(f"default premium->conditions mapping: {k_default:.2f} financial-factor sd per pct point of premium")
"""),
("md", """
## 5.5 Sensitivity of the trigger to its own free numbers

Persistence N, the feasible benchmark, and the r > g requirement, re-evaluated on the same simulated paths (no
re-simulation needed).
"""),
("code", """
rows = []
for n, r in runs.items():
    ok = r.admissible
    for which in ("actual", "envelope", "reaction"):
        for require in (True, False):
            for months in (6, 12, 24):
                diag = evaluate_paths(r.paths["debt_gdp"][ok] / 100, r.paths["r_eff"][ok] / 100, r.paths["g_nom"][ok] / 100, r.paths["pb"][ok] / 100,
                                      feas, months, which=which, u=r.paths["u"][ok], require_r_gt_g=require)
                rows.append({"block": n, "benchmark": which, "require r>g": require, "persistence_months": months, "P(trigger)": diag.probability(), **diag.lead_indicator_test()})
display(pd.DataFrame(rows).round(3))
"""),
], K=1000)

# ═══════════════════════════════════════════════════════════════════════════
notebook("06_policies_and_shocks", [
("md", """
# Phase 6: policies and shocks on common random numbers

Every policy is a set of hooks on the same simulation; every shock is a dated intervention on the same draws. A
difference between two outcome distributions is therefore the intervention, not the dice. All four shocks are run and
none is privileged; paths are filtered only by feasibility.

Every magnitude inside the policies is a placeholder (P-06 to P-09). The shocks are one-standard-deviation-scale
impulses on the block factors or forced transitions into the stress state; their sizes are stated in the cell.
"""),
("code", HEADER),
("code", """
from bond_sim.sim import ForcedTransition, FactorImpulse, StateBlock
book, auctions, agg, short = nb.cached(f"book_{AS_OF.date()}_{CFG_HASH}", lambda: nb.build_book(ctx), refresh=FORCE_REFRESH)
bf, chain = nb.cached(f"factors_chain_{AS_OF.date()}_{CFG_HASH}", lambda: nb.fit_factors_and_chain(ctx), refresh=FORCE_REFRESH)
var = nb.fit_var(ctx); init = nb.initial_state(ctx, book, auctions); H_, feas, h = nb.history_sustainability(ctx, agg)
SHOCKS = {
    "none": None,
    "recession_now": [ForcedTransition(state=0, at=0, duration=4, label="forced into the stress state for 4 quarters")],
    "rate_spike": [FactorImpulse("rates", +2.0, at=0, decay=0.8), FactorImpulse("financial", -1.5, at=0, decay=0.7)],
    "foreign_withdrawal": [FactorImpulse("fiscal", -1.5, at=0, decay=0.85), FactorImpulse("financial", -1.0, at=0, decay=0.8)],
    "consumer_retrenchment": [FactorImpulse("consumer", -2.0, at=0, decay=0.8)],
}
POLICIES = list(ctx.cfg.policy.menu)
states_factory = lambda shocks: nb.make_blocks(ctx, var, bf, chain, shocks)["states"]
runs = nb.cached(f"runs_policies_shocks_{AS_OF.date()}_{CFG_HASH}_K{K}", lambda: nb.run_scenarios(ctx, init, feas, states_factory, POLICIES, K, SHOCKS), refresh=FORCE_REFRESH)
tab = nb.summary_table(runs)
display(tab.round(3))
"""),
("code", """
piv = tab.pivot(index="policy", columns="shock", values="P(trigger)")[list(SHOCKS)]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
im = ax[0].imshow(piv.to_numpy(), cmap="Reds", vmin=0, vmax=1); ax[0].set_xticks(range(len(piv.columns))); ax[0].set_xticklabels(piv.columns, rotation=30, ha="right")
ax[0].set_yticks(range(len(piv.index))); ax[0].set_yticklabels(piv.index); ax[0].set_title("P(trigger) by policy x shock")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax[0].text(j, i, f"{piv.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
piv2 = tab.pivot(index="policy", columns="shock", values="median_end_debt_gdp")[list(SHOCKS)]
piv2.plot.bar(ax=ax[1]); ax[1].set_title("Median 2056 debt/GDP (%)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
"""),
("md", "## 6.1 Fan charts for the recession shock across policies"),
("code", """
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, v, title in zip(axes, ("debt_gdp", "u", "interest_12m_gdp"), ("Debt / GDP (%)", "Unemployment (%)", "Interest / GDP (%)")):
    for p in POLICIES:
        nb.fan(ax, runs[("recession_now", p)], v, p, band=(p == "status_quo"))
    ax.set_title(f"{title}, recession-now shock"); ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
"""),
("md", """
## 6.2 Act now or act later

The same policy started today, in five years, and in ten years, on the same draws (no shock). The gap between the
curves is the cost of waiting under the placeholder magnitudes.
"""),
("code", """
delayed = {}
for start_q in (0, 20, 40):
    delayed[start_q] = nb.run_scenarios(ctx, init, feas, states_factory, ["austerity", "growth_led"], K, {"none": None}, policy_start_quarter=start_q)
rows = []
for start_q, rr in delayed.items():
    for (s, p), r in rr.items():
        rows.append({"policy": p, "start_year": start_q / 4, "P(trigger)": r.trigger_probability(), "median_end_debt_gdp": float(np.median(r.paths["debt_gdp"][r.admissible, -1])),
                     "median_end_u": float(np.median(r.paths["u"][r.admissible, -1]))})
display(pd.DataFrame(rows).pivot(index="policy", columns="start_year").round(3))
fig, ax = plt.subplots(figsize=(11, 4))
for start_q, rr in delayed.items():
    nb.fan(ax, rr[("none", "austerity")], "debt_gdp", f"austerity from year {start_q // 4}", band=False)
nb.fan(ax, runs[("none", "status_quo")], "debt_gdp", "status quo", band=True)
ax.set_title("Debt / GDP median: austerity started now vs later"); ax.legend(); plt.show()
"""),
("md", "## 6.3 The VAR block as a cross-check (policies, no shocks)"),
("code", """
var_factory = lambda shocks: nb.make_blocks(ctx, var, bf, chain, shocks)["VAR"]
runs_var = nb.cached(f"runs_policies_var_{AS_OF.date()}_{CFG_HASH}_K{K}", lambda: nb.run_scenarios(ctx, init, feas, var_factory, POLICIES, K, {"none": None}), refresh=FORCE_REFRESH)
cmp = pd.concat([nb.summary_table(runs_var).assign(block="VAR"), nb.summary_table({k: v for k, v in runs.items() if k[0] == "none"}).assign(block="states")])
display(cmp.set_index(["block", "policy"])[["admissible_share", "P(trigger)", "median_years_to_trigger", "median_end_debt_gdp", "median_end_r10", "median_end_u"]].round(3))
"""),
], K=1000)

# ═══════════════════════════════════════════════════════════════════════════
notebook("07_who_gets_hurt_how_long", [
("md", """
# Phase 7: who gets hurt, and for how long

Sector employment growth is regressed on the unemployment change and lagged 10y changes (quarterly, from history), with
the sectors' residual covariance kept so sector shocks are drawn together. Recovery half-lives are measured on five US
episodes and on every simulated path (months from the unemployment peak until half the gap is closed).
"""),
("code", HEADER),
("code", """
from bond_sim.sim import LaborModel, RecoveryModel, ForcedTransition, FactorImpulse
book, auctions, agg, short = nb.cached(f"book_{AS_OF.date()}_{CFG_HASH}", lambda: nb.build_book(ctx), refresh=FORCE_REFRESH)
bf, chain = nb.cached(f"factors_chain_{AS_OF.date()}_{CFG_HASH}", lambda: nb.fit_factors_and_chain(ctx), refresh=FORCE_REFRESH)
var = nb.fit_var(ctx); init = nb.initial_state(ctx, book, auctions); H_, feas, h = nb.history_sustainability(ctx, agg)
labor = LaborModel(list(ctx.cfg.labor.sectors)).fit(ctx.w, ctx.qs)
display(labor.table().round(3))
rec = RecoveryModel()
display(rec.historical(ctx.w["UNRATE"].dropna())[["peak_date", "peak_u", "trough_u", "half_life_months"]])
"""),
("code", """
SHOCKS = {"none": None, "recession_now": [ForcedTransition(0, 0, 4)], "rate_spike": [FactorImpulse("rates", 2.0, 0, 0.8), FactorImpulse("financial", -1.5, 0, 0.7)],
          "consumer_retrenchment": [FactorImpulse("consumer", -2.0, 0, 0.8)]}
states_factory = lambda shocks: nb.make_blocks(ctx, var, bf, chain, shocks)["states"]
runs = nb.cached(f"runs_labor_{AS_OF.date()}_{CFG_HASH}_K{K}", lambda: nb.run_scenarios(ctx, init, feas, states_factory, ["status_quo", "no_layoff_mandate", "austerity"], K, SHOCKS), refresh=FORCE_REFRESH)
rows, hl_rows = [], []
sector_paths = {}
for (s, p), r in runs.items():
    ok = r.admissible
    d_u = nb.quarterly_changes(r.paths["u"][ok], init.u_pct); d_r = nb.quarterly_changes(r.paths["r10"][ok], init.r10_pct)
    sector_paths[(s, p)] = (ok, labor.simulate(d_u, d_r, np.random.default_rng(ctx.cfg.sim.seed + 1)))
    hl = rec.simulated(r.paths["u"][ok], init.u_pct)
    hl_rows.append({"shock": s, "policy": p, "share_with_downturn": float(hl.notna().mean()), "median_half_life_months": float(hl.median()),
                    "p75_half_life_months": float(hl.quantile(0.75)), "median_peak_u": float(np.median(r.paths["u"][ok].max(axis=1)))})
# Sector damage is measured against the no-shock status-quo path on the same draws (common random numbers) at the same
# date, so secular trends (manufacturing, autos, federal employment all trend down over 30 years) cancel and what is
# left is the cyclical loss the shock or policy caused. Reported at its worst point within the first ten years.
ok0, base = sector_paths[("none", "status_quo")]
for (s, p), (ok, paths) in sector_paths.items():
    common = ok & ok0
    for sec_, path in paths.items():
        rel = 100.0 * (path[common[ok]] / base[sec_][common[ok0]] - 1.0)
        worst = rel[:, :40].min(axis=1)
        rows.append({"shock": s, "policy": p, "sector": sec_, "median_worst_gap_pct": float(np.median(worst)), "p05_worst_gap_pct": float(np.quantile(worst, 0.05))})
sec = pd.DataFrame(rows); hl_tab = pd.DataFrame(hl_rows)
print("Employment shortfall vs the no-shock status-quo path, worst point in the first 10 years (%), status quo by shock:")
display(sec[sec.policy == "status_quo"].pivot(index="sector", columns="shock", values="median_worst_gap_pct").round(1))
display(hl_tab.round(2))
"""),
("code", """
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
sec[(sec.policy == "status_quo") & (sec["shock"] != "none")].pivot(index="sector", columns="shock", values="median_worst_gap_pct").plot.bar(ax=ax[0])
ax[0].set_title("Median employment shortfall vs no-shock path (%), status quo, by shock"); ax[0].legend(fontsize=7)
sec[(sec["shock"] == "recession_now")].pivot(index="sector", columns="policy", values="median_worst_gap_pct").plot.bar(ax=ax[1])
ax[1].set_title("Median employment shortfall vs no-shock status quo (%), recession-now shock, by policy"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
fig, ax = plt.subplots(figsize=(11, 4))
for (s, p), r in runs.items():
    if p == "status_quo":
        hl = rec.simulated(r.paths["u"][r.admissible], init.u_pct).dropna()
        ax.hist(hl, bins=40, alpha=0.45, label=f"{s} (n={len(hl)})")
ax.set_title("Simulated recovery half-lives (months from unemployment peak), status quo, by shock"); ax.legend(fontsize=8); plt.show()
"""),
("md", """
## What is still placeholder

The no-layoff mandate's floor and growth penalty (P-06), the austerity multiplier (P-07), the monetization inflation cost
(P-08), the growth uplift (P-09), the premium slope and form (P-01, P-02), and the premium-to-conditions mapping (P-16)
all still carry placeholder values. The sector betas are estimated but reduced-form (rates rise in booms, so the rate
coefficients partly reflect that); the unemployment channel carries the cyclical effect.
"""),
], K=1000)
