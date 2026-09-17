"""Debt-sustainability diagnostics from the accounting identity (docs/math.md 1-4).

Replaces the hand-picked debt/GDP trigger (old P-04) with quantities the
identity defines:

    delta_d   = snowball + primary + residual        (first order, decomposed)
    pb_star   = d_{t-1} (r - g) / (1 + g)            (debt-stabilizing primary balance)
    g_star    = (d_{t-1} r - pb) / (d_{t-1} + pb)    (debt-stabilizing nominal growth)
    trigger   = pb_star > feasible pb for N consecutive quarters

Feasible pb comes from history (envelope) or from an estimated fiscal
reaction function with fatigue (Bohn / Ghosh et al.), which also yields a
debt limit. Second and third differences are computed and tested as leading
indicators, never used as the trigger.

Units: d as a ratio (1.0 = 100% of GDP), r and g annualized decimals, pb as
a decimal share of GDP (surplus positive). Quarterly frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .. import obs


def _q(x_annual):
    """Annualized decimal rate -> per-quarter decimal rate."""
    return (1.0 + np.asarray(x_annual, float)) ** 0.25 - 1.0


# ── the identity ────────────────────────────────────────────────────────────

def decompose(d: pd.Series, r: pd.Series, g: pd.Series, pb: pd.Series) -> pd.DataFrame:
    """First-, second-, third-order dynamics of the ratio with the snowball /
    primary / residual decomposition. ``pb`` is the annual-rate balance; the
    per-quarter flow is pb/4."""
    rq, gq = _q(r), _q(g)
    d_lag = d.shift(1)
    out = pd.DataFrame(index=d.index)
    out["d"] = d
    out["delta_d"] = d.diff()
    out["snowball"] = d_lag * (rq - gq) / (1.0 + gq)
    out["primary"] = -pb / 4.0
    out["residual"] = out["delta_d"] - out["snowball"] - out["primary"]
    out["r_minus_g"] = r - g
    out["delta2_d"] = out["delta_d"].diff()
    out["delta3_d"] = out["delta2_d"].diff()
    out["delta2_snowball"] = out["snowball"].diff()
    out["delta2_primary"] = out["primary"].diff()
    return out


def stabilizing_primary_balance(d_lag, r, g):
    """Annualized pb* (decimal of GDP) that holds the ratio flat."""
    rq, gq = _q(r), _q(g)
    return 4.0 * np.asarray(d_lag, float) * (rq - gq) / (1.0 + gq)


def stabilizing_growth(d_lag, r, pb):
    """Annualized nominal growth g* that holds the ratio flat given r and pb.
    Derived from pb_q = d (r_q - g_q)/(1+g_q) with pb_q = pb/4, solved for g_q."""
    d = np.asarray(d_lag, float)
    rq = _q(r)
    pbq = np.asarray(pb, float) / 4.0
    gq = (d * rq - pbq) / (d + pbq)
    return (1.0 + gq) ** 4 - 1.0


# ── what is feasible ────────────────────────────────────────────────────────

@dataclass
class FiscalReaction:
    """pb_t = a + b1 d_{t-1} + b2 d_{t-1}^2 + b3 d_{t-1}^3 + c u_t + e_t (decimals)."""
    coef: np.ndarray
    r2: float
    n: int

    def feasible(self, d, u=None) -> np.ndarray:
        """Shape-agnostic: works on (T,) history and (K, H) path arrays alike."""
        d = np.asarray(d, float)
        u = np.zeros_like(d) if u is None else np.asarray(u, float)
        c = self.coef
        return c[0] + c[1] * d + c[2] * d ** 2 + c[3] * d ** 3 + c[4] * u

    @property
    def fatigue(self) -> bool:
        """Negative cubic: the response weakens at high debt."""
        return bool(self.coef[3] < 0)


def fit_fiscal_reaction(pb: pd.Series, d: pd.Series, u: Optional[pd.Series] = None) -> FiscalReaction:
    df = pd.DataFrame({"pb": pb, "d": d.shift(1)})
    df["u"] = u if u is not None else 0.0
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = np.column_stack([np.ones(len(df)), df["d"], df["d"] ** 2, df["d"] ** 3, df["u"] - df["u"].mean()])
    coef, *_ = np.linalg.lstsq(X, df["pb"].to_numpy(), rcond=None)
    pred = X @ coef
    r2 = 1.0 - np.var(df["pb"] - pred) / np.var(df["pb"])
    fr = FiscalReaction(coef=coef, r2=float(r2), n=len(df))
    obs.event(channel="analysis", kind="fiscal_reaction.fit", n=fr.n, r2=round(fr.r2, 3),
              coef=np.round(coef, 4).tolist(), fatigue=fr.fatigue)
    return fr


@dataclass(frozen=True)
class FeasiblePB:
    """The two benchmarks, evaluated per quarter (decimal of GDP)."""
    envelope: float                       # historical maximum (or quantile) primary balance
    reaction: Optional[FiscalReaction]    # estimated reaction function, may be None

    @classmethod
    def from_history(cls, pb: pd.Series, d: pd.Series, u: Optional[pd.Series] = None,
                     quantile: float = 1.0, fit_reaction: bool = True) -> "FeasiblePB":
        env = float(pb.dropna().quantile(quantile))
        fr = fit_fiscal_reaction(pb, d, u) if fit_reaction else None
        return cls(envelope=env, reaction=fr)

    def evaluate(self, d, u=None, which: str = "envelope") -> np.ndarray:
        d = np.asarray(d, float)
        if which == "envelope" or self.reaction is None:
            return np.full_like(d, self.envelope)
        if which == "reaction":
            return self.reaction.feasible(d, u)
        if which == "min":
            return np.minimum(self.envelope, self.reaction.feasible(d, u))
        raise ValueError(which)


def debt_limit(feasible: FeasiblePB, r_of_d, g: float, grid=np.linspace(0.2, 4.0, 381)) -> Optional[float]:
    """Smallest d on the grid where pb*(d) exceeds the reaction-function
    feasible balance: beyond it no historically observed response stabilizes
    the ratio. ``r_of_d`` maps d (ratio) to the annualized effective rate,
    including any premium."""
    if feasible.reaction is None:
        return None
    pbs = stabilizing_primary_balance(grid, np.asarray([r_of_d(x) for x in grid]), np.full_like(grid, g))
    feas = feasible.evaluate(grid, which="reaction")
    over = np.nonzero(pbs > feas)[0]
    return float(grid[over[0]]) if len(over) else None


# ── the trigger ─────────────────────────────────────────────────────────────

def first_run_index(mask: np.ndarray, n: int) -> np.ndarray:
    """(K,) index of the first column starting a run of >= n consecutive True
    values in each row of a (K, H) boolean array; -1 if none. Vectorized via a
    cumulative run-length."""
    m = mask.astype(np.int8)
    K, H = m.shape
    run = np.zeros_like(m, dtype=np.int32)
    run[:, 0] = m[:, 0]
    for t in range(1, H):                       # H is a few hundred; K is vectorized
        run[:, t] = (run[:, t - 1] + 1) * m[:, t]
    hit = run >= n
    first = np.where(hit.any(axis=1), hit.argmax(axis=1) - (n - 1), -1)
    return first


@dataclass
class SustainabilityDiagnostics:
    """Per-path outputs of the trigger evaluation."""
    trigger_index: np.ndarray            # (K,) first period of a sustained breach, -1 if none
    pb_star: np.ndarray                  # (K, H)
    g_star: np.ndarray                   # (K, H)
    feasible: np.ndarray                 # (K, H)
    gap: np.ndarray                      # (K, H) pb_star - feasible (>0 = beyond what is feasible)
    delta2_d: np.ndarray                 # (K, H)
    delta3_d: np.ndarray                 # (K, H)

    def probability(self) -> float:
        return float((self.trigger_index >= 0).mean())

    def lead_indicator_test(self, window: int = 8) -> Dict[str, float]:
        """Share of triggered paths on which delta2_d (delta3_d) was positive in
        the ``window`` periods before the trigger: a crude leading-indicator score."""
        out = {}
        hit = self.trigger_index >= 0
        for name, arr in (("delta2_d", self.delta2_d), ("delta3_d", self.delta3_d)):
            scores = []
            for k in np.nonzero(hit)[0]:
                t = self.trigger_index[k]
                pre = arr[k, max(t - window, 0): t]
                if len(pre):
                    scores.append(float(np.nanmean(pre > 0)))
            out[f"{name}_positive_share_pre_trigger"] = float(np.mean(scores)) if scores else np.nan
        return out


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean along the last axis with a growing window at the start."""
    if window <= 1:
        return x
    c = np.cumsum(x, axis=-1)
    out = np.empty_like(x, dtype=float)
    out[..., :window] = c[..., :window] / np.arange(1, window + 1)
    out[..., window:] = (c[..., window:] - c[..., :-window]) / window
    return out


def evaluate_paths(d: np.ndarray, r: np.ndarray, g: np.ndarray, pb: np.ndarray, feasible: FeasiblePB,
                   n_periods: int, which: str = "envelope", u: Optional[np.ndarray] = None,
                   periods_per_year: int = 12, require_r_gt_g: bool = True,
                   growth_smoothing: int = 12) -> SustainabilityDiagnostics:
    """Apply the trigger to (K, H) simulated paths. ``d`` ratio, ``r``/``g``
    annualized decimals, ``pb`` decimal share of GDP. ``n_periods`` is the
    persistence requirement in the arrays' own period units.

    A breach is pb* above the feasible balance; with ``require_r_gt_g`` it must
    also have a positive snowball (r > g). A rising ratio with r < g is a
    primary-deficit problem that growth is already eroding, not a loop; the
    loop is the explosive case where the stock feeds on itself.

    ``growth_smoothing``: growth enters as a trailing mean over this many
    periods so it is compared like-for-like with the effective rate, which is
    itself a trailing-12-month object. Raw quarterly growth made every
    one-quarter dip look like r > g."""
    g = trailing_mean(np.asarray(g, float), growth_smoothing)
    d_lag = np.concatenate([d[:, :1], d[:, :-1]], axis=1)
    pb_star = stabilizing_primary_balance(d_lag, r, g)
    g_star = stabilizing_growth(d_lag, r, pb)
    feas = feasible.evaluate(d_lag, u, which=which)
    gap = pb_star - feas
    breach = gap > 0
    if require_r_gt_g:
        breach &= (r > g)
    trig = first_run_index(breach, n_periods)
    dd = np.diff(d, axis=1, prepend=d[:, :1])
    d2 = np.diff(dd, axis=1, prepend=dd[:, :1])
    d3 = np.diff(d2, axis=1, prepend=d2[:, :1])
    diag = SustainabilityDiagnostics(trig, pb_star, g_star, feas, gap, d2, d3)
    obs.event(channel="sim", kind="sustainability.evaluate", K=d.shape[0], H=d.shape[1],
              which=which, n_periods=n_periods, p_trigger=diag.probability())
    return diag


def evaluate_history(q: pd.DataFrame, feasible: FeasiblePB, n_quarters: int = 4,
                     which: str = "envelope", require_r_gt_g: bool = True,
                     growth_smoothing_quarters: int = 4) -> pd.DataFrame:
    """Same trigger on the historical quarterly frame (columns d, r_eff, g_nom,
    pb, u in the units above). Returns the inputs, the decomposition, pb*, g*,
    the feasible balance, the gap, and the 'breach' / 'triggered' flags. Growth
    is smoothed over ``growth_smoothing_quarters`` for the trigger (see
    ``evaluate_paths``); the decomposition itself uses raw quarterly growth."""
    dec = decompose(q["d"], q["r_eff"], q["g_nom"], q["pb"])
    for c in ("r_eff", "g_nom", "pb", "u"):
        if c in q:
            dec[c] = q[c]
    g_s = q["g_nom"].rolling(growth_smoothing_quarters, min_periods=1).mean()
    dec["g_smooth"] = g_s
    d_lag = q["d"].shift(1)
    dec["pb_star"] = stabilizing_primary_balance(d_lag, q["r_eff"], g_s)
    dec["g_star"] = stabilizing_growth(d_lag, q["r_eff"], q["pb"])
    dec["feasible"] = feasible.evaluate(d_lag.to_numpy(), q["u"].to_numpy() if "u" in q else None, which=which)
    dec["gap"] = dec["pb_star"] - dec["feasible"]
    dec["breach"] = dec["gap"] > 0
    if require_r_gt_g:
        dec["breach"] &= (q["r_eff"] - g_s) > 0
    dec["triggered"] = dec["breach"].rolling(n_quarters).sum() >= n_quarters
    return dec


def evaluate_history_realtime(history_fn, as_of_dates, n_quarters: int = 4, which: str = "envelope",
                              require_r_gt_g: bool = True, growth_smoothing_quarters: int = 4,
                              quantile: float = 1.0, fit_reaction: bool = False, min_obs: int = 2) -> pd.DataFrame:
    """The trigger as it would have read in real time.

    ``history_fn(as_of)`` must return the quarterly frame (d, r_eff, g_nom, pb,
    u) as it was known on ``as_of`` (vintage data, nothing dated after it). For
    each date the feasible balance is fit on that frame only (an expanding
    window, so a surplus that happens later cannot raise the envelope
    earlier), ``evaluate_history`` runs on it, and the last row is kept. The
    result is one row per as_of date: what the trigger said *then*.

    ``evaluate_history`` on today's vintage is the hindsight table; this is the
    real-time one. Decision 0002 claims the former, and the difference between
    the two is a result the paper should report, not assume away."""
    rows = []
    for t in pd.to_datetime(list(as_of_dates)):
        h = history_fn(t)
        h = h.loc[h.index <= t]
        if len(h) < max(min_obs, 2):
            continue
        feas = FeasiblePB.from_history(h["pb"], h["d"], h["u"] if "u" in h else None,
                                       quantile=quantile, fit_reaction=fit_reaction)
        H = evaluate_history(h, feas, n_quarters=n_quarters, which=which, require_r_gt_g=require_r_gt_g,
                             growth_smoothing_quarters=growth_smoothing_quarters)
        last = H.iloc[-1]
        rows.append({"as_of": t, "date": H.index[-1], "n_obs": len(h), "envelope_asof": feas.envelope,
                     "pb_star": last["pb_star"], "feasible": last["feasible"], "gap": last["gap"],
                     "breach": bool(last["breach"]), "triggered": bool(last["triggered"])})
    return pd.DataFrame(rows).set_index("as_of")


def compare_realtime_final(realtime: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Align the real-time table (from ``evaluate_history_realtime``) with the
    hindsight table (from ``evaluate_history`` on the latest vintage) by
    observation date. Returns triggered_realtime, triggered_final, agree."""
    rt = realtime.set_index("date")[["breach", "triggered", "feasible"]].add_suffix("_realtime")
    fn = final[["breach", "triggered", "feasible"]].add_suffix("_final")
    out = rt.join(fn, how="inner")
    out["agree"] = out["triggered_realtime"] == out["triggered_final"]
    return out
