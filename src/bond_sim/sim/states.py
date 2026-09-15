"""Latent state chain on the block factors (docs/math.md 6).

    F_t = c_{S_t} + A F_{t-1} + e_t,   e_t ~ empirical residuals of state S_t
    S_t  a Markov chain with transition matrix P (Dirichlet-shrunk toward persistence)

Two-step, transparent estimation: a Gaussian HMM on the factor vector gives
the states and P; OLS with state dummies gives A (common) and c_s; residuals
by state give the bootstrap banks. States are ordered by an activity score
(consumer + corporate + labor factor means) so state 0 is always the
contraction / stress state and the last state the expansion.

Doom-loop channel: the hazard of entering state 0 is logistic in the
financial-conditions factor (slope estimated on history); the fiscal premium
shifts that factor through the Baa-spread loading, so a rising premium
raises the stress hazard. Shocks are dated forced transitions or factor
impulses. Everything random is drawn up front so policies and scenarios run
on common random numbers.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import obs

logging.getLogger("hmmlearn").setLevel(logging.ERROR)     # "Model is not converging" on rejected candidates is expected


@dataclass(frozen=True)
class ForcedTransition:
    """Hold the chain in ``state`` for ``duration`` quarters starting at quarter ``at``."""
    state: int
    at: int
    duration: int = 4
    label: str = ""


@dataclass(frozen=True)
class FactorImpulse:
    """Add ``size`` (factor standard deviations) to block ``block`` at quarter ``at``,
    decaying geometrically by ``decay`` per quarter."""
    block: str
    size: float
    at: int
    decay: float = 0.7
    label: str = ""


@dataclass
class Draws:
    """Common random numbers for one (K, HQ) run."""
    u_state: np.ndarray          # (K, HQ) uniforms for state transitions
    resid_idx: np.ndarray        # (K, HQ) uniforms mapped to residual-bank indices per state
    idio_idx: np.ndarray         # (K, HQ) uniforms for idiosyncratic bootstrap


class LatentStateModel:
    def __init__(self, n_states: Optional[int] = None, candidates=(2, 3, 4), prior_strength: float = 2.0,
                 seed: int = 0, financial_block: str = "financial",
                 activity_blocks=("consumer", "corporate", "labor"), winsor_sd: float = 3.5,
                 min_state_occupancy: int = 8, hazard_beta_override: Optional[float] = None):
        """``winsor_sd``: factors are clipped at +/- this many standard deviations for
        estimation (2020Q2 sits at -8 to -9 on three blocks and would otherwise get
        a state of its own). ``min_state_occupancy``: a candidate state count is
        admissible only if every state has at least this many quarters; a
        "regime" occupied by one observation is an outlier, not a regime (P-18)."""
        self.n_states, self.candidates, self.prior_strength, self.seed = n_states, candidates, prior_strength, seed
        self.financial_block, self.activity_blocks = financial_block, activity_blocks
        self.winsor_sd, self.min_state_occupancy = winsor_sd, min_state_occupancy
        self.hazard_beta_override = hazard_beta_override      # P-19: set the stress-hazard slope by hand
        self.selection: Dict[int, dict] = {}
        self.blocks: List[str] = []
        self.k: int = 0
        self.P: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None          # (k, B) HMM state means
        self.A: Optional[np.ndarray] = None              # (B, B)
        self.c: Optional[np.ndarray] = None              # (k, B)
        self.resid_bank: Dict[int, np.ndarray] = {}      # state -> (n_s, B)
        self.states: Optional[pd.Series] = None
        self.posterior: Optional[pd.DataFrame] = None
        self.hazard_beta: float = 0.0                    # logistic slope of P(-> state 0) on the financial factor (>= 0)
        self.hazard_beta_raw: float = 0.0                # the unrestricted estimate, reported
        self.hazard_ref: float = 0.0
        self.bic: Dict[int, float] = {}

    # ── estimation ──────────────────────────────────────────────────────────
    def fit(self, F: pd.DataFrame) -> "LatentStateModel":
        from hmmlearn.hmm import GaussianHMM
        self.blocks = list(F.columns)
        X_raw = F.to_numpy()
        X = np.clip(X_raw, -self.winsor_sd, self.winsor_sd) if self.winsor_sd else X_raw
        T, B = X.shape
        fits = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in ([self.n_states] if self.n_states else self.candidates):
                m = GaussianHMM(n_components=k, covariance_type="full", n_iter=500, random_state=self.seed).fit(X)
                n_par = k * B + k * B * (B + 1) / 2 + k * (k - 1) + (k - 1)
                occ = np.bincount(m.predict(X), minlength=k)
                self.bic[k] = float(-2 * m.score(X) + n_par * np.log(T))
                self.selection[k] = {"bic": round(self.bic[k], 1), "min_occupancy": int(occ.min()),
                                     "admissible": bool(occ.min() >= self.min_state_occupancy)}
                fits[k] = m
        admissible = [k for k, s in self.selection.items() if s["admissible"]]
        if admissible:
            k = min(admissible, key=lambda kk: self.bic[kk])
        else:                                                   # nothing qualifies: fewest states is the honest choice
            k = min(fits)
        hmm = fits[k]
        self.k = k
        raw_states = hmm.predict(X)
        post = hmm.predict_proba(X)
        # order by activity + financial conditions: state 0 = weakest and tightest,
        # the stress state the doom-loop hazard refers to (not merely low activity:
        # a post-crisis recovery is weak but loose and must not be labeled stress)
        act = [self.blocks.index(b) for b in list(self.activity_blocks) + [self.financial_block] if b in self.blocks]
        score = hmm.means_[:, act].sum(axis=1)
        order = np.argsort(score)
        remap = {int(old): int(new) for new, old in enumerate(order)}
        states = np.vectorize(remap.get)(raw_states)
        self.means = hmm.means_[order]
        self.states = pd.Series(states, index=F.index, name="state")
        self.posterior = pd.DataFrame(post[:, order], index=F.index, columns=[f"p{s}" for s in range(k)])
        # transition matrix: counts + sticky Dirichlet prior (the RateWalk lesson: shrink sparse regime data)
        counts = np.zeros((k, k))
        for a, b in zip(states[:-1], states[1:]):
            counts[a, b] += 1
        prior = self.prior_strength * (0.5 * np.eye(k) + 0.5 / k)
        self.P = (counts + prior) / (counts + prior).sum(axis=1, keepdims=True)
        # within-state VAR(1): F_t = c_s + A F_{t-1} + e_t with common A
        Y, Xl = X[1:], X[:-1]
        D = np.eye(k)[states[1:]]                                # state dummies
        Z = np.concatenate([D, Xl], axis=1)
        coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
        self.c, self.A = coef[:k], coef[k:].T
        resid = Y - Z @ coef
        self.resid_bank = {s: resid[states[1:] == s] for s in range(k)}
        for s in range(k):
            if len(self.resid_bank[s]) < 8:                     # too thin to bootstrap from; pool
                self.resid_bank[s] = resid
        # stress hazard: logistic of (next state == 0) on the financial factor, given current state != 0
        fin = self.blocks.index(self.financial_block) if self.financial_block in self.blocks else None
        if fin is not None:
            self.hazard_ref = float(X[:, fin].mean())
            beta = self._fit_hazard(X[:-1, fin] - self.hazard_ref, states[:-1], states[1:])
            # Sign restriction from the mechanism: tighter conditions cannot lower the
            # stress hazard. A negative estimate means the states do not separate
            # stress from recovery on this sample; the channel is then switched off
            # (beta = 0) and reported, never used with the wrong sign.
            self.hazard_beta_raw = beta
            self.hazard_beta = max(beta, 0.0)
            if beta < 0:
                obs.event(channel="sim", kind="states.hazard_sign", level="WARNING", beta_estimate=round(beta, 3),
                          note="negative slope on the financial factor; hazard channel disabled")
            if self.hazard_beta_override is not None:
                self.hazard_beta = float(self.hazard_beta_override)
        obs.event(channel="sim", kind="states.fit", k=k, selection=self.selection, P=np.round(self.P, 3).tolist(),
                  state_means={s: dict(zip(self.blocks, np.round(self.means[s], 2).tolist())) for s in range(k)},
                  state_counts=np.bincount(states, minlength=k).tolist(), hazard_beta=round(self.hazard_beta, 3),
                  max_eig_A=round(float(np.abs(np.linalg.eigvals(self.A)).max()), 3))
        return self

    @staticmethod
    def _fit_hazard(x: np.ndarray, s_now: np.ndarray, s_next: np.ndarray) -> float:
        """Slope of a logistic regression of 1[s_next == 0] on x among quarters with
        s_now != 0 (entering stress), by a few Newton steps. Returns 0 if degenerate."""
        m = s_now != 0
        x, y = x[m], (s_next[m] == 0).astype(float)
        if y.sum() < 3 or y.sum() > len(y) - 3:
            return 0.0
        b = np.zeros(2)
        Xd = np.column_stack([np.ones_like(x), x])
        for _ in range(25):
            p = 1 / (1 + np.exp(-(Xd @ b)))
            W = p * (1 - p)
            H = Xd.T @ (Xd * W[:, None]) + 1e-6 * np.eye(2)
            b = b + np.linalg.solve(H, Xd.T @ (y - p))
        return float(np.clip(b[1], -5, 5))

    # ── simulation ──────────────────────────────────────────────────────────
    def draw(self, K: int, HQ: int, rng: np.random.Generator) -> Draws:
        return Draws(rng.random((K, HQ)), rng.random((K, HQ)), rng.random((K, HQ)))

    def initial_state_probs(self) -> np.ndarray:
        return self.posterior.iloc[-1].to_numpy()

    def transition_row(self, s: np.ndarray, fin_dev: np.ndarray) -> np.ndarray:
        """(K, k) transition probabilities for each path's current state, with the
        stress hazard modulated by the financial factor deviation (looser = lower)."""
        P = self.P[s]                                              # (K, k)
        if self.hazard_beta == 0.0:
            return P
        base = np.clip(P[:, 0], 1e-6, 1 - 1e-6)
        logit = np.log(base / (1 - base)) - self.hazard_beta * fin_dev    # tighter conditions (lower factor) -> higher hazard
        p0 = 1 / (1 + np.exp(-logit))
        out = P.copy()
        rest = np.maximum(1 - P[:, 0], 1e-9)
        out[:, 1:] = P[:, 1:] * ((1 - p0) / rest)[:, None]
        out[:, 0] = p0
        return out

    def step(self, tq: int, F: np.ndarray, s: Optional[np.ndarray], draws: Draws, p0: np.ndarray,
             shocks: List, impulse: np.ndarray, fin_shift: Optional[np.ndarray] = None):
        """One quarter of the chain for K paths, in place on ``F`` and ``impulse``.
        Returns (F, s). ``fin_shift`` (K,) is added to the financial factor
        before the transition draw (the premium channel)."""
        K, B, k = F.shape[0], len(self.blocks), self.k
        fin = self.blocks.index(self.financial_block) if self.financial_block in self.blocks else None
        if fin is not None and fin_shift is not None:
            F[:, fin] += fin_shift
        fin_dev = (F[:, fin] - self.hazard_ref) if fin is not None else np.zeros(K)
        if s is None:
            s = (draws.u_state[:, 0][:, None] > np.cumsum(p0)[None, :]).sum(axis=1).clip(0, k - 1)
        else:
            cum = np.cumsum(self.transition_row(s, fin_dev), axis=1)
            s = (draws.u_state[:, tq][:, None] > cum).sum(axis=1).clip(0, k - 1)
        for sh in shocks:                                          # forced transitions override the draw
            if isinstance(sh, ForcedTransition) and sh.at <= tq < sh.at + sh.duration:
                s = np.full(K, sh.state)
        e = np.empty((K, B))
        for st in range(k):                                        # residual bootstrap from the state's bank
            m = s == st
            if m.any():
                bank = self.resid_bank[st]
                e[m] = bank[(draws.resid_idx[m, tq] * len(bank)).astype(int)]
        F[:] = self.c[s] + F @ self.A.T + e
        for sh in shocks:
            if isinstance(sh, FactorImpulse) and tq == sh.at:
                impulse[:, self.blocks.index(sh.block)] += sh.size
        F += impulse
        for sh in shocks:
            if isinstance(sh, FactorImpulse) and tq >= sh.at:
                impulse[:, self.blocks.index(sh.block)] *= sh.decay
        return F, s

    def simulate(self, K: int, HQ: int, draws: Draws, F0: np.ndarray, p0: np.ndarray,
                 shocks: Optional[List] = None) -> Dict[str, np.ndarray]:
        """Standalone chain simulation (no debt engine): (K, HQ, B) factor paths and
        (K, HQ) states, for inspecting state occupancy and factor distributions."""
        shocks = shocks or []
        B = len(self.blocks)
        F = np.broadcast_to(F0, (K, B)).copy()
        s = None
        impulse = np.zeros((K, B))
        out_F, out_S = np.empty((K, HQ, B)), np.empty((K, HQ), dtype=int)
        for tq in range(HQ):
            F, s = self.step(tq, F, s, draws, p0, shocks, impulse)
            out_F[:, tq, :], out_S[:, tq] = F, s
        return {"F": out_F, "S": out_S}

    def occupancy(self, S: np.ndarray) -> pd.DataFrame:
        """Share of paths in each state per quarter, (HQ, k)."""
        return pd.DataFrame({s: (S == s).mean(axis=0) for s in range(self.k)})

    def describe(self) -> str:
        return f"{self.k}-state latent chain on {len(self.blocks)} block factors (hazard beta {self.hazard_beta:.2f})"
