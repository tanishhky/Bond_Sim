"""Pluggable quarterly macro blocks for the doom-loop engine.

The debt accounting (monthly, bottom-up) is the same whichever model
produces the quarterly macro state. Two blocks implement one interface so
the notebook can run them on identical draws and compare outcome
distributions:

    VARBlock     the estimated levels VAR (Gaussian shocks; decision 0005)
    StateBlock   block factors + latent Markov chain with bootstrapped
                 residuals and forced-transition / impulse shocks (math.md 5-6)

Interface (all arrays are (K,) per path):
    prepare(K, HQ, rng)                 -> draws (common random numbers, opaque)
    reset(K)                            -> per-run mutable state
    advance(tq, draws, d_prem, policy, ctx) -> dict(r10_base, r3m, g_nom, pb, u, infl)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..analysis.factors import BlockFactors
from .macro import VARS, MacroVAR
from .states import Draws, LatentStateModel


class VARBlock:
    def __init__(self, var: MacroVAR, transmission: str = "var", credit_elasticity: float = -0.006):
        """``transmission``: "var" (default, the VAR's own lagged-10y
        coefficients) or "credit" (Arellano-Bai-Bocola/Bocola-calibrated
        sovereign-risk pass-through, decision 0005, P-20). ``credit_elasticity``
        only matters when ``transmission="credit"``; pass the low/central/high
        values from MacroVAR.credit_channel_transmission's docstring to run
        the band, not just one point."""
        if transmission not in ("var", "credit"):
            raise ValueError(f"unknown transmission {transmission!r}")
        self.var = var
        self.transmission = transmission
        self.credit_elasticity = credit_elasticity
        self.i = {v: var.index_of(v) for v in VARS}
        self.hist = None
        self.u_prev = None

    def _transmit(self, d_prem: np.ndarray) -> np.ndarray:
        if self.transmission == "credit":
            return self.var.credit_channel_transmission(d_prem, output_elasticity=self.credit_elasticity)
        return self.var.rate_transmission(d_prem)

    def describe(self) -> str:
        chan = "VAR-own-coefficients" if self.transmission == "var" else f"credit-channel ({self.credit_elasticity})"
        return f"levels VAR({self.var.lag}), Gaussian shocks, {chan} transmission"

    def prepare(self, K: int, HQ: int, rng: np.random.Generator) -> np.ndarray:
        return self.var.draw_shocks(K, HQ, rng)

    def reset(self, K: int, u0: float) -> None:
        p = self.var.lag
        self.hist = np.broadcast_to(self.var.last_history(), (K, p, len(VARS))).copy()
        self.u_prev = np.full(K, u0)

    def advance(self, tq: int, draws: np.ndarray, d_prem: np.ndarray, policy, ctx: dict) -> Dict[str, np.ndarray]:
        i = self.i
        y = self.var.step(self.hist, draws[:, tq, :], self._transmit(d_prem))
        # effective lower bound on nominal yields: the one structural floor (cash exists)
        y[:, i["r10"]] = np.maximum(y[:, i["r10"]], 0.0)
        y[:, i["r3m"]] = np.maximum(y[:, i["r3m"]], 0.0)
        y[:, i["g_nom"]] = policy.growth(tq, y[:, i["g_nom"]], ctx)
        y[:, i["pb"]] = policy.primary_balance(tq, y[:, i["pb"]], ctx)
        u_level = np.exp(y[:, i["u"]])                                # the VAR carries log(u)
        d_u = policy.unemployment_change(tq, u_level - self.u_prev, ctx)
        u_level = np.maximum(self.u_prev + d_u, 0.1)
        y[:, i["u"]] = np.log(u_level)
        self.u_prev = u_level.copy()
        p = self.var.lag
        self.hist = np.concatenate([self.hist[:, 1:, :], y[:, None, :]], axis=1) if p > 1 else y[:, None, :]
        return {"r10_base": y[:, i["r10"]], "r3m": y[:, i["r3m"]], "g_nom": y[:, i["g_nom"]],
                "pb": y[:, i["pb"]], "u": u_level, "infl": np.full(len(y), np.nan)}


class StateBlock:
    def __init__(self, factors: BlockFactors, model: LatentStateModel, shocks: Optional[List] = None,
                 premium_to_financial: Optional[float] = None):
        """``premium_to_financial``: standardized financial-factor shift per pct
        point of premium, signed (negative = tighter, since the factor's high
        side is loose). Default derives it from the Baa-spread loading: a 1pt
        premium is treated like a 1pt Baa spread widening (P-16), and the
        loading of the spread on the factor is negative, so the default is
        negative."""
        self.factors, self.model, self.shocks = factors, model, shocks or []
        self.bidx = factors.block_index()
        if premium_to_financial is None:
            f = factors.fits["financial"]
            j = f.columns.index("BAA10Y") if "BAA10Y" in f.columns else None
            premium_to_financial = float(f.loadings[j] / f.std[j]) if j is not None else 0.0
        self.k_prem = premium_to_financial
        self.F = None
        self.s = None
        self.u_prev = None
        self._impulse = None

    def describe(self) -> str:
        return self.model.describe() + (f", {len(self.shocks)} shock(s)" if self.shocks else "")

    def prepare(self, K: int, HQ: int, rng: np.random.Generator) -> Draws:
        return self.model.draw(K, HQ, rng)

    def reset(self, K: int, u0: float) -> None:
        B = len(self.model.blocks)
        self.F = np.broadcast_to(self.factors.last_factors(), (K, B)).copy()
        p0 = self.model.initial_state_probs()
        self.s = None
        self.p0 = p0
        self.u_prev = np.full(K, u0)
        self._impulse = np.zeros((K, B))
        self.idio = {k: np.broadcast_to(v, (K,)).copy() for k, v in self.factors.idio_start().items()}

    def advance(self, tq: int, draws: Draws, d_prem: np.ndarray, policy, ctx: dict) -> Dict[str, np.ndarray]:
        m, K = self.model, self.F.shape[0]
        # premium channel: k_prem is signed (negative = tighter), so a rising premium lowers the factor
        self.F, self.s = m.step(tq, self.F, self.s, draws, self.p0, self.shocks, self._impulse,
                                fin_shift=self.k_prem * d_prem)
        # idiosyncratic residuals follow their estimated AR(1) with bootstrapped innovations
        self.idio = self.factors.idio_step(self.idio, draws.idio_idx[:, tq])
        out = self.factors.engine_targets(self.F, self.bidx, self.idio)
        out["r10_base"] = np.maximum(out.pop("r10"), 0.0)          # effective lower bound, the one structural floor
        out["r3m"] = np.maximum(out["r3m"], 0.0)
        out["g_nom"] = policy.growth(tq, out["g_nom"], ctx)
        out["pb"] = policy.primary_balance(tq, out["pb"], ctx)
        d_u = policy.unemployment_change(tq, out["u"] - self.u_prev, ctx)
        out["u"] = self.u_prev + d_u
        self.u_prev = out["u"].copy()
        out.setdefault("infl", np.full(K, np.nan))
        out["state"] = self.s.copy()
        return out
