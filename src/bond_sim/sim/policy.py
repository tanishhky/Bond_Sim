"""Policy counterfactuals as structured interventions on the same simulation.

Common random numbers: every policy is run on the identical shock draws, so
the difference between two policies' outcome distributions is the policy, not
the dice. Each policy is a set of hooks the simulator calls every quarter;
the base class makes every hook a no-op, so ``StatusQuo`` is the empty policy.

Hooks (all vectorized over K paths; ``state`` is the simulator's live dict):
    primary_balance(t, pb)         -> adjusted pb (pct of GDP)
    growth(t, g)                   -> adjusted nominal growth (annualized pct)
    unemployment_change(t, d_u)    -> adjusted quarterly change in u
    issuance_yield(t, y, y_base)   -> yield paid on this quarter's net issuance
    inflation_uplift(t)            -> extra nominal growth from monetization

Every magnitude below is a placeholder awaiting Tanishk's call (P-06..P-09):
the mechanics are built, the numbers are not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import PolicyConfig


class Policy:
    name = "base"

    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

    def primary_balance(self, t: int, pb: np.ndarray, state: dict) -> np.ndarray:
        return pb

    def growth(self, t: int, g: np.ndarray, state: dict) -> np.ndarray:
        return g

    def unemployment_change(self, t: int, d_u: np.ndarray, state: dict) -> np.ndarray:
        return d_u

    def issuance_yield(self, t: int, y: np.ndarray, y_base: np.ndarray, state: dict) -> np.ndarray:
        return y

    def in_stress(self, t: int, state: dict) -> np.ndarray:
        """Boolean (K,) mask of paths currently in a stress episode. Default
        trigger: the premium is positive (debt/GDP above the anchor) AND
        unemployment rose over the last four quarters. TODO(Tanishk) P-06."""
        return (state["premium"] > 0) & (state["u"] - state["u_4q_ago"] > 0)


class StatusQuo(Policy):
    name = "status_quo"


class NoLayoffMandate(Policy):
    """Cap the quarterly rise in unemployment during stress; pay for it with a
    growth penalty (labor hoarding lowers productivity, business failures rise).

    Mechanism choice (P-06): ``floor_with_penalty`` clips d_u at the floor and
    subtracts a growth penalty proportional to the clipped amount;
    ``propensity_dampener`` scales d_u by a factor instead of clipping.
    """
    name = "no_layoff_mandate"
    growth_penalty_per_pt_u: float = 0.5     # pct pts of annualized growth per pt of prevented unemployment. TODO(Tanishk) P-06

    def unemployment_change(self, t, d_u, state):
        stress = self.in_stress(t, state)
        if self.cfg.no_layoff_cost_mechanism == "propensity_dampener":
            adj = np.where(stress & (d_u > 0), d_u * (1.0 - 0.5), d_u)   # damp 50%. TODO(Tanishk) P-06
        else:
            cap = self.cfg.no_layoff_floor_pct
            adj = np.where(stress, np.minimum(d_u, cap), d_u)
        state["_prevented_u"] = np.maximum(d_u - adj, 0.0)
        return adj

    def growth(self, t, g, state):
        prevented = state.get("_prevented_u", 0.0)
        return g - self.growth_penalty_per_pt_u * prevented


class Austerity(Policy):
    """Move the primary balance toward a surplus target over a ramp; growth falls
    by multiplier x consolidation (P-07: target and multiplier)."""
    name = "austerity"
    ramp_quarters: int = 12
    multiplier: float = 1.0                  # fiscal multiplier on nominal growth. TODO(Tanishk) P-07

    def primary_balance(self, t, pb, state):
        w = min(t / self.ramp_quarters, 1.0)
        target = self.cfg.austerity_primary_balance_pct_gdp
        adj = pb + w * np.maximum(target - pb, 0.0)
        consolidation = adj - pb
        # The multiplier acts on the *change* in fiscal stance, not its level:
        # a level-based penalty feeds back through the VAR (lower g -> lower pb
        # -> bigger gap -> bigger penalty) and diverges. Standard fiscal-impulse form.
        state["_consolidation_delta"] = consolidation - state.get("_consolidation", np.zeros_like(consolidation))
        state["_consolidation"] = consolidation
        return adj

    def growth(self, t, g, state):
        return g - self.multiplier * state.get("_consolidation_delta", 0.0)


class Monetization(Policy):
    """The Fed absorbs a share of net issuance at the premium-free base yield;
    the price is higher inflation, modeled as extra nominal growth (which
    mechanically lowers debt/GDP) and no real gain (P-08: share, inflation)."""
    name = "monetization"
    inflation_uplift_pct: float = 2.0         # extra annualized nominal growth. TODO(Tanishk) P-08

    def issuance_yield(self, t, y, y_base, state):
        s = self.cfg.monetization_share
        return s * y_base + (1.0 - s) * y

    def growth(self, t, g, state):
        return g + self.inflation_uplift_pct


class GrowthLed(Policy):
    """Higher trend growth outruns the debt (P-09: uplift)."""
    name = "growth_led"

    def growth(self, t, g, state):
        return g + self.cfg.growth_uplift_pct


def policy_from_name(name: str, cfg: PolicyConfig) -> Policy:
    table = {p.name: p for p in (StatusQuo, NoLayoffMandate, Austerity, Monetization, GrowthLed)}
    if name not in table:
        raise KeyError(f"unknown policy {name!r}; choose from {sorted(table)}")
    return table[name](cfg)
