"""Scenario discounting: rate paths -> discount factors -> present values.

Problem 4 used flat rates (``DF[k,t] = 1/(1+r_k/2)^t``). Real paths move every
month, so the discount factor to column t under path k is the running product
of one-month factors along that path:

    DF[k, t] = prod_{s=1..t} 1 / (1 + r[k, s] / 12)

which is one ``cumprod`` along axis 1, no loop. Pricing the aggregate cash
flow vector under every path is still one matmul, ``DF (K, M) @ cf (M,)``.
"""
from __future__ import annotations

import numpy as np


def discount_factors(rate_paths: np.ndarray, periods_per_year: int = 12) -> np.ndarray:
    """(K, M) discount factors from (K, M) annualized decimal rate paths.
    Column 0 is today (DF = 1); column t discounts through rates 1..t."""
    r = np.asarray(rate_paths, dtype=float)
    step = 1.0 / (1.0 + r / periods_per_year)
    step[:, 0] = 1.0                                   # nothing to discount at t=0
    return np.cumprod(step, axis=1)


def flat_scenario_factors(rates: np.ndarray, M: int, periods_per_year: int = 2) -> np.ndarray:
    """Problem 4's closed form for flat scenarios (default semiannual, as in the toy)."""
    r = np.asarray(rates, dtype=float)[:, None]
    t = np.arange(M)[None, :]
    return 1.0 / (1.0 + r / periods_per_year) ** t


def present_value(df: np.ndarray, cashflows: np.ndarray) -> np.ndarray:
    """(K,) present values: one matmul collapses the period axis for every path."""
    return np.asarray(df) @ np.asarray(cashflows, dtype=float)
