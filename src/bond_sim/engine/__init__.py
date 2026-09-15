"""Cash-flow engine: the bond book and scenario discounting.

    book      BondBook: one row per auction tranche on the shared grid; dense
              (N, M) masks for attribution and O(pairs) aggregates for speed,
              proven equal by test
    discount  path-dependent discount factors from simulated rate paths and the
              (K, M) @ (M,) present-value collapse
"""
from .book import BondBook, Aggregates
from .discount import discount_factors, flat_scenario_factors, present_value

__all__ = ["BondBook", "Aggregates", "discount_factors", "flat_scenario_factors", "present_value"]
