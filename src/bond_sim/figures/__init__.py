"""Signature figures (thesis charter A9): brief -> prototype -> stress-test -> publish -> update.

Prototypes live in notebooks and never leave the repo. A figure that has passed its
stress test gets a builder here, registered in FIGURES under its id, and is published
with `bond_sim figures build F-##`. Every figure uses the house style (style.py) and
the two-layer uncertainty band (bands.py); publish.py versions the output, records
the manifest and takes the headline verbatim from the thesis claims register.

A builder is `fn(cfg, args) -> (FigureSpec, draw, data, run_meta)`.
"""
from __future__ import annotations

from typing import Callable, Dict

from .bands import Envelope, envelope, two_layer_band
from .publish import FigureSpec, caption, headline, load_claims, publish

FIGURES: Dict[str, Callable] = {}   # "F-01": build_clock, once N-009 reaches the publish stage

__all__ = ["FIGURES", "Envelope", "FigureSpec", "caption", "envelope", "headline", "load_claims",
           "publish", "two_layer_band"]
