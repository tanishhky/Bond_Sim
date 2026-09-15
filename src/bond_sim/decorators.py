"""Cross-cutting decorators: timing/observability, disk caching, point-in-time guards.

Three concerns that would otherwise be copy-pasted into every data loader and
engine method live here once:

``@timed``          wraps a call in an ``obs.timed`` block (duration, status, args)
``@disk_cache``     memoizes a DataFrame-returning call to Parquet keyed by its
                    arguments, so a 30-year ALFRED vintage pull happens once
``@point_in_time``  enforces the no-lookahead contract at the boundary: the
                    function must be called with ``as_of``, and nothing it
                    returns may be dated (release or observation) after it
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from . import obs

# Cache root: env override, else ./data_cache under the repo (gitignored).
_CACHE_ROOT = Path(os.getenv("BOND_SIM_CACHE_DIR", Path(__file__).resolve().parents[2] / "data_cache"))


def timed(channel: str, kind: str, **static_fields: Any) -> Callable:
    """Emit one obs event with duration_ms and status around every call."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with obs.timed(channel=channel, kind=kind, **static_fields):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def _stable_key(fn: Callable, args: tuple, kwargs: dict, ignore: tuple) -> str:
    """Deterministic hash of the bound arguments (minus ignored names, e.g. api keys)."""
    bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    bound.apply_defaults()
    payload = {k: v for k, v in bound.arguments.items() if k not in ignore and k != "self"}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def disk_cache(namespace: str, ignore: tuple = ("api_key", "session", "self"),
               enabled_env: str = "BOND_SIM_CACHE") -> Callable:
    """Parquet-memoize a DataFrame-returning function.

    Cache key = sha256 of the bound arguments except ``ignore``. Set
    ``BOND_SIM_CACHE=0`` to bypass (forces a live fetch and rewrites the file).
    The cached frame is the *raw* vintage-bearing pull, never an as_of view, so
    one cache entry serves every as_of query.
    """
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            use_cache = os.getenv(enabled_env, "1") != "0"
            key = _stable_key(fn, args, kwargs, ignore)
            path = _CACHE_ROOT / namespace / f"{key}.parquet"
            if use_cache and path.exists():
                obs.event(channel="cache", kind="hit", namespace=namespace, key=key)
                return pd.read_parquet(path)
            df = fn(*args, **kwargs)
            if isinstance(df, pd.DataFrame):
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
                obs.event(channel="cache", kind="write", namespace=namespace, key=key, n=len(df))
            return df
        return wrapper
    return deco


def point_in_time(date_col: str = "date", release_col: Optional[str] = "realtime_start") -> Callable:
    """Boundary guard for the no-lookahead contract.

    The decorated function must accept ``as_of`` (keyword or positional). After
    it returns a DataFrame, every row is checked: the release column (if the
    frame has one) and the observation date must both be <= as_of. A violation
    raises instead of silently leaking the future into a past estimate.
    """
    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        if "as_of" not in sig.parameters:
            raise TypeError(f"{fn.__qualname__} must take an as_of argument to be @point_in_time")

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            as_of = bound.arguments.get("as_of")
            if as_of is None:
                raise ValueError(f"{fn.__qualname__}: as_of is required (no-lookahead contract)")
            cut = pd.Timestamp(as_of)
            out = fn(*args, **kwargs)
            if isinstance(out, pd.DataFrame) and len(out):
                if release_col and release_col in out.columns:
                    late = out[release_col] > cut
                    assert not late.any(), (
                        f"{fn.__qualname__}: {int(late.sum())} rows released after as_of={cut.date()}")
                if date_col in out.columns:
                    late = out[date_col] > cut
                    assert not late.any(), (
                        f"{fn.__qualname__}: {int(late.sum())} rows observed after as_of={cut.date()}")
            return out
        return wrapper
    return deco
