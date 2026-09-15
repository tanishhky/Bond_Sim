"""Observability spine: one structured JSONL event per fetch, fit, simulation.

Same pattern as RateWalk / PinSight / DriftEdge so a run is fully traceable:
every event carries a UTC timestamp, run id, channel, kind, level, and any
fields the caller adds (row counts, durations, series ids, config hash).

Usage
-----
    from bond_sim import obs
    obs.event(channel="data", kind="fred.fetch", series="GFDEBTN", n=312)

    with obs.timed(channel="engine", kind="book.build", n_bonds=2140):
        ...  # duration_ms and status are emitted on exit, errors re-raised
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOG_PATH: Optional[Path] = None
_RUN_ID: Optional[str] = None
# Credentials never reach a log line, whatever a library puts in an error message.
_SECRET_RE = re.compile(r"(api_key|apikey|token|password)=([^&\s\"']+)", re.IGNORECASE)


def _redact(text: str) -> str:
    return _SECRET_RE.sub(lambda m: f"{m.group(1)}=REDACTED", text)


def configure(log_dir: Optional[Path] = None, run_id: Optional[str] = None) -> None:
    """Point the spine at a log directory. Without this, events go to stderr only
    (and only WARNING/ERROR unless BOND_SIM_VERBOSE is set). Idempotent."""
    global _LOG_PATH, _RUN_ID
    _RUN_ID = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = log_dir / f"run-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"


def event(*, channel: str, kind: str, level: str = "INFO", **fields: Any) -> None:
    """Emit one structured event line."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "run_id": _RUN_ID,
        "pid": os.getpid(),
        "channel": channel,
        "kind": kind,
        "level": level,
        **fields,
    }
    line = _redact(json.dumps(rec, default=str))
    if _LOG_PATH is not None:
        try:
            with open(_LOG_PATH, "a") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    if level in ("WARNING", "ERROR") or os.getenv("BOND_SIM_VERBOSE"):
        print(line, file=sys.stderr)


@contextmanager
def timed(*, channel: str, kind: str, level: str = "INFO", **fields: Any):
    """Time a block; emit duration_ms and status on exit. Exceptions are logged
    as ERROR events and re-raised, never swallowed."""
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        status = "error"
        fields["err"] = str(exc)
        fields["exc_type"] = type(exc).__name__
        raise
    finally:
        event(channel=channel, kind=kind, level=("ERROR" if status == "error" else level),
              duration_ms=round((time.perf_counter() - t0) * 1000, 3), status=status, **fields)
