"""Typed, YAML-driven configuration with a content hash on every output.

Nothing in the model path is hard-coded. Load ``config/default.yaml`` (or a
file passed on the CLI), get a frozen validated ``Config``, and stamp
``cfg.content_hash()`` on every artifact so a number in the paper traces back
to the exact inputs that produced it.

Parameters that still need Tanishk's decision (not derivable from data) are
marked ``TODO(Tanishk)`` here and listed in ``PARAMETERS.md``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class DataConfig:
    history_start: str = "1980-01-01"       # earliest month on the grid (MTS starts 1980-10)
    as_of: Optional[str] = None             # None -> today; every loader filters on this
    vintages: bool = True                   # full ALFRED vintages (decision 0002)
    fred_key_env: str = "FRED_API_KEY"
    cache: bool = True                      # parquet cache of raw vintage pulls


@dataclass(frozen=True)
class CalendarConfig:
    freq: str = "M"                         # monthly grid (decision 0001); only "M" is implemented
    horizon_years: int = 30                 # forward simulation window (CBO long-term outlook length)


@dataclass(frozen=True)
class BookConfig:
    """How auction records become bond rows."""
    include_tips: bool = True               # TIPS carried at issued face; inflation accrual is a documented gap
    include_frn: bool = True                # FRNs: coupon floats with 3m bill rate, modeled as bills for interest
    include_cmb: bool = True                # cash-management bills are real debt, keep them
    coupon_months: int = 6                  # notes/bonds pay semiannually
    face_units: float = 1e6                 # store face in $ millions to keep arrays well-scaled
    validate_against_mspd: bool = True      # cross-check reconstructed outstanding vs MSPD table 3


@dataclass(frozen=True)
class CorrelationConfig:
    """Nothing is left independent: every pair gets a full-sample, rolling, and
    regime-split correlation, plus a lead/lag scan and a Granger test."""
    rolling_window_months: int = 60         # 5y rolling correlation window
    max_lag_months: int = 24                # lead/lag scan +/- 2 years
    regime_method: str = "quantile"         # quantile | hmm
    n_regimes: int = 3
    fdr_alpha: float = 0.05                 # Benjamini-Hochberg control on the Granger grid
    transform: str = "diff"                 # diff | level | pct; applied before correlating (stationarity)


@dataclass(frozen=True)
class SimConfig:
    n_paths: int = 5000
    seed: int = 42
    dt_months: int = 1                      # one grid column per step
    max_rate_pct: float = 25.0              # hard ceiling on any simulated yield (bounds explosive tails)


@dataclass(frozen=True)
class RiskPremiumConfig:
    """Phase 4 feedback: yield = base + premium(fiscal state).

    TODO(Tanishk): functional form and slope are the load-bearing thesis
    assumption. ``linear`` is the simplest defensible default; the slope
    below is a placeholder to be replaced by an estimated or expert-calibrated
    value with a documented sensitivity sweep (see PARAMETERS.md, P-01).
    """
    model: str = "linear"                   # linear | threshold | none
    bps_per_pct_debt_gdp: float = 2.0       # TODO(Tanishk) P-01: bps of premium per 1pt of debt/GDP above anchor
    anchor_debt_gdp_pct: float = 100.0      # premium is zero at this debt/GDP
    threshold_debt_gdp_pct: float = 130.0   # threshold model only: kink location. TODO(Tanishk) P-02
    threshold_extra_bps_per_pct: float = 6.0  # threshold model only: slope beyond the kink. TODO(Tanishk) P-02


@dataclass(frozen=True)
class DoomLoopConfig:
    var_start: str = "1985-01-01"           # estimation sample start for the quarterly levels VAR
    var_max_lag: int = 4
    # Long-run anchors. None -> the VAR's estimated sample mean (reported at run
    # time). Setting one is an explicit view and is logged as such.
    pb_anchor_pct_gdp: Optional[float] = None   # primary balance, % GDP (P-03)
    r10_anchor_pct: Optional[float] = None      # long-run 10y (P-13)
    u_anchor_pct: Optional[float] = None        # natural rate of unemployment (P-14)
    risk_premium: RiskPremiumConfig = field(default_factory=RiskPremiumConfig)
    loop_trigger_debt_gdp_pct: float = 150.0   # diagnostic only: "loop took hold" if crossed. TODO(Tanishk) P-04
    loop_trigger_interest_rev_pct: float = 30.0  # or interest / receipts crosses this. TODO(Tanishk) P-04


@dataclass(frozen=True)
class LaborConfig:
    sectors: tuple = ("USCONS", "CES5553000001", "CES3133600101", "CES5552200001",
                      "USFIRE", "CES9091000001", "USPBS", "USTRADE", "MANEMP")
    beta_source: str = "estimated"          # estimated (from analysis) | literature (PARAMETERS.md P-05)
    shock_lag_months: int = 6               # rate shock -> employment response lag; estimated, override here


@dataclass(frozen=True)
class PolicyConfig:
    menu: tuple = ("status_quo", "no_layoff_mandate", "austerity", "monetization", "growth_led")
    stress_window_months: int = 24          # no-layoff mandate duration. TODO(Tanishk) P-06
    no_layoff_floor_pct: float = 0.0        # max allowed sector employment decline under mandate. TODO(Tanishk) P-06
    no_layoff_cost_mechanism: str = "floor_with_penalty"   # floor_with_penalty | propensity_dampener. TODO(Tanishk) P-06
    austerity_primary_balance_pct_gdp: float = 2.0         # primary surplus target. TODO(Tanishk) P-07
    monetization_share: float = 0.5          # share of issuance absorbed by the Fed. TODO(Tanishk) P-08
    growth_uplift_pct: float = 0.5           # extra trend real growth (pct pts). TODO(Tanishk) P-09


@dataclass(frozen=True)
class RecoveryConfig:
    half_life_months_prior: float = 30.0    # unemployment gap half-life prior; estimated from episodes. P-10
    episodes: tuple = ("gfc_2008", "volcker_1981", "covid_2020")   # US calibration episodes


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    book: BookConfig = field(default_factory=BookConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    doomloop: DoomLoopConfig = field(default_factory=DoomLoopConfig)
    labor: LaborConfig = field(default_factory=LaborConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def content_hash(self) -> str:
        """Stable 12-hex hash of the whole config; stamped on every artifact."""
        blob = json.dumps(_to_plain(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _build(dc_type, raw: Optional[dict]):
    """Frozen dataclass from a (possibly partial) dict; recurses into nested
    dataclasses; YAML lists become tuples so the result stays hashable."""
    if not raw:
        return dc_type()
    kwargs = {}
    for f in fields(dc_type):
        if f.name not in raw:
            continue
        val = raw[f.name]
        default = f.default_factory() if f.default_factory is not MISSING else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(val, dict):
            kwargs[f.name] = _build(type(default), val)
        elif isinstance(val, list):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = _coerce(val, default)
    return dc_type(**kwargs)


def _coerce(val: Any, default: Any) -> Any:
    """Cast a YAML scalar to the declared field's type. PyYAML reads "1.0e6" as
    a string and "1e3" likewise; a typed default tells us what was meant."""
    if isinstance(default, bool) or default is None or val is None:
        return val
    if isinstance(default, int) and not isinstance(val, bool):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return val
    if isinstance(default, float):
        try:
            return float(val)
        except (TypeError, ValueError):
            return val
    return val


def load(path: Optional[Path] = None) -> Config:
    """Load from YAML (default: config/default.yaml). Unknown keys are ignored,
    missing keys take the dataclass default, so a partial override file works."""
    raw: dict = {}
    p = Path(path) if path else _DEFAULT_PATH
    if p.exists():
        with open(p) as fh:
            raw = yaml.safe_load(fh) or {}
    return Config(
        data=_build(DataConfig, raw.get("data")),
        calendar=_build(CalendarConfig, raw.get("calendar")),
        book=_build(BookConfig, raw.get("book")),
        correlation=_build(CorrelationConfig, raw.get("correlation")),
        sim=_build(SimConfig, raw.get("sim")),
        doomloop=_build(DoomLoopConfig, raw.get("doomloop")),
        labor=_build(LaborConfig, raw.get("labor")),
        policy=_build(PolicyConfig, raw.get("policy")),
        recovery=_build(RecoveryConfig, raw.get("recovery")),
    )
