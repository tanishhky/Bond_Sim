"""Simulation layer on a synthetic levels VAR: shapes, sanity, premium ordering,
mean reversion, and common-random-number discipline. No network."""
import numpy as np
import pandas as pd
import pytest

from bond_sim.config import load
from bond_sim.sim import (DoomLoopSimulator, InitialState, LinearPremium, MacroVAR, NoPremium, VARBlock,
                          policy_from_name)
from bond_sim.sim.macro import VARS

H = 120                      # months
K = 200


def synthetic_quarterly(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Stable VAR(1) in levels with a known correlated shock structure."""
    rng = np.random.default_rng(seed)
    V = len(VARS)                                   # r10, r3m, g_nom, pb, u
    A = np.diag([0.93, 0.90, 0.50, 0.85, 0.92])
    A[2, 0] = -0.4                                  # higher 10y -> lower growth
    A[4, 2] = -0.05                                 # higher growth -> lower unemployment
    corr = np.eye(V)
    corr[0, 1] = corr[1, 0] = 0.7                   # 10y and 3m shocks move together
    corr[2, 4] = corr[4, 2] = -0.5                  # growth and unemployment shocks oppose
    sd = np.array([0.35, 0.45, 2.0, 0.5, 0.3])
    L = np.linalg.cholesky(corr * np.outer(sd, sd))
    mu = np.array([4.8, 3.5, 4.8, -2.5, 5.5])
    x = np.zeros((n, V))
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + A @ (x[t - 1] - mu) + L @ rng.standard_normal(V)
    return pd.DataFrame(x, columns=VARS, index=pd.date_range("1975-01-01", periods=n, freq="QS"))


@pytest.fixture(scope="module")
def var():
    return MacroVAR(max_lag=2).fit(synthetic_quarterly())


@pytest.fixture(scope="module")
def init():
    return InitialState(debt_public_mm=30e6, gdp_saar_mm=30e6, r10_pct=4.8, r3m_pct=4.0, u_pct=4.2,
                        pb_pct_gdp=-3.0, g_nom_pct=4.5,
                        legacy_interest_mm=np.full(H, 60_000.0), legacy_principal_mm=np.full(H, 300_000.0),
                        avg_new_maturity_months=70.0, bill_share=0.22)


def _run(var, init, premium, policy="status_quo", seed=1):
    cfg = load()
    sim = DoomLoopSimulator(cfg, VARBlock(var), premium, init, policy_from_name(policy, cfg.policy))
    return sim.run(K=K, seed=seed)


def test_var_fit_recovers_structure(var):
    c = var.resid_corr()
    assert c.loc["r10", "r3m"] > 0.5
    assert c.loc["g_nom", "u"] < -0.3
    assert var.max_eigenvalue() < 1.0
    lr = var.long_run_means()
    # persistent AR(0.93) processes have noisy sample means over 200 quarters; this is a sanity band
    assert abs(lr["u"] - 5.5) < 1.0 and abs(lr["r10"] - 4.8) < 1.0


def test_shapes_finiteness_and_mean_reversion(var, init):
    res = _run(var, init, NoPremium())
    for k, v in res.paths.items():
        assert v.shape == (K, H), k
        if k not in ("infl", "state"):
            assert np.isfinite(v).all(), k
    assert res.admissible.mean() > 0.9, res.rejections.to_dict()
    assert 0.0 <= res.trigger_probability() <= 1.0
    assert (res.paths["premium"] == 0).all()
    # levels VAR: the unemployment median stays near its long-run mean, not at a floor
    assert 3.0 < np.median(res.paths["u"][:, -1]) < 8.0
    assert 2.0 < np.median(res.paths["r10"][:, -1]) < 8.0


def test_premium_raises_debt_and_rates(var, init):
    a = _run(var, init, NoPremium(), seed=7)
    b = _run(var, init, LinearPremium(bps_per_pct=20.0, anchor_pct=90.0), seed=7)
    assert (b.paths["premium"] >= 0).all() and b.paths["premium"].max() > 0
    assert np.median(b.paths["debt_gdp"][:, -1]) > np.median(a.paths["debt_gdp"][:, -1])
    assert np.median(b.paths["r10"][:, -1]) > np.median(a.paths["r10"][:, -1])
    # the premium is on top of the base rate, which keeps its own dynamics
    np.testing.assert_allclose(b.paths["r10"], b.paths["r10_base"] + b.paths["premium"])


def test_common_random_numbers_and_policy_effect(var, init):
    p = LinearPremium(bps_per_pct=5.0, anchor_pct=90.0)
    sq1 = _run(var, init, p, "status_quo", seed=3)
    sq2 = _run(var, init, p, "status_quo", seed=3)
    np.testing.assert_array_equal(sq1.paths["debt_gdp"], sq2.paths["debt_gdp"])   # reproducible
    gl = _run(var, init, p, "growth_led", seed=3)
    assert np.median(gl.paths["debt_gdp"][:, -1]) < np.median(sq1.paths["debt_gdp"][:, -1])
    au = _run(var, init, p, "austerity", seed=3)
    assert np.median(au.paths["pb"][:, -1]) > np.median(sq1.paths["pb"][:, -1])
    nl = _run(var, init, p, "no_layoff_mandate", seed=3)
    assert np.isfinite(nl.paths["u"]).all()
