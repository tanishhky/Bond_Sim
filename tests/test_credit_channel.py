"""Credit-channel transmission (decision 0005, P-20): the sovereign-risk
pass-through alternative to the VAR's own lagged-10y coefficients."""
import numpy as np
import pandas as pd
import pytest

from bond_sim.config import load
from bond_sim.sim import DoomLoopSimulator, InitialState, LinearPremium, MacroVAR, VARBlock, policy_from_name
from bond_sim.sim.macro import VARS

H = 120
K = 200


def synthetic_quarterly(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    V = len(VARS)
    A = np.diag([0.93, 0.90, 0.50, 0.85, 0.92])
    A[2, 0] = -0.4
    A[4, 2] = -0.05
    corr = np.eye(V)
    corr[0, 1] = corr[1, 0] = 0.7
    corr[2, 4] = corr[4, 2] = -0.5
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


def test_credit_channel_transmission_shape_and_zero_rows(var):
    d = np.array([0.0, 1.0, 2.0])
    out = var.credit_channel_transmission(d)
    assert out.shape == (3, len(VARS))
    i = {v: var.index_of(v) for v in VARS}
    assert np.allclose(out[:, i["r10"]], 0.0)
    assert np.allclose(out[:, i["r3m"]], 0.0)
    assert np.allclose(out[:, i["pb"]], 0.0)


def test_credit_channel_transmission_sign_and_scaling(var):
    d = np.array([1.0, 2.0])   # 100bp, 200bp
    out = var.credit_channel_transmission(d, output_elasticity=-0.006)
    i_g, i_u = var.index_of("g_nom"), var.index_of("u")
    assert out[0, i_g] < 0                              # a positive premium hurts growth
    assert np.isclose(out[1, i_g], 2 * out[0, i_g])      # linear in the premium
    assert out[0, i_u] > 0                               # and raises log-unemployment


def test_credit_channel_band_low_high_bracket_central():
    var = MacroVAR(max_lag=2).fit(synthetic_quarterly())
    d = np.array([1.0])
    central = var.credit_channel_transmission(d, output_elasticity=-0.006)
    low = var.credit_channel_transmission(d, output_elasticity=-0.002)
    high = var.credit_channel_transmission(d, output_elasticity=-0.010)
    i_g = var.index_of("g_nom")
    assert low[0, i_g] > central[0, i_g] > high[0, i_g]   # -0.002 is the mildest, -0.010 the harshest


def test_varblock_rejects_unknown_transmission(var):
    with pytest.raises(ValueError):
        VARBlock(var, transmission="fed_hike_typo")


def test_varblock_describe_reflects_transmission_choice(var):
    assert "VAR-own-coefficients" in VARBlock(var, transmission="var").describe()
    assert "credit-channel" in VARBlock(var, transmission="credit").describe()


def test_credit_channel_changes_simulated_outcomes(var, init):
    """The whole point: switching transmission must actually change the
    simulated path, not just the label. Same draws, same premium, only the
    transmission differs. Doesn't assert a direction: this fixture's
    hand-built coefficient (A[2, 0] = -0.4) makes the "var" channel harsher
    here, and the real fitted VAR agrees in *instantaneous* ordering
    (rate_transmission's g_nom coefficient is ~-0.56 vs credit-central's
    -0.006, see `scripts/premium_transmission_comparison.py`), but how much
    of that gap survives 120 quarters of the VAR's own mean reversion is an
    empirical question about premium path and dynamics, not something a
    synthetic single-fixture unit test should assert a number for; the real
    side-by-side comparison (`docs/premium_transmission_comparison_findings.md`)
    found the simulated gap much smaller than the instantaneous one at the
    default premium slope."""
    cfg = load()
    premium = LinearPremium(bps_per_pct=50.0, anchor_pct=100.0)
    policy = policy_from_name("status_quo", cfg.policy)

    sim_var = DoomLoopSimulator(cfg, VARBlock(var, transmission="var"), premium, init, policy)
    sim_credit = DoomLoopSimulator(cfg, VARBlock(var, transmission="credit", credit_elasticity=-0.006),
                                   premium, init, policy)
    res_var = sim_var.run(K=K, seed=7)
    res_credit = sim_credit.run(K=K, seed=7)

    # same K and H regardless of admissibility, compare the full paths directly
    assert not np.allclose(res_var.paths["g_nom"], res_credit.paths["g_nom"])
    assert res_credit.summary()["median_end_g_nom"] != res_var.summary()["median_end_g_nom"]
