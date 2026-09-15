"""The accounting identity and the trigger (docs/math.md 1-4), on numbers checkable by hand."""
import numpy as np
import pandas as pd

from bond_sim.sim.sustainability import (FeasiblePB, decompose, evaluate_history, evaluate_paths, first_run_index,
                                         stabilizing_growth, stabilizing_primary_balance)


def test_stabilizing_growth_closed_form():
    # d = 100%, r = 4%, pb = -2%: annual approximation says ~6.1%; exact quarterly compounding is close
    g = stabilizing_growth(np.array([1.0]), np.array([0.04]), np.array([-0.02]))
    assert 0.058 < g[0] < 0.064
    # with a balanced primary budget, g* = r exactly
    np.testing.assert_allclose(stabilizing_growth(np.array([1.0]), np.array([0.04]), np.array([0.0])), 0.04, rtol=1e-9)


def test_stabilizing_primary_balance_signs():
    pb = stabilizing_primary_balance(np.array([1.0, 1.0]), np.array([0.05, 0.03]), np.array([0.04, 0.04]))
    assert pb[0] > 0 > pb[1]             # r > g needs a surplus; r < g tolerates a deficit


def test_decomposition_adds_up():
    idx = pd.date_range("2000-01-01", periods=8, freq="QS")
    d = pd.Series([1.0, 1.01, 1.03, 1.02, 1.05, 1.09, 1.10, 1.16], index=idx)
    r = pd.Series(0.04, index=idx); g = pd.Series(0.03, index=idx); pb = pd.Series(-0.02, index=idx)
    dec = decompose(d, r, g, pb)
    np.testing.assert_allclose((dec["snowball"] + dec["primary"] + dec["residual"]).iloc[1:], dec["delta_d"].iloc[1:])
    assert dec["delta3_d"].notna().sum() == 5


def test_first_run_index():
    m = np.array([[0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0]], bool)
    np.testing.assert_array_equal(first_run_index(m, 3), [1, -1, -1])
    np.testing.assert_array_equal(first_run_index(m, 2), [1, 2, -1])


def test_trigger_on_paths_and_history():
    K, H = 3, 48
    d = np.tile(np.linspace(1.0, 1.5, H), (K, 1))
    r = np.full((K, H), 0.05); g = np.full((K, H), 0.03); pb = np.full((K, H), -0.03)
    r[2] = 0.02                                                     # path 2: r < g, never breaches
    feas = FeasiblePB(envelope=0.01, reaction=None)
    diag = evaluate_paths(d, r, g, pb, feas, n_periods=6, which="envelope")
    assert diag.trigger_index[0] >= 0 and diag.trigger_index[1] >= 0 and diag.trigger_index[2] == -1
    assert 0.0 < diag.probability() < 1.0
    q = pd.DataFrame({"d": d[0][:12], "r_eff": r[0][:12], "g_nom": g[0][:12], "pb": pb[0][:12], "u": 5.0},
                     index=pd.date_range("2010-01-01", periods=12, freq="QS"))
    hist = evaluate_history(q, feas, n_quarters=4)
    assert hist["breach"].iloc[1:].all() and hist["triggered"].iloc[-1]
