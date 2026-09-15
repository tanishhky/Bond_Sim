"""Problems 1-5 from main.ipynb, pinned. If any of these move, the engine no
longer computes what the notebook hand-verified."""
import numpy as np

from bond_sim.engine import BondBook, flat_scenario_factors, present_value

ISSUE, T, C, F = [0, 2, 4], [2, 3, 1], [0.02, 0.04, 0.035], [100, 100, 100]


def _book():
    return BondBook.from_toy(ISSUE, T, C, F)


def test_problem1_active_mask():
    exp = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 0, 0]], bool)
    assert _book().M == 9
    np.testing.assert_array_equal(_book().dense_active(), exp)


def test_problem2_flows_and_outstanding():
    b = _book()
    coupon = np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 2, 2, 2, 2, 2, 2],
                       [0, 0, 0, 0, 0, 1.75, 1.75, 0, 0]])
    principal = np.zeros((3, 9)); principal[0, 4] = principal[1, 8] = principal[2, 6] = 100
    outstanding = np.array([[100, 100, 100, 100, 0, 0, 0, 0, 0],
                            [0, 0, 100, 100, 100, 100, 100, 100, 0],
                            [0, 0, 0, 0, 100, 100, 0, 0, 0]], float)
    np.testing.assert_allclose(b.dense_coupon_flow(), coupon)
    np.testing.assert_allclose(b.dense_principal_flow(), principal)
    np.testing.assert_allclose(b.dense_outstanding(), outstanding)


def test_problem3_column_sums_sparse_equals_dense():
    b = _book()
    agg = b.aggregates(by_kind=False).frame
    np.testing.assert_allclose(agg["outstanding"], [100, 100, 200, 200, 200, 200, 100, 100, 0])
    np.testing.assert_allclose(agg["interest"], [0, 1, 1, 3, 3, 3.75, 3.75, 2, 2])
    np.testing.assert_allclose(agg["debt_service"], [0, 1, 1, 3, 103, 3.75, 103.75, 2, 102])
    # the O(pairs) path must equal the dense reduction exactly
    np.testing.assert_allclose(agg["outstanding"], b.dense_outstanding().sum(0))
    np.testing.assert_allclose(agg["interest"], b.dense_coupon_flow().sum(0))
    np.testing.assert_allclose(agg["principal"], b.dense_principal_flow().sum(0))


def test_problem4_flat_discount_factors():
    df = flat_scenario_factors([0.02, 0.04, 0.06], M=9, periods_per_year=2)
    assert df.shape == (3, 9)
    np.testing.assert_allclose(df[:, 0], 1.0)
    np.testing.assert_allclose(df[:, 1], [1 / 1.01, 1 / 1.02, 1 / 1.03])
    np.testing.assert_allclose(df[2, 8], 1 / 1.03 ** 8)


def test_problem5_sensitivity_profile():
    b = _book()
    cf = b.aggregates(by_kind=False).frame["debt_service"].to_numpy()
    pv = present_value(flat_scenario_factors([0.02, 0.04, 0.06], M=9), cf)
    np.testing.assert_allclose(pv, [301.22905527, 284.24526648, 268.4427601], rtol=1e-9)
    assert pv[0] > pv[1] > pv[2], "lower rates must give higher PV"
