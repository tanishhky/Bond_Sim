"""Block factors, latent chain, and the state-driven debt engine on synthetic data. No network."""
import numpy as np
import pandas as pd
import pytest

from bond_sim.analysis.factors import BlockFactors
from bond_sim.config import load
from bond_sim.sim import (DoomLoopSimulator, FactorImpulse, ForcedTransition, InitialState, LatentStateModel,
                          NoPremium, StateBlock, policy_from_name)

BLOCKS = {
    "consumer": {"anchor": ("PCE", +1), "series": [("PCE", "log_diff"), ("PSAVERT", "level")]},
    "corporate": {"anchor": ("GDP", +1), "series": [("GDP", "log_diff"), ("INDPRO", "log_diff")]},
    "financial": {"anchor": ("NFCI", -1), "series": [("NFCI", "level"), ("BAA10Y", "level")]},
    "fiscal": {"anchor": ("pb", +1), "series": [("pb", "level"), ("debt_gdp", "diff")]},
    "labor": {"anchor": ("UNRATE", -1), "series": [("UNRATE", "log"), ("PAYEMS", "log_diff")]},
    "rates": {"anchor": ("DGS10", +1), "series": [("DGS10", "level"), ("DGS3MO", "level")]},
    "prices": {"anchor": ("CPIAUCSL", +1), "series": [("CPIAUCSL", "log_diff")]},
}
H = 96


def synthetic_panel(n=220, seed=1) -> pd.DataFrame:
    """Two-regime economy: a latent cycle z drives growth-type series; levels random-walk gently."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1970-01-01", periods=n, freq="QS")
    z = np.zeros(n); s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < 0.9 else 1 - s[t - 1]
        z[t] = 0.7 * z[t - 1] + (0.5 if s[t] else -0.8) + 0.5 * rng.standard_normal()
    def growth(level0, drift, beta, sd):
        g = drift + beta * z + sd * rng.standard_normal(n)
        return level0 * np.exp(np.cumsum(g) / 100.0)
    df = pd.DataFrame(index=idx)
    df["PCE"] = growth(1000, 1.2, 0.3, 0.4); df["PSAVERT"] = 7 - 0.5 * z + rng.standard_normal(n) * 0.5
    df["GDP"] = growth(3000, 1.3, 0.4, 0.4); df["INDPRO"] = growth(50, 0.6, 0.6, 0.8)
    df["NFCI"] = -0.3 - 0.4 * z + 0.3 * rng.standard_normal(n); df["BAA10Y"] = 2.5 - 0.3 * z + 0.2 * rng.standard_normal(n)
    df["UNRATE"] = np.clip(6 - 0.8 * np.convolve(z, np.ones(3) / 3, "same") + 0.3 * rng.standard_normal(n), 2, 12)
    df["PAYEMS"] = growth(90000, 0.4, 0.3, 0.2)
    df["DGS10"] = np.clip(5 + 0.3 * z + np.cumsum(0.15 * rng.standard_normal(n)), 0.5, 15)
    df["DGS3MO"] = np.clip(df["DGS10"] - 1 + 0.2 * rng.standard_normal(n), 0.1, 15)
    df["CPIAUCSL"] = growth(100, 0.8, 0.1, 0.3)
    df["pb"] = -1.5 + 0.4 * z + 0.5 * rng.standard_normal(n)
    df["debt_gdp"] = 60 + np.cumsum(0.4 - 0.1 * z + 0.3 * rng.standard_normal(n))
    return df


@pytest.fixture(scope="module")
def fitted():
    q = synthetic_panel()
    bf = BlockFactors(BLOCKS, start="1975-01-01").fit(q)
    model = LatentStateModel(candidates=(2, 3), prior_strength=2.0, seed=0).fit(bf.factors())
    return q, bf, model


def test_block_factors_structure(fitted):
    q, bf, _ = fitted
    F = bf.factors()
    assert list(F.columns) == list(BLOCKS)
    np.testing.assert_allclose(F.std(ddof=0), 1.0, atol=0.15)
    # anchors load with the stated sign
    assert bf.fits["financial"].loadings[0] < 0 and bf.fits["labor"].loadings[0] < 0 and bf.fits["consumer"].loadings[0] > 0
    assert all(0 < f.explained <= 1 for f in bf.fits.values())


def test_latent_chain_is_a_proper_markov_model(fitted):
    _, _, m = fitted
    assert m.k in (2, 3)
    np.testing.assert_allclose(m.P.sum(axis=1), 1.0)
    assert (m.P >= 0).all() and np.abs(np.linalg.eigvals(m.A)).max() < 1.05
    # state 0 is the weakest and tightest by construction
    act = [m.blocks.index(b) for b in ("consumer", "corporate", "labor", "financial")]
    assert m.means[0, act].sum() <= m.means[-1, act].sum()
    assert m.hazard_beta >= 0.0
    rows = m.transition_row(np.array([1, 1, 0]), np.array([-2.0, 2.0, 0.0]))
    np.testing.assert_allclose(rows.sum(axis=1), 1.0)
    if m.hazard_beta > 0:
        assert rows[0, 0] >= rows[1, 0]        # tighter conditions -> higher stress hazard


def test_state_block_drives_the_debt_engine(fitted):
    _, bf, m = fitted
    cfg = load()
    init = InitialState(debt_public_mm=30e6, gdp_saar_mm=30e6, r10_pct=4.8, r3m_pct=4.0, u_pct=4.2,
                        pb_pct_gdp=-3.0, g_nom_pct=4.5, legacy_interest_mm=np.full(H, 60_000.0),
                        legacy_principal_mm=np.full(H, 300_000.0), avg_new_maturity_months=70.0, bill_share=0.22)
    block = StateBlock(bf, m)
    res = DoomLoopSimulator(cfg, block, NoPremium(), init, policy_from_name("status_quo", cfg.policy)).run(K=150, seed=2)
    assert res.paths["debt_gdp"].shape == (150, H)
    assert res.admissible.mean() > 0.8, res.rejections.to_dict()
    assert np.isfinite(res.paths["infl"][res.admissible]).all()
    assert set(np.unique(res.paths["state"][:, 2::3])) <= set(range(m.k))
    # common random numbers: identical draws reproduce; a forced recession changes the state path
    draws = block.prepare(150, (H + 2) // 3, np.random.default_rng(9))
    a = DoomLoopSimulator(cfg, StateBlock(bf, m), NoPremium(), init).run(K=150, draws=draws)
    b = DoomLoopSimulator(cfg, StateBlock(bf, m), NoPremium(), init).run(K=150, draws=draws)
    np.testing.assert_array_equal(a.paths["debt_gdp"], b.paths["debt_gdp"])
    shocked = StateBlock(bf, m, shocks=[ForcedTransition(state=0, at=0, duration=4, label="recession now"),
                                        FactorImpulse(block="financial", size=-1.5, at=0, decay=0.7)])
    c = DoomLoopSimulator(cfg, shocked, NoPremium(), init).run(K=150, draws=draws)
    assert (c.paths["state"][:, 2] == 0).all()
    assert np.median(c.paths["u"][c.admissible, 11]) >= np.median(a.paths["u"][a.admissible, 11]) - 0.5
