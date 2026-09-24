"""Signature-figure pipeline (thesis charter A9): versioning, manifest, headline gate, two-layer band."""
import json

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from bond_sim.figures import FigureSpec, envelope, headline, publish, two_layer_band

CLEAN = {"commit": "abc1234", "tag": "SDL-N-009", "dirty": False}
META = {"as_of": "2026-09-30", "config_hash": "d9222bd624fd", "variants": ["var", "states"], "K": 1000, "seed": 7}


def claims(status):
    return {"C-001": {"id": "C-001", "kind": "result", "status": status,
                      "wording": "Exact approved wording.", "conditions": "Under A1-A3."}}


def spec(claim=None):
    return FigureSpec(id="F-01", node="N-009", title="P(entered by year t), status quo", source="FRED/ALFRED", claim=claim)


def draw(fig, ax, data, render):
    x = data.index.values
    env = envelope({"var": data["var"], "states": data["states"]})
    two_layer_band(ax, x, data["center"], data["lo"], data["hi"], env, label="Status quo")


@pytest.fixture
def data():
    t = np.arange(2026, 2056)
    c = 1 - np.exp(-(t - 2025) / 8)
    return pd.DataFrame({"center": c, "lo": c * 0.95, "hi": np.minimum(c * 1.05, 1),
                         "var": c * 0.9, "states": np.minimum(c * 1.1, 1)}, index=t)


def test_versions_never_overwrite(tmp_path, data):
    v1 = publish(spec(), draw, data, META, out_root=tmp_path, claims={}, git=CLEAN)
    v2 = publish(spec(), draw, data, META, out_root=tmp_path, claims={}, git=CLEAN)
    assert (v1.name, v2.name) == ("v1", "v2")
    m = json.loads((v1 / "manifest.json").read_text())
    assert m["git"]["tag"] == "SDL-N-009" and m["run"]["config_hash"] == "d9222bd624fd"
    assert set(m["files"]) == {"data.csv", "F-01_v1_paper.pdf", "F-01_v1_slide.png"}
    assert "Not claimable" in m["caption"]


def test_dirty_tree_only_drafts(tmp_path, data):
    dirty = {**CLEAN, "dirty": True}
    with pytest.raises(RuntimeError):
        publish(spec(), draw, data, META, out_root=tmp_path, claims={}, git=dirty)
    d = publish(spec(), draw, data, META, out_root=tmp_path, claims={}, git=dirty, draft=True)
    assert d.parent.parent.name == "_draft"


def test_model_range_required(tmp_path, data):
    with pytest.raises(ValueError):
        publish(spec(), draw, data, {**META, "variants": ["var"]}, out_root=tmp_path, claims={}, git=CLEAN)
    with pytest.raises(ValueError):
        envelope({"var": [1.0, 2.0]})


@pytest.mark.parametrize("status,slide,social_ok", [
    ("draft", "P(entered by year t), status quo", False),
    ("internal", "Exact approved wording.", False),
    ("public", "Exact approved wording.", True),
])
def test_headline_gate(status, slide, social_ok):
    s, c = spec("C-001"), claims(status)
    assert headline(s, "paper", c) is None
    assert headline(s, "slide", c) == slide
    if social_ok:
        assert headline(s, "social", c) == "Exact approved wording."
    else:
        with pytest.raises(ValueError):
            headline(s, "social", c)


def test_retracted_claim_blocks_every_render():
    for r in ("paper", "slide"):
        with pytest.raises(ValueError):
            headline(spec("C-001"), r, claims("retracted"))
    with pytest.raises(ValueError):
        headline(spec("C-404"), "slide", claims("public"))


def test_social_refused_before_writing(tmp_path, data):
    with pytest.raises(ValueError):
        publish(spec("C-001"), draw, data, META, renders=("paper", "social"), out_root=tmp_path,
                claims=claims("internal"), git=CLEAN)
    assert not (tmp_path / "F-01").exists()


def test_envelope_drivers_and_outer_contains_inner():
    env = envelope({"persist_6m": [0.9, 1.0], "persist_24m": [0.2, 0.5], "var": [0.5, 0.8]})
    assert list(env.lo) == [0.2, 0.5] and list(env.hi) == [0.9, 1.0]
    assert env.drivers() == {"lower": "persist_24m", "upper": "persist_6m"}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    two_layer_band(ax, np.array([0, 1]), [0.6, 0.7], [0.1, 0.6], [0.95, 0.9], env, label="x")
    outer = ax.collections[0].get_paths()[0].vertices[:, 1]
    assert outer.min() <= 0.1 and outer.max() >= 1.0
    plt.close(fig)
