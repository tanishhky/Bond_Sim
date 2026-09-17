"""Real-time trigger evaluation (decision 0002 follow-up).

Three properties, on a synthetic quarterly history so it runs offline:
1. With unrevised data and an envelope fixed from the first quarter, the
   real-time table equals the hindsight table row for row.
2. No look-ahead: a revision published after as_of cannot change the
   real-time result at as_of.
3. The hindsight mechanism itself: when the envelope-defining surplus comes
   late in the sample, today's vintage says "no breach" for quarters that, in
   real time, breached. The two tables must disagree somewhere.
"""
import numpy as np
import pandas as pd

from bond_sim.sim import evaluate_history, evaluate_history_realtime, compare_realtime_final, FeasiblePB


def _history(n=40, r=0.08, g=0.04, d=0.9, pb=-0.03, surplus_at=None, surplus=0.02):
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    h = pd.DataFrame({"d": d, "r_eff": r, "g_nom": g, "pb": pb, "u": 5.0}, index=idx)
    if surplus_at is not None:
        h.loc[idx[surplus_at], "pb"] = surplus
    return h


def _store(full, revised_at=None, revise_quarter=None, revised_g=0.10):
    """history_fn(as_of): the frame as known on as_of. If revised_at is given,
    vintages from that date on carry a revised g_nom for revise_quarter."""
    def fn(as_of):
        h = full.loc[full.index <= as_of].copy()
        if revised_at is not None and as_of >= pd.Timestamp(revised_at):
            h.loc[full.index[revise_quarter], "g_nom"] = revised_g
        return h
    return fn


def test_realtime_equals_hindsight_when_nothing_is_revised_and_envelope_is_fixed_early():
    full = _history(surplus_at=0)            # the max pb is the first quarter: expanding envelope is constant
    dates = full.index[1:]                   # start early enough that the first rows are not yet triggered
    rt = evaluate_history_realtime(_store(full), dates, n_quarters=4)
    final = evaluate_history(full, FeasiblePB.from_history(full["pb"], full["d"], full["u"], fit_reaction=False), n_quarters=4)
    cmp = compare_realtime_final(rt, final)
    assert len(cmp) == len(dates)
    assert cmp["agree"].all()
    assert cmp["triggered_final"].any() and not cmp["triggered_final"].all()   # non-trivial: mixes True and False


def test_no_lookahead_a_later_revision_cannot_change_an_earlier_as_of_row():
    full = _history(surplus_at=0)
    dates = full.index[4:]
    a = evaluate_history_realtime(_store(full), dates, n_quarters=4)
    b = evaluate_history_realtime(_store(full, revised_at="2005-01-01", revise_quarter=2), dates, n_quarters=4)
    before = a.index < pd.Timestamp("2005-01-01")
    pd.testing.assert_frame_equal(a[before], b[before])


def test_hindsight_understates_breaches_when_the_envelope_surplus_comes_late():
    full = _history(r=0.05, g=0.04, surplus_at=30)   # pb* ~ 0.0087; real-time envelope -0.03, hindsight 0.02
    dates = full.index[4:]
    rt = evaluate_history_realtime(_store(full), dates, n_quarters=4)
    final = evaluate_history(full, FeasiblePB.from_history(full["pb"], full["d"], full["u"], fit_reaction=False), n_quarters=4)
    cmp = compare_realtime_final(rt, final)
    early = cmp.index < full.index[30]
    assert cmp.loc[early, "triggered_realtime"].any()
    assert not cmp.loc[early, "triggered_final"].any()
    assert not cmp["agree"].all()
