import numpy as np

from src import denysko as d
from src.fitting import preferred_tail_orientation
from src.topology import BoundaryPath, Corridor


def _corr(lower, upper, *, join_score=0, pref=None):
    xs = np.array([0.0, 0.5, 1.0])
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    center = 0.5 * (lower + upper)
    return Corridor(
        path=BoundaryPath(points=np.column_stack([xs, center]), contour_id=-1),
        xa=0.0, xb=1.0, xs=xs, lower=lower, upper=upper,
        ylo=0.0, yhi=1.0, ylo_local=float(lower.min()),
        yhi_local=float(upper.max()), join_score=join_score,
        preferred_orientation=pref,
    )


def test_joined_conflicting_endpoints_use_dominant_vertical_region():
    # Left endpoint is nearer the bottom; right endpoint nearer the top.
    # Without a junction join, preserve the ordinary opposite-tail rule.
    upper = _corr([0.10, 0.64, 0.70], [0.20, 0.76, 0.80])
    assert preferred_tail_orientation(upper) == (-1, 1)

    # The same geometry joined through a real junction belongs dominantly to
    # the upper half, so both unbounded tails escape upward.
    upper.join_score = 1
    assert preferred_tail_orientation(upper) == (1, 1)

    # Symmetric lower-region mechanism.
    lower = _corr([0.20, 0.24, 0.80], [0.30, 0.36, 0.90], join_score=1)
    assert preferred_tail_orientation(lower) == (-1, -1)


def test_component_preference_still_overrides_join_rule():
    c = _corr([0.10, 0.64, 0.70], [0.20, 0.76, 0.80],
              join_score=1, pref=(-1, -1))
    assert preferred_tail_orientation(c) == (-1, -1)


def test_e_upper_routes_escape_up_both_sides():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("e")
    orientations = [preferred_tail_orientation(c) for c in selected]
    assert sorted(orientations) == [(-1, -1), (1, 1), (1, 1)]

    # The lower route is the only unjoined/downward one. The two joined
    # routes are the upper curves and must both escape upward.
    joined = [c for c in selected if c.join_score > 0]
    assert len(joined) == 2
    assert all(preferred_tail_orientation(c) == (1, 1) for c in joined)


def test_unjoined_C_and_agreeing_A_are_unchanged():
    for letter, expected in {
        "C": [(-1, -1), (-1, 1)],
        "A": [(-1, -1), (-1, -1)],
    }.items():
        *_, selected = d.build_phase1(letter)
        got = sorted(preferred_tail_orientation(c) for c in selected)
        assert got == sorted(expected)
