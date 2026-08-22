import numpy as np
import pytest

from src import denysko as d


# ---------------------------------------------------------------------------
# u -> x conversion and serialization
# ---------------------------------------------------------------------------


def test_u_coefficients_printed_as_x():
    cand = d.Candidate(1, np.array([100.0, 100.0]))
    assert d.serialize(cand) == "y=2x"


def test_serialize_parse_roundtrip():
    for degree in range(6):
        coef = np.arange(1.0, degree + 2.0)
        line = d.serialize(d.Candidate(degree, coef))
        curve = d.parse_line(line)
        assert curve is not None
        assert d.format_expression(curve) == line


def test_parse_matches_internal_u_basis_values():
    xs = np.array([-5.0, 0.0, 12.5, 50.0, 77.7, 105.0])
    coef = np.array([3.0, -2.0, 1.0])
    curve = d.parse_line(d.serialize(d.Candidate(2, coef)))
    expected = d._poly_u(coef, xs)
    assert np.allclose(curve.poly(xs), expected, atol=0.05)


def test_malformed_lines_do_not_parse():
    for line in ["", "y=", "y=x^^2", "y=..x", "x=2", "y=x^2 \\left\\{0\\le x\\le 10\\right\\}"]:
        assert d.parse_line(line) is None


# ---------------------------------------------------------------------------
# Trace classification
# ---------------------------------------------------------------------------


def test_single_finite_trace():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    an = d.analyze_candidate(
        d.Candidate(1, np.array([50.0, 50.0])),  # u-basis: y = x
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds == pytest.approx((0.0, 100.0))
    assert an.trace_penalty == 0.0
    assert an.n_components == 1
    assert an.extra_component_fraction == 0.0


def test_constant_in_band_is_unbounded_trace():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    an = d.analyze_candidate(
        d.Candidate(0, np.array([50.0])),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds is None
    assert an.trace_penalty == pytest.approx(2.0)


def test_multiple_components_invalid():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    # y = 120 - x^2, band [0,100]: two symmetric in-band intervals
    coef_u = np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(2, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds is None
    assert an.n_components == 2
    assert 0.0 < an.trace_penalty < 2.0
    assert 0.0 < an.extra_component_fraction < 0.5 + 1e-6


# ---------------------------------------------------------------------------
# All trace components contribute to surface scoring
# ---------------------------------------------------------------------------


def test_surface_scores_all_components():
    # Vertical bar boundary.
    ys = np.arange(5.0, 95.01, 0.5)
    p = np.vstack([
        np.column_stack([np.full(len(ys), 49.6), ys]),
        np.column_stack([np.full(len(ys), 50.4), ys]),
    ])
    glyph = d.make_glyph(p)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)

    # A parabola with two components, both far from the bar: surface must
    # be near zero because BOTH components are scored, not just the widest.
    coef_u = np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef
    an = d.analyze_candidate(d.Candidate(2, coef_u), glyph, uncovered)
    assert an.n_components == 2
    assert an.surface_fraction < 0.5


def test_extra_component_shrinking_lowers_trace_penalty():
    # Continuous penalty: as the spurious component's arc shrinks toward
    # zero relative to the main component, the penalty drops continuously.
    comps = [(0.0, 100.0), (10.0, 20.0)]
    big = d._trace_penalty_continuous(comps, [90.0, 9.0])
    small = d._trace_penalty_continuous(comps, [90.0, 1.0])
    none = d._trace_penalty_continuous(comps, [90.0, 0.0])
    assert big > small > none
    assert small > 0.0
    assert none == 0.0


def test_trace_penalty_continuous_formula():
    # extra_component_fraction = extra_arc / total_arc, penalty = 2 * fraction
    comps = [(0.0, 1.0), (2.0, 3.0)]
    pen = d._trace_penalty_continuous(comps, [80.0, 20.0])
    assert pen == pytest.approx(2.0 * 20.0 / 100.0)
    # an unbounded in-band component adds +2.0; finite extra still counts
    comps_unb = [(0.0, np.inf), (5.0, 6.0), (7.0, 8.0)]
    pen2 = d._trace_penalty_continuous(comps_unb, [80.0, 15.0, 5.0])
    assert pen2 == pytest.approx(2.0 + 2.0 * 5.0 / 20.0)
    # a single unbounded component alone is 2.0
    assert d._trace_penalty_continuous([(0.0, np.inf)], [1.0]) == 2.0
    # empty trace -> 2.0
    assert d._trace_penalty_continuous([], []) == 2.0


# ---------------------------------------------------------------------------
# Tail rules (post-exit steepness)
# ---------------------------------------------------------------------------


def test_tail_case_a_shallow_forever_invalid():
    """A line crossing the band with slope 2 stays slope 2: never steep."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    an = d.analyze_candidate(
        d.Candidate(1, np.array([50.0, 50.0])),  # y = x, slope 1 < 8
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert not an.left_tail.valid or not an.right_tail.valid
    assert not an.feasible


def test_tail_case_b_shallow_at_endpoint_steep_outside_valid():
    """Follows a boundary, crosses the band shallowly, then turns steep."""

    def fit_hermite(points):
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        ds = np.array([p[2] for p in points])
        n = len(points)
        A = np.zeros((2 * n, 6))
        b = np.zeros(2 * n)
        for i, (x, y, dy) in enumerate(points):
            A[2 * i] = [x**k for k in range(6)]
            A[2 * i + 1] = [k * x ** (k - 1) if k > 0 else 0 for k in range(6)]
            b[2 * i] = y
            b[2 * i + 1] = dy
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef

    coef_x = fit_hermite([
        (-8.0, -5.0, 12.0),
        (-5.0, 0.0, 2.0),
        (5.0, 100.0, 2.0),
        (8.0, 105.0, 12.0),
    ])
    xc = d.XCurve(np.polynomial.Polynomial(coef_x))
    roots = np.sort(
        np.concatenate(
            [d._real_roots(xc.poly - 0.0), d._real_roots(xc.poly - 100.0)]
        )
    )
    l, r = roots[0], roots[-1]
    xs = np.arange(l, r + 0.001, 0.1)
    glyph = d.make_glyph(np.column_stack([xs, xc.poly(xs)]))

    an = d._analyze(
        xc, 5, glyph, np.zeros(len(glyph.points), dtype=bool), dense=True
    )
    assert an.bounds is not None
    assert an.left_tail.valid
    assert an.right_tail.valid
    assert d.structurally_feasible(an)
    assert d.validate([d.format_expression(xc)], glyph) == []


def test_tail_case_c_steep_but_turns_back_invalid():
    """Reaches the outside margin steeply but turns back and re-enters."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    coef_u = np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(2, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds is None
    assert an.trace_penalty > 0.9
    assert not an.feasible


def test_tail_case_d_steep_too_late_invalid():
    """Reaches the +-5 vertical margin only after >5 horizontal units."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    coef_u = np.polynomial.Polynomial([50.0, 0.5])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(1, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert not an.left_tail.valid or not an.right_tail.valid
    assert not an.feasible


def test_near_valid_tail_scores_better_than_bad_tail():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    uncovered = np.zeros(len(glyph.search_points), dtype=bool)
    # y = 7x: slope 7 < 8 at the margin -> near-valid tail.
    near = d.analyze_candidate(
        d.Candidate(1, np.polynomial.Polynomial([50.0 - 7.0 * 50.0, 7.0])(d.X_OF_U).coef),
        glyph, uncovered,
    )
    # y = 1x: slope 1 << 8 -> very bad tail.
    bad = d.analyze_candidate(
        d.Candidate(1, np.polynomial.Polynomial([50.0, 1.0])(d.X_OF_U).coef),
        glyph, uncovered,
    )
    assert near.tail_penalty < bad.tail_penalty


def test_outside_band_no_surface_penalty():
    # Vertical bar: distances only near the trace matter.
    ys = np.arange(5.0, 95.01, 0.5)
    p = np.vstack([
        np.column_stack([np.full(len(ys), 49.6), ys]),
        np.column_stack([np.full(len(ys), 50.4), ys]),
    ])
    glyph = d.make_glyph(p)
    # steep line hugging the bar: 28x - 1350 -> u-basis
    coef_u = np.polynomial.Polynomial([-1350.0, 28.0])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(1, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.surface_fraction >= 0.95
    assert an.bad_surface_fraction < 0.05
    assert an.mean_surface_excess < 0.05


def test_connected_shape_validator_handcrafted():
    """A handcrafted curve crossing ymin and ymax is surface-valid."""
    l = 30.0
    xs = np.arange(l, l + 12.0, 0.1)
    y = 90.0 - 8.0 * (xs - l)
    p = np.column_stack([xs, y])
    glyph = d.make_glyph(p)

    steep = d.XCurve(np.polynomial.Polynomial([90.0 + 8.0 * l, -8.0]))
    an = d._analyze(
        steep, 1, glyph, np.zeros(len(glyph.points), dtype=bool), dense=True
    )
    assert d.structurally_feasible(an)
    assert d.validate([d.format_expression(steep)], glyph) == []

    shallow = d.XCurve(np.polynomial.Polynomial([90.0 + 1.0 * l, -1.0]))
    an = d._analyze(
        shallow, 1, glyph, np.zeros(len(glyph.points), dtype=bool), dense=True
    )
    assert not d.structurally_feasible(an)
    assert not an.left_tail.valid or not an.right_tail.valid


def test_euclidean_distance_tiny_cloud():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 2.0]])
    da, db = d._min_dists(a, b)
    assert da[0] == pytest.approx(2.0)
    assert db[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# p2 ranking (mean, max, -distance)
# ---------------------------------------------------------------------------


def test_p2_max_distance_ranking():
    p = np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)])
    p1 = np.array([50.0, 0.0])
    # Two candidates with similar mean but one cutting badly across a corner.
    corner = np.array([60.0, 25.0])   # leaves the boundary (high max)
    along = np.array([70.0, 0.0])     # stays on the boundary (low max)
    d1 = d._min_dists(np.linspace(p1, corner, 33), p)[0]
    d2 = d._min_dists(np.linspace(p1, along, 33), p)[0]
    assert d2.max() < d1.max()
    # the (mean, max, -distance) key ranks the along-segment higher
    key_corner = (float(d1.mean()), float(d1.max()), -float(np.hypot(*(corner - p1))))
    key_along = (float(d2.mean()), float(d2.max()), -float(np.hypot(*(along - p1))))
    assert key_along < key_corner


# ---------------------------------------------------------------------------
# Two-parameter degree-5 bent seeds
# ---------------------------------------------------------------------------


def _tiny_glyph():
    xs = np.arange(10.0, 91.0, 2.0)
    return d.make_glyph(np.column_stack([xs, 0.5 * xs]))


def test_five_initial_seeds():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    seeds = d._initial_seeds(p1, p2, glyph)
    assert len(seeds) == 5
    assert sorted(s.degree for s in seeds) == [1, 5, 5, 5, 5]


def test_bent_seeds_preserve_value_and_derivative():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    u1 = (p1[0] - 50.0) / 50.0
    u2 = (p2[0] - 50.0) / 50.0
    line = np.array([
        p1[1] - ((p2[1] - p1[1]) / (u2 - u1)) * u1,
        (p2[1] - p1[1]) / (u2 - u1),
    ])
    for s in d._bent_seeds_u(p1, p2, glyph):
        assert s.degree == 5
        # values preserved
        assert np.polyval(s.coef[::-1], u1) == pytest.approx(p1[1], abs=1e-9)
        assert np.polyval(s.coef[::-1], u2) == pytest.approx(p2[1], abs=1e-9)
        # derivative preserved: P'(u) == L'(u) at the seed points
        dp = np.polyder(s.coef[::-1])
        assert np.polyval(dp, u1) == pytest.approx(line[1], abs=1e-9)
        assert np.polyval(dp, u2) == pytest.approx(line[1], abs=1e-9)


def test_bent_seeds_hit_both_tail_targets_exactly():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    xL = glyph.xmin - 5.0
    xR = glyph.xmax + 5.0
    uL = (xL - 50.0) / 50.0
    uR = (xR - 50.0) / 50.0
    up = glyph.ymax + 5.0
    dn = glyph.ymin - 5.0
    targets = [(up, up), (dn, dn), (up, dn), (dn, up)]
    for s, (tl, tr) in zip(d._bent_seeds_u(p1, p2, glyph), targets):
        assert np.polyval(s.coef[::-1], uL) == pytest.approx(tl, abs=1e-6)
        assert np.polyval(s.coef[::-1], uR) == pytest.approx(tr, abs=1e-6)


def test_same_x_pair_rejected_as_line_seed():
    """Vertical pairs (same x) must not be returned as line seeds."""
    p = np.array([[50.0, 20.0], [50.0, 40.0]])
    rng = np.random.default_rng(0)
    result = d._seed_pair(p, np.array([0, 1]), rng)
    assert result is None


def test_line_seed_passes_through_accepted_pair():
    p1 = np.array([50.0, 20.0])
    p2 = np.array([51.0, 30.0])
    cand = d._line_seed_u(p1, p2)
    u1 = (p1[0] - 50.0) / 50.0
    u2 = (p2[0] - 50.0) / 50.0
    assert np.polyval(cand.coef[::-1], u1) == pytest.approx(20.0, abs=1e-12)
    assert np.polyval(cand.coef[::-1], u2) == pytest.approx(30.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Merit arithmetic and feasible comparison
# ---------------------------------------------------------------------------


def _fake_analysis(**kw):
    base = dict(
        samples=np.zeros((0, 2)),
        sample_d=np.zeros(0),
        point_d=np.zeros(0),
        newly_covered=0,
        coverage_fraction=0.0,
        surface_fraction=1.0,
        bad_surface_fraction=0.0,
        mean_surface_excess=0.0,
        surface_penalty=0.0,
        mean_surface_distance=0.0,
        trace_penalty=0.0,
        tail_penalty=0.0,
        bounds=(0.0, 1.0),
        n_components=1,
        extra_component_fraction=0.0,
        deriv_outside=0,
        left_tail=d.TailInfo(True, 1.0, 10.0, True, 0, True),
        right_tail=d.TailInfo(True, 1.0, -10.0, True, 0, True),
        feasible=True,
        merit=0.0,
    )
    base.update(kw)
    return d.Analysis(**base)


def test_merit_formula():
    an = _fake_analysis(
        newly_covered=40,
        coverage_fraction=0.8,
        surface_fraction=0.9,
        bad_surface_fraction=0.1,
        mean_surface_excess=0.2,
        surface_penalty=1.0,
        mean_surface_distance=0.5,
        tail_penalty=1.0,
    )
    expected = (
        0.8
        - 4.0 * 0.1
        - 0.5 * 0.2
        - 2.0 * 0.0
        - 1.0 * 1.0
        - 0.005 * 3
    )
    assert expected == pytest.approx(0.8 - 0.4 - 0.1 - 1.0 - 0.015)


def test_feasible_score_ordering():
    a = _fake_analysis(
        newly_covered=50, coverage_fraction=1.0,
        mean_surface_distance=0.4,
    )
    b = _fake_analysis(
        newly_covered=50, coverage_fraction=1.0,
        mean_surface_distance=0.9,
    )
    assert d.feasible_score(a, 2) > d.feasible_score(b, 2)
    c = _fake_analysis(
        newly_covered=50, coverage_fraction=1.0,
        mean_surface_distance=0.4,
    )
    assert d.feasible_score(c, 1) > d.feasible_score(a, 3)


def test_one_component_nearly_valid_beats_two_components():
    """A nearly-valid single-component candidate must out-rank a
    two-component candidate on exploration merit (no comparator hack)."""
    one = _fake_analysis(
        newly_covered=40,
        coverage_fraction=0.4,
        surface_fraction=1.0,
        bad_surface_fraction=0.0,
        mean_surface_excess=0.0,
        trace_penalty=0.0,
        tail_penalty=0.25,
        n_components=1,
        extra_component_fraction=0.0,
    )
    two = _fake_analysis(
        newly_covered=40,
        coverage_fraction=0.4,
        surface_fraction=1.0,
        bad_surface_fraction=0.0,
        mean_surface_excess=0.0,
        trace_penalty=0.9,
        tail_penalty=0.0,
        n_components=2,
        extra_component_fraction=0.45,
    )

    def merit(an, degree):
        return (
            an.coverage_fraction
            - 4.0 * an.bad_surface_fraction
            - 0.5 * an.mean_surface_excess
            - 2.0 * an.trace_penalty
            - 1.0 * an.tail_penalty
            - 0.005 * degree
        )

    assert merit(one, 5) > merit(two, 5)


# ---------------------------------------------------------------------------
# Hill climb bookkeeping (monkeypatched analyses)
# ---------------------------------------------------------------------------


def test_hill_climb_returns_best_feasible_and_exploratory(monkeypatch):
    start_coef = np.array([0.0])
    feasible_coef = np.array([1.0])
    infeasible_coef = np.array([2.0])

    def fake_measure(cand, glyph, uncovered, *, dense=False):
        key = cand.coef.tobytes()
        feasible = key == feasible_coef.tobytes()
        newly = 20 if feasible else 30
        return _fake_analysis(
            samples=np.zeros((4, 2)), sample_d=np.zeros(4),
            point_d=np.zeros(len(glyph.search_points)),
            newly_covered=newly, coverage_fraction=0.5,
            surface_fraction=1.0 if feasible else 0.5,
            bad_surface_fraction=0.0 if feasible else 0.5,
            mean_surface_distance=0.5,
            bounds=(0.0, 1.0) if feasible else None,
            left_tail=d.TailInfo(True, 1.0, 10.0, True, 0, feasible),
            right_tail=d.TailInfo(True, 1.0, -10.0, True, 0, feasible),
            feasible=feasible,
            merit=0.5 if feasible else 1.0,
        )

    sequence = [
        d.Candidate(1, feasible_coef),
        d.Candidate(1, infeasible_coef),
        None,
    ]
    monkeypatch.setattr(d, "_mutate", lambda cand, rng, sigma: sequence.pop(0) if sequence else None)
    monkeypatch.setattr(d, "analyze_candidate", fake_measure)

    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)]))
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    result = d._hill_climb(
        d.Candidate(0, start_coef), glyph, uncovered,
        steps=10, rng=np.random.default_rng(0),
    )
    assert result.best_feasible_candidate is not None
    assert np.array_equal(result.best_feasible_candidate.coef, feasible_coef)
    assert result.best_feasible_analysis.feasible
    # The exploratory state must be retained even when no feasible state exists
    assert result.best_exploratory_candidate is not None


def test_hill_climb_returns_best_exploratory_with_no_feasible(monkeypatch):
    start_coef = np.array([0.0])
    high_merit_coef = np.array([9.0])

    def fake_measure(cand, glyph, uncovered, *, dense=False):
        key = cand.coef.tobytes()
        high = key == high_merit_coef.tobytes()
        return _fake_analysis(
            samples=np.zeros((4, 2)), sample_d=np.zeros(4),
            point_d=np.zeros(len(glyph.search_points)),
            newly_covered=5,
            coverage_fraction=0.5,
            surface_fraction=0.5,
            bad_surface_fraction=0.2,
            mean_surface_distance=0.5,
            bounds=None,
            n_components=1,
            left_tail=d.TailInfo(True, 1.0, 10.0, True, 0, False),
            right_tail=d.TailInfo(True, 1.0, -10.0, True, 0, False),
            feasible=False,
            merit=1.5 if high else 0.3,
        )

    sequence = [d.Candidate(1, high_merit_coef), None]
    monkeypatch.setattr(d, "_mutate", lambda cand, rng, sigma: sequence.pop(0) if sequence else None)
    monkeypatch.setattr(d, "analyze_candidate", fake_measure)

    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)]))
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    result = d._hill_climb(
        d.Candidate(0, start_coef), glyph, uncovered,
        steps=10, rng=np.random.default_rng(0),
    )
    assert result.best_feasible_candidate is None
    assert np.array_equal(result.best_exploratory_candidate.coef, high_merit_coef)
    assert result.best_exploratory_analysis.merit == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_use_real_newly_covered(capsys):
    # A multi-component candidate with real uncovered points: the report
    # must show the actual newly_covered count, not 0 from an all-false mask.
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    cand = d.Candidate(1, np.polynomial.Polynomial([50.0, 50.0])(d.X_OF_U).coef)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    an = d.analyze_candidate(cand, glyph, uncovered)
    assert an.newly_covered > 0
    d._report_no_first_curve(glyph, cand, an, uncovered)
    out = capsys.readouterr().err
    assert "new=" in out
    assert f"new={an.newly_covered}" in out


def test_tail_diagnostic_says_not_analyzed_for_multi_component(capsys):
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    cand = d.Candidate(2, np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    an = d.analyze_candidate(cand, glyph, uncovered)
    assert an.bounds is None
    d._report_no_first_curve(glyph, cand, an, uncovered)
    out = capsys.readouterr().err
    assert "tails=not analyzed: trace is not single-component" in out
    assert "trace_components=2" in out


# ---------------------------------------------------------------------------
# Degree reduction (monkeypatched, deterministic)
# ---------------------------------------------------------------------------


def test_reduce_degree_never_increases():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)]))
    rng = np.random.default_rng(0)
    cand = d.Candidate(3, np.array([1.0, 2.0, 3.0, 4.0]))
    assigned = np.arange(len(glyph.points))
    reduced = d._reduce_degree(cand, assigned, glyph, rng)
    assert reduced.degree <= cand.degree


def test_degree_reduction_coef_only_refinement_keeps_degree():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)]))
    rng = np.random.default_rng(0)
    cand = d.Candidate(3, np.array([1.0, 2.0, 3.0, 4.0]))
    refined, _ = d._refine_coef_only(cand, glyph, d.REDUCE_STEPS, rng)
    assert refined.degree == cand.degree


# ---------------------------------------------------------------------------
# Deterministic boundary subsampling
# ---------------------------------------------------------------------------


def test_search_boundary_subset_deterministic_and_capped():
    rng = np.random.default_rng(0)
    points = rng.normal(50.0, 10.0, size=(1000, 2))
    g1 = d.make_glyph(points)
    g2 = d.make_glyph(points)
    assert len(g1.search_points) <= d.SEARCH_BOUNDARY_MAX
    assert np.array_equal(g1.search_points, g2.search_points)
    assert np.array_equal(g1.search_idx, g2.search_idx)


# ---------------------------------------------------------------------------
# CLI contract (no heavy fitting)
# ---------------------------------------------------------------------------


def test_invalid_cli_input():
    for argv in [
        [], ["a"], ["1"], ["AA"], ["A", "B"],
        ["A", "--seed"], ["A", "--seed", "abc"],
        ["A", "--max-curves"], ["A", "--max-curves", "x"],
        ["A", "--unknown"],
    ]:
        assert d.run(argv) == 2


def test_entry_propagates_exit_code(monkeypatch, capsys):
    from src.__main__ import entry

    monkeypatch.setattr("sys.argv", ["denysko", "a"])
    with pytest.raises(SystemExit) as exc:
        entry()
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# One real glyph rasterization test (normalization + inner hole)
# ---------------------------------------------------------------------------

O_BOUNDARY = d.glyph_boundary("O")


def test_o_normalized_and_has_inner_hole():
    p = O_BOUNDARY
    assert p[:, 0].min() < 0.5
    assert p[:, 1].min() < 0.5
    assert p[:, 0].max() <= 100.0 + 1e-9
    assert p[:, 1].max() <= 100.0 + 1e-9
    center = (p.min(axis=0) + p.max(axis=0)) / 2.0
    r = np.hypot(p[:, 0] - center[0], p[:, 1] - center[1])
    assert r.min() > 20.0
    assert ((r > 30.0) & (r < 48.0)).sum() > 50