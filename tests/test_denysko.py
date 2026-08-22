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


# ---------------------------------------------------------------------------
# Endpoint-anchored seed geometry (deterministic construction)
# ---------------------------------------------------------------------------


def _diagonal_glyph():
    """Long diagonal stroke y = 2.5x - 5 across a finite in-band span."""
    xs = np.arange(2.0, 40.01, 0.5)
    return d.make_glyph(np.column_stack([xs, 2.5 * xs - 5.0]))


def test_endpoint_anchored_seed_geometry():
    """A natural-orientation bent seed follows the provisional straight
    route through the band, anchors value+slope at both provisional
    exits, and escapes past each exit within MAX_TAIL_X_RUN."""
    glyph = _diagonal_glyph()
    p1, p2 = glyph.search_points[10], glyph.search_points[20]
    entries = {e.name: e for e in d._seed_family(p1, p2, glyph)}
    line = entries["line"].candidate
    l0, r0 = d._provisional_trace_window(p1, p2, line, glyph)
    Lx = d.x_curve_of_candidate(line)

    seed = entries["down/up"].candidate  # rising stroke: exits low-left, high-right
    Sx = d.x_curve_of_candidate(seed)

    # values and derivatives equal the line at both provisional exits
    for t in (l0, r0):
        assert Sx.poly(t) == pytest.approx(float(Lx.poly(t)), abs=1e-9)
        assert Sx.poly.deriv()(t) == pytest.approx(
            float(Lx.poly.deriv()(t)), abs=1e-9
        )

    # the whole interior stays close to the line and inside the band
    mid = np.linspace(l0, r0, 201)
    dev = np.abs(Sx.poly(mid) - Lx.poly(mid))
    assert dev.max() <= d.TAIL_VERTICAL_MARGIN
    interior = Sx.poly(mid)
    assert interior.min() >= glyph.ymin - 1e-9
    assert interior.max() <= glyph.ymax + 1e-9

    # both margins are reached strictly inside MAX_TAIL_X_RUN
    tgt_l = glyph.ymin - d.TAIL_VERTICAL_MARGIN
    tgt_r = glyph.ymax + d.TAIL_VERTICAL_MARGIN
    grid_l = np.linspace(l0 - d.SEED_TAIL_X_RUN, l0, 20001)[::-1]
    grid_r = np.linspace(r0, r0 + d.SEED_TAIL_X_RUN, 20001)
    cross_l = grid_l[np.where(Sx.poly(grid_l) < tgt_l)[0][0]]
    cross_r = grid_r[np.where(Sx.poly(grid_r) > tgt_r)[0][0]]
    assert l0 - cross_l <= d.MAX_TAIL_X_RUN
    assert cross_r - r0 <= d.MAX_TAIL_X_RUN


def test_unbounded_horizontal_line_exploratory_scoring():
    """A horizontal crossbar line scores real surface/coverage during
    exploration despite being structurally unbounded."""
    xs = np.arange(30.0, 71.0, 1.0)
    rows = np.vstack([
        np.column_stack([xs, np.full(len(xs), 49.6)]),
        np.column_stack([xs, np.full(len(xs), 50.4)]),
    ])
    glyph = d.make_glyph(rows)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)

    const_u = np.polynomial.Polynomial([50.0])(d.X_OF_U).coef
    an = d.analyze_candidate(d.Candidate(0, const_u), glyph, uncovered)
    assert an.surface_fraction >= 0.95      # recognized as a great local stroke
    assert an.newly_covered > 0             # coverage is real, not zero
    assert an.bounds is None                # trace is unbounded -> infeasible
    assert not an.feasible                  # topology still rejected
    assert an.trace_penalty == pytest.approx(2.0)  # unbounded penalty kept

    # bending it produces a fully finite in-band set
    p1, p2 = np.array([40.0, 50.0]), np.array([55.0, 50.0])
    entries = {e.name: e for e in d._seed_family(p1, p2, glyph)}
    xc = d.x_curve_of_candidate(entries["down/up"].candidate)
    breaks = np.unique(
        np.concatenate([
            d._real_roots(xc.poly - glyph.ymin),
            d._real_roots(xc.poly - glyph.ymax),
        ])
    )
    comps = d._components_from_breaks(breaks, xc.poly, glyph.ymin, glyph.ymax)
    assert all(np.isfinite(a) and np.isfinite(b) for a, b in comps)


def test_turn_penalty_not_double_counted():
    """One outside derivative root contributes one unit of turn penalty,
    not two: the total equals the sum of the two side penalties."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    coef_u = np.polynomial.Polynomial([-3005.0, 900.0, 90.0, 3.0])(d.X_OF_U).coef
    cand = d.Candidate(3, coef_u)
    an = d.analyze_candidate(cand, glyph, np.zeros(len(glyph.search_points), dtype=bool))
    assert an.bounds is not None
    assert an.deriv_outside >= 1
    # composition identity: deriv_outside is fully accounted for by the sides
    assert an.deriv_outside == an.left_tail.turns + an.right_tail.turns

    xc = d.x_curve_of_candidate(cand)
    l, r = an.bounds
    lp = d._tail_side_penalty(
        an.left_tail, glyph.ymin, glyph.ymax, xc.poly, xc.poly.deriv(), l, "L"
    )
    rp = d._tail_side_penalty(
        an.right_tail, glyph.ymin, glyph.ymax, xc.poly, xc.poly.deriv(), r, "R"
    )
    assert an.tail_penalty == pytest.approx(lp + rp)


def test_tail_side_penalty_counts_turns_once():
    info1 = d.TailInfo(True, 1.0, 10.0, True, 1, False)
    info2 = d.TailInfo(True, 1.0, 10.0, True, 2, False)
    poly = np.polynomial.Polynomial([0.0])
    glyph_ = (0.0, 100.0)
    p1v = d._tail_side_penalty(info1, *glyph_, poly, poly.deriv(), 50.0, "R")
    p2v = d._tail_side_penalty(info2, *glyph_, poly, poly.deriv(), 50.0, "R")
    assert p2v - p1v == pytest.approx(1.0)


def test_missing_margin_gradient():
    """A tail that nearly reaches the +-5 margin scores better than one
    that barely leaves the band (no flat invalid-tail plateau)."""
    ymin, ymax = 0.0, 100.0
    end = 30.0
    info = d.TailInfo(False, float("inf"), 0.0, True, 0, False)
    # exited below: P(x) = -s (x - end); probe at end+5 gives -5s
    near = np.polynomial.Polynomial([0.96 * end, -0.96])   # probe -> -4.8
    far = np.polynomial.Polynomial([0.10 * end, -0.10])    # probe -> -0.5
    pen_near = d._tail_side_penalty(
        info, ymin, ymax, near, near.deriv(), end, "R"
    )
    pen_far = d._tail_side_penalty(
        info, ymin, ymax, far, far.deriv(), end, "R"
    )
    assert pen_near < pen_far
    assert pen_near == pytest.approx(1.0 + (5.0 - 4.8) / 5.0)
    assert pen_far == pytest.approx(1.0 + (5.0 - 0.5) / 5.0)
    assert pen_far - pen_near > 0.5  # a useful gradient, not flat


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
    entries = d._seed_family(p1, p2, glyph)
    assert len(entries) == 5
    assert [e.name for e in entries] == [
        "line", "up/up", "down/down", "up/down", "down/up",
    ]
    assert sorted(e.candidate.degree for e in entries) == [1, 5, 5, 5, 5]
    # only the line has no structured basis
    assert entries[0].basis is None
    for e in entries[1:]:
        assert e.basis is not None


def test_bent_seeds_preserve_value_and_derivative_at_trace_exits():
    """Bent seeds are anchored at the provisional trace endpoints."""
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    line = d._line_seed_u(p1, p2)
    l0, r0 = d._provisional_trace_window(p1, p2, line, glyph)
    ul = (l0 - 50.0) / 50.0
    ur = (r0 - 50.0) / 50.0
    Lx = d.x_curve_of_candidate(line)
    LvL, LvR = float(Lx.poly(l0)), float(Lx.poly(r0))
    dvL = float(Lx.poly.deriv()(l0))
    for e in d._seed_family(p1, p2, glyph)[1:]:
        xc = d.x_curve_of_candidate(e.candidate)
        assert xc.poly(l0) == pytest.approx(LvL, abs=1e-9)
        assert xc.poly(r0) == pytest.approx(LvR, abs=1e-9)
        assert xc.poly.deriv()(l0) == pytest.approx(dvL, abs=1e-9)
        assert xc.poly.deriv()(r0) == pytest.approx(dvL, abs=1e-9)


def test_bent_seeds_hit_tail_targets_at_provisional_exits():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    line = d._line_seed_u(p1, p2)
    l0, r0 = d._provisional_trace_window(p1, p2, line, glyph)
    xL = l0 - d.SEED_TAIL_X_RUN
    xR = r0 + d.SEED_TAIL_X_RUN
    up = glyph.ymax + d.TAIL_VERTICAL_MARGIN
    dn = glyph.ymin - d.TAIL_VERTICAL_MARGIN
    targets = [(up, up), (dn, dn), (up, dn), (dn, up)]
    for e, (tl, tr) in zip(d._seed_family(p1, p2, glyph)[1:], targets):
        xc = d.x_curve_of_candidate(e.candidate)
        assert xc.poly(xL) == pytest.approx(tl, abs=1e-6)
        assert xc.poly(xR) == pytest.approx(tr, abs=1e-6)


def test_unbounded_line_gets_working_window():
    """A horizontal line inside the band has no finite trace; the seed
    family must still anchor its bends around a finite working window."""
    xs = np.arange(30.0, 71.0, 1.0)
    rows = np.vstack([
        np.column_stack([xs, np.full(len(xs), 49.6)]),
        np.column_stack([xs, np.full(len(xs), 50.4)]),
    ])
    glyph = d.make_glyph(rows)
    p1, p2 = np.array([40.0, 50.0]), np.array([55.0, 50.0])
    line = d._line_seed_u(p1, p2)
    l0, r0 = d._provisional_trace_window(p1, p2, line, glyph)
    assert np.isfinite(l0) and np.isfinite(r0)
    center = 0.5 * (p1[0] + p2[0])
    assert l0 == pytest.approx(center - d.UNBOUNDED_SEED_HALF_WIDTH)
    assert r0 == pytest.approx(center + d.UNBOUNDED_SEED_HALF_WIDTH)
    # and the seed family still contains all five members
    assert len(d._seed_family(p1, p2, glyph)) == 5


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
    # A single-component candidate with real uncovered points: the report
    # must show the actual newly_covered count, not 0 from an all-false mask.
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    cand = d.Candidate(1, np.polynomial.Polynomial([50.0, 50.0])(d.X_OF_U).coef)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    an = d.analyze_candidate(cand, glyph, uncovered)
    assert an.newly_covered > 0
    d._report_no_first_curve(glyph, cand, an, uncovered, "line")
    out = capsys.readouterr().err
    assert "new=" in out
    assert f"new={an.newly_covered}" in out
    assert "seed=line" in out


def test_tail_diagnostic_says_not_analyzed_for_multi_component(capsys):
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    cand = d.Candidate(2, np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    an = d.analyze_candidate(cand, glyph, uncovered)
    assert an.bounds is None
    d._report_no_first_curve(glyph, cand, an, uncovered, "up/up")
    out = capsys.readouterr().err
    assert "tails=not analyzed: trace is not single-component" in out
    assert "trace_components=2" in out
    assert "seed=up/up" in out


def test_single_trace_diagnostic_reports_trace_bounds_and_tails(capsys):
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    cand = d.Candidate(1, np.polynomial.Polynomial([50.0, 50.0])(d.X_OF_U).coef)
    uncovered = np.ones(len(glyph.search_points), dtype=bool)
    an = d.analyze_candidate(cand, glyph, uncovered)
    d._report_no_first_curve(glyph, cand, an, uncovered, "down/up")
    out = capsys.readouterr().err
    assert f"trace_bounds=[{an.bounds[0]:.2f}, {an.bounds[1]:.2f}]" in out
    assert "direction=" in out
    assert "margin=" in out
    assert "turns=" in out
    assert "seed=down/up" in out


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
# Per-restart budget split and all-seed refinement
# ---------------------------------------------------------------------------


def test_split_steps_preserves_budget():
    assert d._split_steps(120, 5) == [24, 24, 24, 24, 24]
    split = d._split_steps(122, 5)
    assert sum(split) == 122
    # remainder distributed deterministically to the first seeds
    assert split == [25, 25, 24, 24, 24]
    assert d._split_steps(120, 1) == [120]


def test_run_restart_refines_every_seed(monkeypatch):
    """Every generated seed is independently refined, with the total
    refinement budget split across them rather than multiplied."""
    glyph = _diagonal_glyph()
    p1, p2 = glyph.search_points[10], glyph.search_points[20]
    entries = d._seed_family(p1, p2, glyph)

    recorded = []

    def fake_hill_climb(cand, glyph_, uncovered_, steps, rng, basis=None):
        recorded.append((steps, cand.degree, basis is not None))
        return d.HillResult(
            None, None,
            cand,
            _fake_analysis(
                samples=np.zeros((4, 2)), sample_d=np.zeros(4),
                point_d=np.zeros(len(glyph.search_points)),
                feasible=False,
            ),
        )

    monkeypatch.setattr(d, "_hill_climb", fake_hill_climb)
    rng = np.random.default_rng(0)
    result = d._run_restart(entries, glyph, np.ones(len(glyph.search_points), dtype=bool), rng, d.REFINE_STEPS)

    assert len(recorded) == len(entries)          # every seed gets a hill
    assert sum(s for s, _, _ in recorded) <= d.REFINE_STEPS  # budget not multiplied
    assert sum(s for s, _, _ in recorded) == d.REFINE_STEPS
    steps_list = [s for s, _, _ in recorded]
    assert all(steps_list[i] >= steps_list[i + 1] for i in range(len(steps_list) - 1))
    # structured bases are passed through to the bent-seed hills
    assert recorded[0][2] is False                # line: no basis
    for _, _, has_basis in recorded[1:]:
        assert has_basis
    assert result.best_seed_name is not None


def test_structured_mutations_preserve_degree_early(monkeypatch):
    """During the first half of a structured hill, degree mutation is
    suppressed so the quintic basin is refined first."""
    glyph = _diagonal_glyph()
    entries = {e.name: e for e in d._seed_family(
        glyph.search_points[10], glyph.search_points[20], glyph
    )}
    basis = entries["down/up"].basis
    rng = np.random.default_rng(0)
    cand = entries["down/up"].candidate

    seen_degrees = set()
    for t in range(50):
        mutant = d._mutate_search(
            cand, basis, rng, d._coef_sigma(t, 100), allow_degree=(t >= 25)
        )
        if t < 25:
            assert mutant.degree == cand.degree
        seen_degrees.add(mutant.degree)
    # after the first half, degree mutation may appear (degree-5 can only drop)
    assert min(seen_degrees) <= cand.degree


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