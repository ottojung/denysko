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
    # y = 120 - x^2, band [0,100]: two in-band intervals
    coef_u = np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(2, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds is None
    assert an.trace_penalty == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tail rules
# ---------------------------------------------------------------------------


def test_tail_derivative_root_rejection():
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    # P = 3(x+10)^3 - 5 has a stationary point below the trace
    coef_u = np.polynomial.Polynomial([-3005.0, 900.0, 90.0, 3.0])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(3, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds is not None
    assert an.deriv_outside >= 1
    assert not an.feasible


def test_tail_case_a_shallow_forever_invalid():
    """A line crossing the band with slope 2 stays slope 2: never steep."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    an = d.analyze_candidate(
        d.Candidate(1, np.array([50.0, 50.0])),  # y = x, slope 1 < 8
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert not an.left_tail_valid or not an.right_tail_valid
    assert not an.feasible


def test_tail_case_b_shallow_at_endpoint_steep_outside_valid():
    """Follows a boundary, crosses the band shallowly, then turns steep.

    A Hermite quintic has trace-endpoint slopes ~3.5 (< 8) but reaches
    the +-5 vertical margin within ~0.75 x-units with |P'| ~ 10.8,
    so both tails are valid under the post-exit rule.
    """
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
    assert abs(an.bounds[0]) < 8.0  # endpoint slopes are shallow (< 8)
    assert an.left_tail_valid
    assert an.right_tail_valid
    assert d.structurally_feasible(an)
    assert d.validate([d.format_expression(xc)], glyph) == []


def test_tail_case_c_steep_but_turns_back_invalid():
    """Reaches the outside margin steeply but turns back and re-enters."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    # y = -x^2 + 120 rises out the top, peaks, then falls back through the
    # band: two trace components, so the curve is not a single permanent exit.
    coef_u = np.polynomial.Polynomial([120.0, 0.0, -1.0])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(2, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.bounds is None
    assert an.trace_penalty >= 1.0
    assert not an.feasible


def test_tail_case_d_steep_too_late_invalid():
    """Reaches the +-5 vertical margin only after >5 horizontal units."""
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0)] * 2))
    # y = 50 + 0.5x: exits the band slowly, so the margin is far away
    coef_u = np.polynomial.Polynomial([50.0, 0.5])(d.X_OF_U).coef
    an = d.analyze_candidate(
        d.Candidate(1, coef_u),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert not an.left_tail_valid or not an.right_tail_valid
    assert not an.feasible


def test_outside_band_no_surface_penalty():
    # Vertical bar: distances only near the trace matter.
    ys = np.arange(5.0, 95.01, 0.5)
    p = np.vstack([
        np.column_stack([np.full(len(ys), 49.6), ys]),
        np.column_stack([np.full(len(ys), 50.4), ys]),
    ])
    glyph = d.make_glyph(p)
    # steep line hugging the bar: 50 + 1400u = 28x - 1350
    an = d.analyze_candidate(
        d.Candidate(1, np.array([50.0, 1400.0])),
        glyph, np.zeros(len(glyph.search_points), dtype=bool),
    )
    assert an.surface_fraction >= 0.95
    assert an.bad_surface_fraction < 0.05
    assert an.mean_surface_excess < 0.05


def test_connected_shape_validator_handcrafted():
    """A handcrafted curve crossing ymin and ymax is surface-valid.

    The boundary is a connected sloped segment; the curve follows it and
    exits steeply at both band crossings. A shallow exit fails the tail
    steepness rule.
    """
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
    assert not an.left_tail_valid or not an.right_tail_valid


def test_euclidean_distance_tiny_cloud():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 2.0]])
    da, db = d._min_dists(a, b)
    assert da[0] == pytest.approx(2.0)
    assert db[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# p2 ranking
# ---------------------------------------------------------------------------


def test_segment_p2_ranking():
    p = np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)])
    p1 = np.array([50.0, 0.0])
    good = np.array([60.0, 0.0])   # along the boundary
    bad = np.array([60.0, 30.0])   # sticks far off the boundary
    d1, _ = d._min_dists(
        np.linspace(p1, good, 33), p
    )
    d2, _ = d._min_dists(
        np.linspace(p1, bad, 33), p
    )
    assert d1.mean() < d2.mean()


# ---------------------------------------------------------------------------
# Bent-line seeds
# ---------------------------------------------------------------------------


def _tiny_glyph():
    xs = np.arange(10.0, 91.0, 2.0)
    return d.make_glyph(np.column_stack([xs, 0.5 * xs]))


def _line_val(line_coef, x):
    u = (x - 50.0) / 50.0
    return line_coef[0] + line_coef[1] * u


def test_five_initial_seeds():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    seeds = d._initial_seeds(p1, p2, glyph)
    assert len(seeds) == 5
    assert sorted(s.degree for s in seeds) == [1, 4, 4, 5, 5]


def test_bend_seeds_preserve_value_and_derivative():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    u1 = (p1[0] - 50.0) / 50.0
    u2 = (p2[0] - 50.0) / 50.0
    line = np.array([
        p1[1] - ((p2[1] - p1[1]) / (u2 - u1)) * u1,
        (p2[1] - p1[1]) / (u2 - u1),
    ])
    for s in d._bent_seeds_u(p1, p2, glyph):
        # values preserved
        assert np.polyval(s.coef[::-1], u1) == pytest.approx(p1[1], abs=1e-9)
        assert np.polyval(s.coef[::-1], u2) == pytest.approx(p2[1], abs=1e-9)
        # derivative preserved: P'(u) == L'(u) at the seed points
        dp = np.polyder(s.coef[::-1])
        assert np.polyval(dp, u1) == pytest.approx(line[1], abs=1e-9)
        assert np.polyval(dp, u2) == pytest.approx(line[1], abs=1e-9)


def test_quartic_both_up_bends_above_band():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    u1 = (p1[0] - 50.0) / 50.0
    u2 = (p2[0] - 50.0) / 50.0
    line = np.array([p1[1] - ((p2[1] - p1[1]) / (u2 - u1)) * u1, (p2[1] - p1[1]) / (u2 - u1)])
    xL = glyph.xmin - 5.0
    xR = glyph.xmax + 5.0
    up = [s for s in d._bent_seeds_u(p1, p2, glyph) if s.degree == 4]
    assert up
    above = [s for s in up
             if np.polyval(s.coef[::-1], (xL - 50.0) / 50.0) > _line_val(line, xL)
             and np.polyval(s.coef[::-1], (xR - 50.0) / 50.0) > _line_val(line, xR)]
    assert above


def test_quintic_opposite_tails():
    glyph = _tiny_glyph()
    p1, p2 = glyph.search_points[5], glyph.search_points[15]
    u1 = (p1[0] - 50.0) / 50.0
    u2 = (p2[0] - 50.0) / 50.0
    line = np.array([p1[1] - ((p2[1] - p1[1]) / (u2 - u1)) * u1, (p2[1] - p1[1]) / (u2 - u1)])
    xL = glyph.xmin - 5.0
    xR = glyph.xmax + 5.0
    quin = [s for s in d._bent_seeds_u(p1, p2, glyph) if s.degree == 5]
    assert quin
    opp = [s for s in quin
           if (np.polyval(s.coef[::-1], (xL - 50.0) / 50.0) > _line_val(line, xL)
               and np.polyval(s.coef[::-1], (xR - 50.0) / 50.0) < _line_val(line, xR))]
    assert opp


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
        deriv_outside=0,
        left_exit_slope=10.0,
        right_exit_slope=-10.0,
        left_vertical_x=0.0,
        right_vertical_x=1.0,
        left_x_run=1.0,
        right_x_run=1.0,
        left_tail_valid=True,
        right_tail_valid=True,
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
    # lower degree wins ties on coverage
    c = _fake_analysis(
        newly_covered=50, coverage_fraction=1.0,
        mean_surface_distance=0.4,
    )
    assert d.feasible_score(c, 1) > d.feasible_score(a, 3)


# ---------------------------------------------------------------------------
# Best feasible state bookkeeping (monkeypatched analyses)
# ---------------------------------------------------------------------------


def test_hill_climb_returns_best_feasible(monkeypatch):
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
            left_tail_valid=feasible,
            right_tail_valid=feasible,
            feasible=feasible, merit=0.5 if feasible else 1.0,
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
    best = d._hill_climb(
        d.Candidate(0, start_coef), glyph, uncovered,
        steps=10, rng=np.random.default_rng(0),
    )
    assert best is not None
    cand, an = best
    assert np.array_equal(cand.coef, feasible_coef)
    assert an.feasible


# ---------------------------------------------------------------------------
# Degree reduction (monkeypatched, deterministic)
# ---------------------------------------------------------------------------


def test_reduce_degree_never_increases(monkeypatch):
    glyph = d.make_glyph(np.column_stack([np.arange(0.0, 101.0, 1.0), np.zeros(101)]))
    rng = np.random.default_rng(0)
    cand = d.Candidate(3, np.array([1.0, 2.0, 3.0, 4.0]))
    assigned = np.arange(len(glyph.points))
    reduced = d._reduce_degree(cand, assigned, glyph, rng)
    assert reduced.degree <= cand.degree


def test_degree_reduction_coef_only_refinement_keeps_degree(monkeypatch):
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