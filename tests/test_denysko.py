import numpy as np
import pytest

from src import denysko as d
from src.topology import (
    glyph_geometry,
    BoundaryPath,
    GlyphGeometry,
    build_corridors,
    dedupe_paths,
    extract_paths,
    assign_coverage,
    select_paths,
    TAU,
)
from src import fitting as _fitting
from src.fitting import fit_degree, min_degree, INITIAL_FIT_DEGREE, POCS_SWEEPS


# ---------------------------------------------------------------------------
# Serialization / parsing (ordinary powers of x, no domain restrictions)
# ---------------------------------------------------------------------------


def test_u_to_x_conversion_regression():
    # internal u-basis [100, 100] -> y = 2x
    cand = d.parse_line("y=2x")
    assert cand is not None
    assert d.format_expression(cand) == "y=2x"


def test_serialize_parse_roundtrip_exact():
    for line in [
        "y=2x",
        "y=0",
        "y=x^5-3x^2+0.5x-7",
        "y=-x^3",
        "y=0.000000000000000000053525304330325684x^14+1",
    ]:
        curve = d.parse_line(line)
        assert curve is not None
        assert d.format_expression(curve) == line


def test_fmt_num_round_trips_tiny_coefficients():
    v = 5.3525304330325684e-20
    s = d.fmt_num(v)
    assert "e" not in s and "E" not in s
    assert float(s) == v


def test_malformed_lines_do_not_parse():
    for line in ["", "y=", "y=x^^2", "y=..x", "x=2", "y=x^2 \\left\\{0\\le x\\le 10\\right\\}"]:
        assert d.parse_line(line) is None


# ---------------------------------------------------------------------------
# Phase 1a: boundary contours -> x-monotone paths
# ---------------------------------------------------------------------------


def _rect_contour(x0=10, x1=60, y0=20, y1=70):
    return np.array([
        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
    ], dtype=float)


def test_rectangle_decomposes_into_monotone_paths():
    paths = extract_paths([_rect_contour()])
    assert len(paths) >= 2
    for p in paths:
        xs = p.points[:, 0]
        assert np.all(np.diff(xs) >= -1e-9)


def test_c_like_contour_splits_at_x_reversal():
    t = np.linspace(-np.pi / 2, np.pi / 2, 200)
    # horseshoe opening to the right: x reverses nowhere? use full circle
    # minus right opening so traversal goes down-left-up (one reversal).
    ang = np.linspace(0.25 * np.pi, 1.75 * np.pi, 240)
    contour = np.column_stack([
        40 + 30 * np.cos(ang),
        50 + 30 * np.sin(ang),
    ])
    paths = extract_paths([np.vstack([contour, contour[:1]])])
    assert len(paths) >= 2
    for p in paths:
        assert np.all(np.diff(p.points[:, 0]) >= -1e-9)


def test_hole_contours_become_their_own_paths():
    outer = _rect_contour(0, 100, 0, 100)
    hole = _rect_contour(30, 70, 30, 70)
    paths = extract_paths([outer, hole])
    cids = {p.contour_id for p in paths}
    assert cids == {0, 1}
    hole_paths = [p for p in paths if p.contour_id == 1]
    assert hole_paths


def test_paths_are_x_monotone():
    rng = np.random.default_rng(3)
    pts = rng.normal(50, 20, size=(80, 2))
    paths = extract_paths([pts])
    for p in paths:
        assert np.all(np.diff(p.points[:, 0]) >= -1e-9)


# ---------------------------------------------------------------------------
# Phase 1b: corridors
# ---------------------------------------------------------------------------


def _path_from_y(xs, ys):
    return BoundaryPath(points=np.column_stack([xs, ys]), contour_id=0)


def _diag_geom():
    xs = np.arange(2.0, 42.0, 0.5)
    return GlyphGeometry(
        letter="T",
        contours=[],
        points=np.column_stack([xs, 2.5 * xs - 5.0]),
        xmin=float(xs.min()), xmax=float(xs.max()),
        ymin=0.0, ymax=100.0,
    )


def test_corridor_contains_its_own_path():
    geom = _diag_geom()
    p = _path_from_y(geom.points[:60, 0], geom.points[:60, 1])
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    mid_x = corr.path.points[:, 0]
    lo = corr.lower_at(mid_x)
    hi = corr.upper_at(mid_x)
    assert np.all(lo <= corr.path.points[:, 1])
    assert np.all(hi >= corr.path.points[:, 1])


def test_corridor_lower_le_upper_everywhere_including_escapes():
    geom = _diag_geom()
    p = _path_from_y(geom.points[:60, 0], geom.points[:60, 1])
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    assert np.all(corr.lower <= corr.upper)


def test_parallel_close_corridors_do_not_merge():
    geom = GlyphGeometry(
        letter="T", contours=[],
        points=np.empty((0, 2)),
        xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0,
    )
    xs = np.linspace(10.0, 90.0, 40)
    lower = _path_from_y(xs, np.full_like(xs, 49.0))
    upper = _path_from_y(xs, np.full_like(xs, 51.0))
    lower.covered = np.zeros(len(geom.points), dtype=bool)
    upper.covered = np.zeros(len(geom.points), dtype=bool)
    cl = build_corridors([lower], geom)[0]
    cu = build_corridors([upper], geom)[0]
    # at matching interior x, corridors must not overlap
    mid = xs[len(xs) // 2]
    assert cl.upper_at(mid) <= cu.lower_at(mid)


def test_escape_rows_move_outside_band():
    geom = _diag_geom()
    p = _path_from_y(geom.points[-60:, 0], geom.points[-60:, 1])  # top part
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    esc = corr.escapes[1]  # right end
    if esc.kind == "band":
        for x, lo, hi in esc.rows:
            bound = lo if np.isfinite(lo) else hi
            if esc.sigma == 1:
                assert bound >= geom.ymax
            else:
                assert bound <= geom.ymin
    else:
        for x, lo, hi in esc.rows:
            if esc.sigma == 1:
                assert lo >= geom.ymax + 1.0
            else:
                assert hi <= geom.ymin - 1.0


# ---------------------------------------------------------------------------
# Set-cover selection
# ---------------------------------------------------------------------------


def test_greedy_set_cover_synthetic():
    class C:
        def __init__(self, mask):
            self.path = type(
                "P", (), {"covered": mask, "points": np.zeros((2, 2))}
            )()

    masks = {
        "A": [True, True, True, False, False],
        "B": [False, False, True, True, True],
        "C": [True, False, False, False, True],
    }
    corrs = [C(np.array(m)) for m in masks.values()]
    selected, covered = select_paths(corrs, coverage_target=1.0, max_paths=12)
    picked = [corrs.index(c) for c in selected]
    assert picked == [0, 1]          # A then B covers everything
    assert all(covered)


def test_selection_respects_max_paths_and_reports_coverage():
    class C:
        def __init__(self, mask):
            self.path = type(
                "P", (), {"covered": mask, "points": np.zeros((2, 2))}
            )()

    one = [True] * 6 + [False] * 4
    others = [[i == j for j in range(10)] for i in range(6, 10)]
    corrs = [C(np.array(one))] + [C(np.array(m)) for m in others]
    selected, covered = select_paths(corrs, coverage_target=1.0, max_paths=12)
    assert covered.all()



@pytest.fixture
def fast_polish(monkeypatch):
    monkeypatch.setattr(_fitting, "POCS_SWEEPS", 120)
    monkeypatch.setattr(_fitting, "USE_LP", False)
    return 120


# ---------------------------------------------------------------------------
# Phase 2: constrained feasibility fitting
# ---------------------------------------------------------------------------


def _corridor_from_bounds(xs, lo, hi):
    p = _path_from_y(xs, (lo + hi) / 2.0)
    p.covered = np.ones(max(len(xs), 1), dtype=bool)
    geom = GlyphGeometry(
        letter="T", contours=[], points=p.points.copy(),
        xmin=float(xs[0]), xmax=float(xs[-1]),
        ymin=float(lo.min()), ymax=float(hi.max()),
    )
    # reuse build_corridors on a single fully-covered path: widths come
    # from foreign distance; there is no foreign geometry here.
    corr = build_corridors([p], geom)[0]
    # tighten to requested analytic bounds for determinism
    corr.xs = np.asarray(xs)
    corr.lower = np.asarray(lo)
    corr.upper = np.asarray(hi)
    return corr


def test_simple_linear_corridor_feasible(fast_polish):
    xs = np.linspace(1.0, 10.0, 24)
    corr = _corridor_from_bounds(xs, 0.9 * xs, 1.1 * xs)
    fit = fit_degree(corr, 4)
    assert fit is not None
    assert fit.dense_max_violation <= 0.25 * TAU


def test_impossible_low_degree_corridor_fails(fast_polish):
    xs = np.linspace(0.0, 10.0, 36)
    # W-shaped bounds cannot be matched by degree 2
    wiggle = 6.0 * np.abs(((xs / 10.0) % 1.0) - 0.5)
    corr = _corridor_from_bounds(xs, 10 * wiggle - 3.0, 10 * wiggle + 3.0)
    fit = fit_degree(corr, 2)
    assert fit is None


def test_degree_reduction_binary_search_minimum(fast_polish):
    xs = np.linspace(0.0, 10.0, 36)
    cubic = ((xs - 5.0) / 5.0) ** 3 * 4.0
    width = 0.08
    corr = _corridor_from_bounds(xs, cubic - width, cubic + width)
    top = fit_degree(corr, INITIAL_FIT_DEGREE)
    assert top is not None
    best = min_degree(corr)
    assert best is not None
    assert best.degree <= 4
    # verified minimum: neighbor below must be infeasible
    below = fit_degree(corr, max(best.degree - 1, 0))
    assert below is None or below.degree > best.degree


def test_candidate_outside_corridor_is_rejected():
    xs = np.linspace(1.0, 10.0, 24)
    corr = _corridor_from_bounds(xs, 0.9 * xs, 1.1 * xs)
    # a polynomial that follows a DIFFERENT nearby route
    other = np.polynomial.Polynomial([30.0, -1.0])
    from src.fitting import _dense_violation
    viol = _dense_violation(corr, np.polynomial.chebyshev.poly2cheb(
        other.coef / (1.0)
    ) * 0 + _cheb_coeffs_of(other, corr.xa, corr.xb))
    assert viol > 0.25 * TAU


def _cheb_coeffs_of(poly, xa, xb):
    zmap_poly = np.polynomial.Polynomial(
        [-(xa + xb) / (xb - xa), 2.0 / (xb - xa)]
    )
    composed = poly(zmap_poly)
    return np.polynomial.chebyshev.poly2cheb(composed.coef)


# ---------------------------------------------------------------------------
# Escape divergence (synthetic route)
# ---------------------------------------------------------------------------


def test_escape_upward_divergence():
    # small route whose right endpoint tops out near ymax: escape up
    geom = _diag_geom()
    xs = np.linspace(geom.xmax - 30, geom.xmax - 1.0, 30)
    ys = geom.ymax - 5.0 + 0.05 * (xs - xs[0])
    p = _path_from_y(xs, ys)
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    esc_right = corr.escapes[1]
    assert esc_right.sigma == 1
    # every escape row demands being above the band edge, growing outward
    prev = geom.ymax
    for x, lo, hi in esc_right.rows:
        bound = lo if np.isfinite(lo) else hi
        assert bound >= prev
        prev = bound
    # and the fitted polynomial actually leaves upward through the rows
    fit = min_degree(corr)
    assert fit is not None
    for x, lo, hi in esc_right.rows:
        v = float(fit.poly(x))
        if np.isfinite(lo):
            assert v >= lo - 0.25 * TAU
        else:
            assert v <= hi + 0.25 * TAU


# ---------------------------------------------------------------------------
# Independent global validation
# ---------------------------------------------------------------------------


def test_validate_flags_low_global_coverage():
    geom = _diag_geom()
    fits = []
    xs = np.linspace(geom.points[0, 0] + 1, geom.points[10, 0], 30)
    ys = 2.5 * xs - 5.0
    p = _path_from_y(xs, ys)
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    fit = fit_degree(corr, 6)
    assert fit is not None
    lines = [d.format_expression(fit.poly)]
    problems = d.validate_lines(lines, geom, [fit])
    assert any(m.startswith("V1") for m in problems)


# ---------------------------------------------------------------------------
# Deterministic boundary subsampling / geometry cache
# ---------------------------------------------------------------------------


O_GEOM = glyph_geometry("O")


def test_o_geometry_has_inner_contour_and_cloud_hole():
    assert len(O_GEOM.contours) >= 2
    center = O_GEOM.points.mean(axis=0)
    r = np.hypot(
        O_GEOM.points[:, 0] - center[0], O_GEOM.points[:, 1] - center[1]
    )
    # ring: boundary samples occupy two radial bands (inner + outer edge)
    hist, edges = np.histogram(r, bins=8)
    assert (hist > 30).sum() >= 4


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_invalid_cli_input():
    for argv in [
        [], ["a"], ["1"], ["AA"], ["A", "B"],
        ["A", "--seed"],                       # missing value
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