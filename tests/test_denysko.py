import numpy as np
import pytest

from src import denysko as d
from src import fitting as _fitting
from src.fitting import (
    fit_degree,
    min_degree,
    INITIAL_FIT_DEGREE,
)
from src.denysko import (
    corridor_adherence_violation,
    tail_reentry_violation,
    uncovered_clusters,
)
from src.topology import (
    BoundaryPath,
    GlyphGeometry,
    TAU,
    build_corridors,
    contour_edge_count,
    dedupe_paths,
    assign_coverage,
    extract_paths,
    select_paths,
    glyph_geometry,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rect_contour(x0=10, y0=20, x1=60, y1=70):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float
    )


def _diag_geom():
    xs = np.arange(2.0, 42.0, 0.5)
    return GlyphGeometry(
        letter="T",
        contours=[],
        points=np.column_stack([xs, 2.5 * xs - 5.0]),
        xmin=float(xs.min()), xmax=float(xs.max()),
        ymin=0.0, ymax=100.0,
    )


def _path_from_y(xs, ys):
    return BoundaryPath(points=np.column_stack([xs, ys]), contour_id=0)


@pytest.fixture
def fast_polish(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", False)
    monkeypatch.setattr(_fitting, "POCS_SWEEPS", 120)
    return 120


# ---------------------------------------------------------------------------
# serialization / parsing
# ---------------------------------------------------------------------------


def test_u_to_x_conversion_regression():
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
    for line in ["", "y=", "y=x^^2", "y=..x", "x=2",
                 "y=x^2 \\left\\{0\\le x\\le 10\\right\\}"]:
        assert d.parse_line(line) is None


# ---------------------------------------------------------------------------
# topology: vertical edges / runs / edge invariant
# ---------------------------------------------------------------------------


def test_exact_vertical_edge_preserved():
    rect = _rect_contour()
    paths = extract_paths([rect])
    vert = [
        p for p in paths if p.source_edge_ids in ((1,), (3,))
    ]
    assert len(vert) == 2
    for p in vert:
        span = p.points[:, 0].max() - p.points[:, 0].min()
        assert span <= 0.5 + 1e-9            # VERTICAL_PATH_X_SPAN
        assert np.all(np.diff(p.points[:, 0]) >= -1e-12)
        assert abs(p.points[:, 1].max() - p.points[:, 1].min()) == 50.0


def test_vertical_run_single_path():
    # three consecutive vertical edges on one side (a staircase of
    # collinear segments) must yield ONE path, not three
    side = np.array([[60, 20], [60, 40], [60, 55], [60, 70]], dtype=float)
    rect = np.vstack([
        [[10, 20]], side, [[10, 70]], [[10, 20]],
    ])
    paths = extract_paths([rect])
    vert_runs = [p for p in paths if set(p.source_edge_ids) == {1, 2, 3}]
    assert len(vert_runs) == 1
    assert abs(vert_runs[0].points[:, 1].max()
               - vert_runs[0].points[:, 1].min()) == 50.0


def test_every_rectangle_edge_represented():
    rect = _rect_contour()
    n_edges = contour_edge_count(rect)
    paths = extract_paths([rect])
    used = []
    for p in paths:
        used.extend(p.source_edge_ids)
    assert sorted(used) == list(range(n_edges))


def test_no_edge_silently_dropped_random_contour():
    rng = np.random.default_rng(5)
    pts = rng.normal(50, 15, size=(40, 2))
    loop = np.vstack([pts, pts[:1]])
    paths = extract_paths([loop])
    used = set()
    for p in paths:
        used.update(p.source_edge_ids)
    degenerate = {
        i for i in range(40)
        if np.allclose(loop[(i + 1) % 40], loop[i])
    }
    assert used | degenerate == set(range(40))


# ---------------------------------------------------------------------------
# corridors
# ---------------------------------------------------------------------------


def test_corridor_contains_own_path():
    geom = _diag_geom()
    p = _path_from_y(geom.points[:60, 0], geom.points[:60, 1])
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    mid_x = corr.path.points[:, 0]
    assert np.all(corr.lower_at(mid_x) <= corr.path.points[:, 1])
    assert np.all(corr.upper_at(mid_x) >= corr.path.points[:, 1])


def test_parallel_close_corridors_do_not_merge():
    def ring(x0, x1, yc):
        return np.array([
            [x0, yc - 2], [x1, yc - 2], [x1, yc + 2],
            [x0, yc + 2], [x0, yc - 2],
        ], dtype=float)

    c0 = ring(10, 90, 49.0)   # occupies y in [47, 51]
    c1 = ring(10, 90, 55.0)   # occupies y in [53, 57]; gap 2 < 2*TAU
    cloud = np.vstack([c0[:-1], c1[:-1]])
    geom = GlyphGeometry("T", [c0, c1], cloud, 0, 100, 40, 60)
    paths = extract_paths([c0, c1])
    masks = assign_coverage(paths, cloud, TAU)
    for p, m in zip(paths, masks):
        p.covered = m                      # bookkeeping may overlap...
    corrs = build_corridors(paths, geom)
    horiz = [
        c for c in corrs
        if c.path.points[-1, 0] - c.path.points[0, 0] > 20
    ]
    levels = sorted((c.lower_at(50.0), c.upper_at(50.0)) for c in horiz)
    # ...but corridors stay distinct (resampling jitter tolerance 0.15)
    for i in range(len(levels) - 1):
        assert levels[i][1] <= levels[i + 1][0] + 0.15


# ---------------------------------------------------------------------------
# set-cover selection
# ---------------------------------------------------------------------------


class _FakeCorridor:
    def __init__(self, mask):
        self.path = type(
            "P", (), {"covered": mask, "points": np.zeros((2, 2))}
        )()


def test_greedy_set_cover_synthetic():
    masks = {
        "A": [True, True, True, False, False],
        "B": [False, False, True, True, True],
        "C": [True, False, False, False, True],
    }
    corrs = [_FakeCorridor(np.array(m)) for m in masks.values()]
    selected, covered = select_paths(corrs, coverage_target=1.0,
                                     max_paths=12)
    picked = [corrs.index(c) for c in selected]
    assert picked == [0, 1]
    assert covered.all()


# ---------------------------------------------------------------------------
# constrained fitting (fast non-LP path unless stated)
# ---------------------------------------------------------------------------


def _corridor_from_bounds(xs, lo, hi):
    p = _path_from_y(xs, (np.asarray(lo) + np.asarray(hi)) / 2.0)
    p.covered = np.ones(max(len(xs), 1), dtype=bool)
    geom = GlyphGeometry(
        letter="T", contours=[], points=p.points.copy(),
        xmin=float(xs[0]), xmax=float(xs[-1]),
        ymin=float(np.min(lo)), ymax=float(np.max(hi)),
    )
    corr = build_corridors([p], geom)[0]
    corr.xs = np.asarray(xs, dtype=float)
    corr.lower = np.asarray(lo, dtype=float)
    corr.upper = np.asarray(hi, dtype=float)
    return corr


def test_production_lp_smoke(monkeypatch):
    """Tiny end-to-end use of the REAL production solver:
    constraint construction -> scipy HiGHS LP -> polish -> dense gate."""
    monkeypatch.setattr(_fitting, "USE_LP", True)
    monkeypatch.setattr(_fitting, "FIT_GRID", 48)
    monkeypatch.setattr(_fitting, "DENSE_GRID", 160)

    xs = np.linspace(1.0, 10.0, 24)
    corr = _corridor_from_bounds(xs, 0.9 * xs, 1.1 * xs)
    fit = fit_degree(corr, 2)
    assert fit is not None
    assert fit.dense_max_violation <= 0.25 * TAU


def test_simple_linear_corridor_feasible(fast_polish):
    xs = np.linspace(1.0, 10.0, 24)
    corr = _corridor_from_bounds(xs, 0.9 * xs, 1.1 * xs)
    fit = fit_degree(corr, 4)
    assert fit is not None
    assert fit.dense_max_violation <= 0.25 * TAU


def test_impossible_low_degree_corridor_fails(fast_polish):
    xs = np.linspace(0.0, 10.0, 36)
    tri = 8.0 * np.abs(((xs + 2.5) % 5.0) - 2.5) / 2.5 - 4.0
    corr = _corridor_from_bounds(xs, tri - 1.0, tri + 1.0)
    fit = fit_degree(corr, 2)
    assert fit is None


def test_degree_minimization_verified_minimum(fast_polish):
    xs = np.linspace(0.0, 10.0, 30)
    cubic = ((xs - 5.0) / 5.0) ** 3 * 4.0
    width = 0.08
    corr = _corridor_from_bounds(xs, cubic - width, cubic + width)
    best = min_degree(corr)
    assert best is not None
    # true lowest verified feasible degree: nothing below works
    for dd in range(0, best.degree):
        assert fit_degree(corr, dd) is None
    # and the returned degree itself is feasible
    again = fit_degree(corr, best.degree)
    assert again is not None


def test_chebyshev_window_covers_all_escape_rows(fast_polish):
    """Endpoint deep inside the band -> escape run exceeds the default
    pad; every constraint x must map into z in [-1, 1]."""
    geom = _diag_geom()
    # endpoint mid-band, far from both edges
    xs = np.linspace(20.0, 30.0, 20)
    p = _path_from_y(xs, np.full_like(xs, 50.0))
    p.covered = np.ones(len(geom.points), dtype=bool)
    # endpoint y=50 is mid-band and NOT near a glyph x-edge relative to
    # this synthetic geom's span -> band ramp with run beyond default pad
    xs = np.linspace(20.0, 30.0, 20)
    p = _path_from_y(xs, np.full_like(xs, 50.0))
    p.covered = np.ones(len(geom.points), dtype=bool)
    wide = GlyphGeometry("T", [], p.points.copy(),
                         xmin=-40.0, xmax=60.0, ymin=0.0, ymax=100.0)
    corr = build_corridors([p], wide)[0]
    # mid-band endpoint far from both edges: with the tiny geom width the
    # endpoint sits ON the x-edge -> side-exit; use the tall geom instead
    corr = build_corridors([_path_from_y(xs, np.full_like(xs, 50.0))], _diag_geom())[0]
    esc = corr.escapes[1]
    if esc.kind == "band":
        run = abs(esc.rows[-1][0] - esc.x_end)
        all_xs = [corr.xa, corr.xb] + [r[0] for r in esc.rows]
    else:
        run = 0.0
        all_xs = [corr.xa, corr.xb]
    for x in all_xs:
        z = (2.0 * x - corr.xa - corr.xb) / (corr.xb - corr.xa)
        assert -1.0 - 1e-9 <= z <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Phase 5: independent corridor adherence + tail re-entry
# ---------------------------------------------------------------------------


def test_emitted_poly_leaving_corridor_rejected():
    xs = np.linspace(1.0, 10.0, 24)
    corr = _corridor_from_bounds(xs, 0.9 * xs, 1.1 * xs)
    good = np.polynomial.Polynomial([0.0, 1.0]).coef      # y = x
    bad = np.polynomial.Polynomial([30.0, -1.0]).coef     # far away
    from src.denysko import corridor_adherence_violation
    assert corridor_adherence_violation(good, corr) <= 0.25 * TAU
    assert corridor_adherence_violation(bad, corr) > 0.25 * TAU


def test_tail_reentry_monotone_passes_reentry_rejected():
    # upward-exit corridor on the right end of a tiny route near ymax
    geom = _diag_geom()
    xs = np.linspace(geom.xmax - 20, geom.xmax - 1.0, 20)
    ys = np.full_like(xs, geom.ymax - 1.0)
    p = _path_from_y(xs, ys)
    p.covered = np.ones(len(geom.points), dtype=bool)
    corr = build_corridors([p], geom)[0]
    esc = corr.escapes[1]
    # route tops out near ymax -> upward edge-exit ramp expected when the
    # endpoint is within TAU of the band edge; otherwise side-exit.
    if esc.kind != "band":
        assert esc.kind == "far"
        return

    def viol_of(coef_list):
        return tail_reentry_violation(np.asarray(coef_list), corr)

    good = list(np.polynomial.Polynomial(
        [geom.ymax + 6.0, 2.0]).coef)
    assert viol_of(good) == 0.0

    bad = list(np.polynomial.Polynomial(
        [geom.ymax + 6.0, 2.0, -1.0]).coef)
    assert viol_of(bad) > 0.0


# ---------------------------------------------------------------------------
# degree minimization invariants / exceptions
# ---------------------------------------------------------------------------


def test_reduce_degree_never_increases(fast_polish):
    xs = np.linspace(0.0, 10.0, 30)
    cubic = ((xs - 5.0) / 5.0) ** 3 * 4.0
    corr = _corridor_from_bounds(xs, cubic - 0.08, cubic + 0.08)
    rng_deg = min_degree(corr)
    assert rng_deg is not None
    assert rng_deg.degree <= INITIAL_FIT_DEGREE
    # verified minimum: everything below is infeasible
    below = fit_degree(corr, rng_deg.degree - 1) \
        if rng_deg.degree > 0 else None
    assert below is None


def test_fit_selected_does_not_swallow_exceptions(monkeypatch):
    geom = _diag_geom()

    def boom(corridor, hi):
        raise RuntimeError("programming error")

    monkeypatch.setattr(d, "min_degree", boom)
    from src.denysko import fit_selected
    with pytest.raises(RuntimeError):
        fit_selected([_FakeCorridorPath(geom)])


class _FakeCorridorPath:
    def __init__(self, geom):
        self.points = geom.points[:10]
        self.contour_id = 0
        self.covered = np.ones(len(geom.points), dtype=bool)


# ---------------------------------------------------------------------------
# geometry cache / O hole
# ---------------------------------------------------------------------------


O_GEOM = glyph_geometry("O")


def test_o_geometry_has_inner_contour_and_ring_cloud():
    assert len(O_GEOM.contours) >= 2
    center = O_GEOM.points.mean(axis=0)
    r = np.hypot(O_GEOM.points[:, 0] - center[0],
                 O_GEOM.points[:, 1] - center[1])
    hist, _ = np.histogram(r, bins=8)
    assert (hist > 30).sum() >= 4


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def test_invalid_cli_input():
    for argv in [
        [], ["a"], ["1"], ["AA"], ["A", "B"],
        ["A", "--seed"],
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
