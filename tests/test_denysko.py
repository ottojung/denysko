import numpy as np
import pytest

from src import denysko as d
from src import fitting as _fitting
from src.fitting import (
    Corridor,
    PathFit,
    fit_degree,
    min_degree,
)
from src.topology import (
    GlyphGeometry,
    RouteEdge,
    RouteGraph,
    RouteVertex,
    SliceInterval,
    build_route_corridor,
    build_route_graph,
    enumerate_complete_routes,
    glyph_geometry,
    route_coverage_fraction,
    select_routes_min_cover,
)


@pytest.fixture
def fast_polish(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", False)


# ---------------------------------------------------------------------------
# Serialization contract (V4)
# ---------------------------------------------------------------------------


def test_serialize_parse_roundtrip_exact():
    coef = np.array([0.5, -1.25, 3.0])
    line = d.format_expression(
        type("C", (), {"poly": np.polynomial.Polynomial(coef)})()
    )
    parsed = d.parse_line(line)
    assert parsed is not None
    again = d.serialize(parsed)
    assert again == line


def test_fmt_num_round_trips_tiny_coefficients():
    for v in (0.0, 1e-13, -1e-13, 1.23456789012345e-7):
        s = d.fmt_num(v)
        assert abs(float(s) - v) <= 1e-15 * max(1.0, abs(v))


def test_malformed_lines_do_not_parse():
    assert d.parse_line("x=y+1") is None
    assert d.parse_line("y=") is None
    assert d.parse_line("y=..") is None
    assert d.parse_line("") is None


# ---------------------------------------------------------------------------
# Synthetic routing-graph topologies
# ---------------------------------------------------------------------------


SCALE = 40  # synthetic row unit -> ~512-row raster convention


def _graph_from_columns(cols):
    """Build a glyph geometry whose fill mask is given by explicit runs
    per column: cols is a list of [(r0, r1), ...] with r increasing.
    Rows are scaled so slice heights clear the meaningful-edge floor."""
    top = max((r for c in cols for _, r in c), default=0) + 1
    rows = top * SCALE + 2
    w = len(cols)
    fill = np.zeros((rows, w), dtype=bool)
    for i, runs in enumerate(cols):
        for a, b in runs:
            fill[a * SCALE : b * SCALE + SCALE + 1, i] = True
    geom = GlyphGeometry(
        letter="?",
        points=np.zeros((0, 2)),
        contours=[],
        fill=fill,
        xmin=0.0,
        xmax=float(w),
        ymin=0.0,
        ymax=float(rows),
    )
    return geom


def test_diamond_split_merge_two_routes():
    # one trunk splitting into two branches then merging (A / O shape)
    cols = (
        [[(4, 6)]] * 4                  # left trunk
        + [[(2, 4), (6, 8)]] * 4        # splits into two gapped branches
        + [[(4, 6)]] * 4                # ...merging back into right trunk
    )
    graph = build_route_graph(_graph_from_columns(cols))
    kinds = sorted(v.kind for v in graph.vertices)
    assert kinds.count("source") == 1
    assert kinds.count("sink") == 1
    assert kinds.count("split") == 1
    assert kinds.count("merge") == 1
    routes = enumerate_complete_routes(graph)
    assert len(routes) == 2
    chosen = select_routes_min_cover(graph, routes)
    covered = set()
    for j in chosen:
        covered |= set(routes[j])
    assert covered == set(graph.meaningful)


def test_single_stripe_one_route():
    cols = [[(3, 5)] for _ in range(10)]
    graph = build_route_graph(_graph_from_columns(cols))
    routes = enumerate_complete_routes(graph)
    assert len(routes) == 1
    chosen = select_routes_min_cover(graph, routes)
    assert len(chosen) == 1


def test_two_disjoint_stripes_two_routes():
    cols = [[(0, 1), (8, 9)] for _ in range(10)]
    graph = build_route_graph(_graph_from_columns(cols))
    routes = enumerate_complete_routes(graph)
    assert len(routes) == 2
    chosen = select_routes_min_cover(graph, routes)
    assert len(chosen) == 2


def test_route_corridor_matches_slice_intervals():
    cols = [[(3, 7)] for _ in range(8)]
    geom = _graph_from_columns(cols)
    graph = build_route_graph(geom)
    routes = enumerate_complete_routes(graph)
    from src.topology import CORRIDOR_MARGIN

    corr = build_route_corridor(graph, routes[0], geom)
    step = 100.0 / 512
    np.testing.assert_allclose(
        corr.lower, 3 * SCALE * step + CORRIDOR_MARGIN, rtol=1e-9)
    np.testing.assert_allclose(
        corr.upper, (8 * SCALE + 1) * step - CORRIDOR_MARGIN, rtol=1e-9)


# ---------------------------------------------------------------------------
# Real-glyph topology (deterministic font data)
# ---------------------------------------------------------------------------


def test_a_topology_is_diamond():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("A")
    kinds = sorted(v.kind for v in graph.vertices)
    assert kinds == ["merge", "sink", "source", "split"]
    assert len(candidates) == 2
    assert len(selected) == 2
    assert route_coverage_fraction(graph, chosen) >= 0.999


def test_o_topology_is_ring():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("O")
    kinds = sorted(v.kind for v in graph.vertices)
    assert kinds == ["merge", "sink", "source", "split"]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _linear_corridor():
    xs = np.linspace(10.0, 60.0, 50)
    mid = 0.5 * xs + 20.0
    from src.topology import BoundaryPath

    return Corridor(
        path=BoundaryPath(points=np.column_stack([xs, mid]), contour_id=-1),
        xa=float(xs[0] - 2),
        xb=float(xs[-1] + 2),
        xs=xs,
        lower=mid - 0.3,
        upper=mid + 0.3,
        escapes=(),
    )


def test_production_lp_smoke(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", True)
    fit = min_degree(_linear_corridor(), hi=8)
    assert fit is not None and fit.degree <= 2


def test_simple_linear_corridor_feasible(fast_polish):
    fit = min_degree(_linear_corridor(), hi=8)
    assert fit is not None and fit.degree <= 2


def test_impossible_low_degree_corridor_fails(fast_polish):
    from src.topology import BoundaryPath

    xs = np.linspace(10.0, 60.0, 50)
    mid = 40.0 + 25.0 * np.sin(np.linspace(0, 6 * np.pi, len(xs)))
    c = Corridor(
        path=BoundaryPath(points=np.column_stack([xs, mid]), contour_id=-1),
        xa=float(xs[0] - 2),
        xb=float(xs[-1] + 2),
        xs=xs,
        lower=mid - 0.3,
        upper=mid + 0.3,
        escapes=(),
    )
    fit0 = fit_degree(c, 0)
    assert fit0 is None  # constant cannot follow a sine tube
    best = min_degree(c, hi=24)
    assert best is not None and best.degree > 0


def test_degree_minimization_verified_minimum(fast_polish):
    fit = min_degree(_linear_corridor(), hi=10)
    assert fit.degree == 1


def test_emitted_poly_leaving_corridor_rejected():
    xs = np.linspace(10.0, 60.0, 30)
    c = Corridor(
        path=None,
        xa=8.0,
        xb=62.0,
        xs=xs,
        lower=np.full(len(xs), 20.0),
        upper=np.full(len(xs), 22.0),
        escapes=(),
    )
    bad = np.polynomial.Polynomial([30.0])
    v = d.corridor_adherence_violation(bad.coef, c)
    assert v > 1.0
    good = np.polynomial.Polynomial([21.0])
    assert d.corridor_adherence_violation(good.coef, c) == pytest.approx(
        0.0, abs=1e-9
    )


def test_validate_lines_flags_violations():
    class _Geom:
        pass

    xs = np.linspace(10.0, 60.0, 30)
    c = Corridor(
        path=None,
        xa=8.0,
        xb=62.0,
        xs=xs,
        lower=np.full(len(xs), 20.0),
        upper=np.full(len(xs), 22.0),
        escapes=(),
    )

    class _Fit:
        poly = np.polynomial.Polynomial([21.0])

    lines = [d.format_expression(_Fit())]
    assert d.validate_lines(lines, _Geom(), [_Fit()], [c]) == []

    class _BadFit:
        poly = np.polynomial.Polynomial([50.0])

    bad_lines = [d.format_expression(_BadFit())]
    problems = d.validate_lines(bad_lines, _Geom(), [_BadFit()], [c])
    assert any(p.startswith("V2") for p in problems)


# ---------------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------------


def test_invalid_cli_input():
    assert d.run(["1"]) == 2
    assert d.run(["a"]) == 2
    assert d.run([]) == 2
    assert d.run(["AA"]) == 2
    assert d.run(["--bogus", "A"]) == 2


def test_entry_propagates_exit_code(monkeypatch, capsys):
    from src.__main__ import entry

    monkeypatch.setattr("sys.argv", ["denysko", "A"])
    with pytest.raises(SystemExit) as ei:
        entry()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert all(line.startswith("y=") for line in out.splitlines())
