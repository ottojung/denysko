import numpy as np
import pytest

from src import denysko as d
from src import fitting as _fitting
from src.fitting import (
    ORIENTATIONS,
    Corridor,
    PathFit,
    fit_degree,
    fit_route,
)
from src.topology import (
    ESC_OFFSETS,
    route_edge_ids,
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
        [[(4, 6)]] * 24                 # left trunk
        + [[(2, 4), (6, 8)]] * 24       # splits into two gapped branches
        + [[(4, 6)]] * 24               # ...merging back into right trunk
    )
    graph = build_route_graph(_graph_from_columns(cols))
    kinds = sorted(v.kind for v in graph.vertices)
    assert kinds.count("source") == 1
    assert kinds.count("sink") == 1
    assert kinds.count("split") == 1
    assert kinds.count("merge") == 1
    routes = enumerate_complete_routes(graph)
    chosen = select_routes_min_cover(graph, routes)
    covered = set()
    for j in chosen:
        covered |= set(route_edge_ids(routes[j]))
    assert covered == set(graph.meaningful)
    assert len(chosen) == 2


def test_single_stripe_one_route():
    cols = [[(3, 5)] for _ in range(40)]
    graph = build_route_graph(_graph_from_columns(cols))
    routes = enumerate_complete_routes(graph)
    chosen = select_routes_min_cover(graph, routes)
    assert len(chosen) == 1


def test_two_disjoint_stripes_two_routes():
    cols = [[(0, 1), (8, 9)] for _ in range(40)]
    graph = build_route_graph(_graph_from_columns(cols))
    routes = enumerate_complete_routes(graph)
    chosen = select_routes_min_cover(graph, routes)
    assert len(chosen) == 2


def test_route_corridor_matches_slice_intervals():
    cols = [[(3, 7)] for _ in range(40)]
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
    """A: exactly 2 complete routes sharing both leg trunks - the roof
    route and the bar route (distinct vertical branch choices)."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("A")
    assert len(candidates) >= 2
    assert len(selected) == 2
    r0, r1 = [set(route_edge_ids(x)) for x in chosen]
    shared = r0 & r1
    only0, only1 = r0 - r1, r1 - r0
    assert shared and only0 and only1        # shared trunks, own middles
    # the two differing middle branches live at different heights
    from src.topology import _route_corridor_from_stroke
    ys0 = [_route_corridor_from_stroke(graph, chosen[0], geom).lower.mean()]
    c0 = _route_corridor_from_stroke(graph, tuple(sorted(only0)), geom)
    c1 = _route_corridor_from_stroke(graph, tuple(sorted(only1)), geom)
    assert abs(c0.lower.mean() - c1.lower.mean()) > 10.0
    assert route_coverage_fraction(graph, chosen) >= 0.999


def test_o_topology_is_ring():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("O")
    terms = [v for v in graph.vertices if v.kind == "terminal"]
    assert len(terms) >= 2
    sel = select_routes_min_cover(graph, candidates)
    assert len(sel) == 2


# ---------------------------------------------------------------------------
# Stroke-junction skeleton topology (synthetic masks only - no font)
# ---------------------------------------------------------------------------


def _rect(mask, r0, r1, c0, c1):
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _h_mask():
    m = np.zeros((200, 200), dtype=bool)
    _rect(m, 20, 180, 10, 25)
    _rect(m, 20, 180, 175, 190)
    _rect(m, 95, 110, 25, 176)
    return m


def _t_mask():
    m = np.zeros((200, 200), dtype=bool)
    _rect(m, 15, 30, 10, 190)
    _rect(m, 30, 180, 93, 108)
    return m


def _e_mask():
    m = np.zeros((200, 200), dtype=bool)
    _rect(m, 20, 180, 10, 30)      # spine
    _rect(m, 20, 40, 30, 170)      # top arm
    _rect(m, 92, 108, 30, 150)     # middle arm
    _rect(m, 160, 180, 30, 170)    # bottom arm
    return m


def test_skeleton_h_has_stem_junction_stem_topology():
    from src.skeleton import stroke_graph

    g = stroke_graph(_h_mask())
    kinds = [n.kind for n in g.nodes]
    assert kinds.count("end") == 4        # stem tops and bottoms
    assert kinds.count("junction") == 2   # crossbar meets each stem
    assert len(g.edges) == 5              # 4 stem halves + crossbar
    junc_y = sorted(n.xy[1] for n in g.nodes if n.kind == "junction")
    assert abs(junc_y[0] - junc_y[1]) < 6  # both at crossbar height


def test_skeleton_t_has_one_three_way_junction():
    from src.skeleton import stroke_graph

    g = stroke_graph(_t_mask())
    kinds = [n.kind for n in g.nodes]
    assert kinds.count("end") == 3
    assert kinds.count("junction") == 1
    assert len(g.edges) == 3


def test_skeleton_e_has_multiple_arms():
    from src.skeleton import stroke_graph

    g = stroke_graph(_e_mask())
    ends = [n for n in g.nodes if n.kind == "end"]
    juncs = [n for n in g.nodes if n.kind == "junction"]
    assert len(ends) >= 3                 # arm tips (+ spine end)
    assert len(juncs) >= 1                # middle arm joins spine
    assert len(g.edges) >= 3              # arms + spine chain


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _corridor_from(xs, lower, upper, ylo=0.0, yhi=100.0):
    from src.topology import BoundaryPath

    xs = np.asarray(xs, dtype=float)
    mid = 0.5 * (np.asarray(lower) + np.asarray(upper))
    pad = ESC_OFFSETS[-1] + 1.0
    return Corridor(
        path=BoundaryPath(points=np.column_stack([xs, mid]), contour_id=-1),
        xa=float(xs[0] - pad),
        xb=float(xs[-1] + pad),
        xs=xs,
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        ylo=ylo,
        yhi=yhi,
    )


def _linear_corridor():
    xs = np.linspace(10.0, 60.0, 50)
    mid = 0.5 * xs + 20.0
    return _corridor_from(xs, mid - 1.0, mid + 1.0)


def test_production_lp_smoke(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", True)
    fit = fit_route(_linear_corridor(), hi=20)
    assert fit is not None and 0 < fit.degree <= 20
    assert fit.orientation in ORIENTATIONS


def test_impossible_low_degree_corridor_fails():
    xs = np.linspace(10.0, 60.0, 50)
    mid = 40.0 + 12.0 * np.sin(np.linspace(0, 3 * np.pi, len(xs)))
    c = _corridor_from(xs, mid - 2.0, mid + 2.0)
    fit0 = fit_degree(c, 0, 1, 1)
    assert fit0 is None  # constant cannot follow a sine tube nor escape
    best = fit_route(c, hi=24)
    assert best is not None and best.degree > 0


def test_degree_minimization_verified_minimum():
    """fit_route must return the LOWEST verified feasible degree: every
    lower degree is infeasible for every tail orientation."""
    c = _linear_corridor()
    fit = fit_route(c, hi=24)
    assert fit is not None
    for dd in range(fit.degree):
        assert all(
            fit_degree(c, dd, *ori) is None for ori in ORIENTATIONS
        )


# ---------------------------------------------------------------------------
# Mandatory tail escape (V3) and orientation choice
# ---------------------------------------------------------------------------


def _slab_corridor(y_lo=49.0, y_hi=51.0):
    xs = np.linspace(10.0, 60.0, 30)
    return _corridor_from(xs, np.full(len(xs), y_lo),
                          np.full(len(xs), y_hi))


def test_constant_line_v2_passes_v3_fails():
    """P(x)=50 inside a slab corridor: perfect V2 adherence, but its
    tails stay horizontal forever - V3 must reject it."""
    corr = _slab_corridor()
    coef = np.array([50.0])
    v2 = d.corridor_adherence_violation(coef, corr)
    assert v2 <= 0.35
    for ori in ORIENTATIONS:
        assert d.tail_reentry_violation(coef, corr, ori) > 0

    class _Fit:
        poly = np.polynomial.Polynomial(coef)
        orientation = (1, -1)

    problems = d.validate_lines(["y=50"], object(), [_Fit()], [corr])
    assert any(p.startswith("V2") is False and p.startswith("V3")
               for p in problems)


def test_escaping_tails_pass_v3():
    corr = _slab_corridor()
    # left-down AND right-up in one stroke: steep S-line through the band
    s_line = np.polynomial.Polynomial([-45.0, 3.0])
    assert s_line(10.0) < 0.0 and s_line(60.0) > 100.0
    assert d.tail_reentry_violation(s_line.coef, corr, (-1, 1)) == 0.0
    # constant already below the band: permanently outside on both sides
    sunk = np.polynomial.Polynomial([-5.0])
    assert d.tail_reentry_violation(sunk.coef, corr, (-1, -1)) == 0.0


def test_reentry_and_wrong_asymptote_fail_v3():
    corr = _slab_corridor()
    u = np.polynomial.Polynomial([-60.0, 1.0])          # u = x - 60
    # outside the band at the checkpoint (P(60)>100); its derivative has
    # roots at u=6 (local max, stays out) and u=40 (local min dipping
    # back under yhi=100):
    k = 0.01
    dip = 110.0 + k * (u ** 3 / 3.0 - 23.0 * u ** 2 + 240.0 * u)
    assert dip(60.0) > 100.0
    rts = [r + 60.0 for r in (6.0, 40.0)]
    assert float(dip(rts[0])) > 100.0
    assert float(dip(rts[1])) < 100.0
    assert d.tail_reentry_violation(dip.coef, corr, (1, 1)) > 0

    # wrong asymptote: outside at both checkpoints, but the parabola
    # opens downward so it must fall back through the band eventually
    wrong = 110.0 + 30 * u - u ** 2
    assert wrong(60.0) > 100.0
    assert d.tail_reentry_violation(wrong.coef, corr, (1, 1)) > 0


def test_orientation_choice_prefers_feasible_low_degree():
    # a corridor hugging the top of the band: only an UP-right tail can
    # escape quickly; fit_route must find some feasible orientation.
    xs = np.linspace(10.0, 60.0, 40)
    c = _corridor_from(xs, np.full(len(xs), 92.0), np.full(len(xs), 98.0))
    fit = fit_route(c, hi=20)
    assert fit is not None
    sig_l, sig_r = fit.orientation
    assert sig_r == 1   # downward from y~97 would fight the ramp rows


def test_emitted_poly_leaving_corridor_rejected():
    xs = np.linspace(10.0, 60.0, 30)
    c = _corridor_from(xs, np.full(len(xs), 20.0), np.full(len(xs), 22.0))
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
    c = _corridor_from(xs, np.full(len(xs), 20.0), np.full(len(xs), 22.0))

    class _Fit:
        poly = np.polynomial.Polynomial([21.0])
        orientation = (1, -1)

    lines = [d.format_expression(_Fit())]
    problems = d.validate_lines(lines, _Geom(), [_Fit()], [c])
    # V2 clean, but the constant tail cannot escape: V3 flags it
    assert not any(p.startswith("V2") or p.startswith("V4") for p in problems)
    assert any(p.startswith("V3") for p in problems)

    class _BadFit:
        poly = np.polynomial.Polynomial([50.0])
        orientation = (1, -1)

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


def test_stale_pinch_branch_terminates():
    """A branch that vanishes for more than PINCH_COLS columns must get
    its own sink at its disappearance point - never reconnect later or
    survive to the global right edge."""
    from src.topology import PINCH_COLS

    gap = PINCH_COLS + 2
    cols = (
        [[(4, 6)]] * 8
        + [[]] * gap                    # branch disappears too long
        + [[(2, 4), (6, 8)]] * 24       # unrelated later structure
        + [[(4, 6)]] * 8
    )
    graph = build_route_graph(_graph_from_columns(cols))
    step = 100.0 / 512
    gap_lo = 8 * step
    gap_hi = (8 + gap) * step
    sinks_mid = [v for v in graph.vertices
                 if v.kind == "sink" and v.x < gap_hi]
    assert sinks_mid                    # explicit disappearance sink
    assert all(v.x < gap_hi for v in sinks_mid)
    for e in graph.edges:               # no edge crosses the empty gap
        assert not (e.xs[0] < gap_lo and e.xs[-1] > gap_hi)
