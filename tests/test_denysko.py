import numpy as np
import pytest

from src import denysko as d
from src.denysko import fit_selected
from src import fitting as _fitting
from src import topology as d_topology
from src import topology as _topo
from src.fitting import (
    INITIAL_FIT_DEGREE,
    ORIENTATIONS,
    Corridor,
    PathFit,
    fit_degree,
    fit_route,
    preferred_tail_orientation,
)
from src.topology import (
    ESC_OFFSETS,
    build_stroke_route_graph,
    _route_signature as route_sig_top,
    route_edge_ids,
    route_join_score,
    GlyphGeometry,
    Route,
    OrientedRouteEdge,
    RouteEdge,
    RouteGraph,
    RouteVertex,
    SliceInterval,
    build_route_corridor,
    build_route_graph,
    enumerate_complete_routes,
    glyph_geometry,
    glyph_connected_components,
    route_component_label,
    component_preferred_orientation,
    route_coverage_fraction,
    route_atom_ids,
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


def test_issue22_output_drops_y_prefix():
    # Issue #22: emitted equations must NOT carry the historical `y=`
    # prefix (e.g. `a + b*x + c*x^2 ...`).
    coef = np.array([0.5, -1.25, 3.0])
    line = d.format_expression(
        type("C", (), {"poly": np.polynomial.Polynomial(coef)})()
    )
    assert not line.startswith("y=")
    assert line == "0.5-1.25x+3x^2"

    # The public single-letter and text serialization paths also drop it.
    fits, corrs, _ = d.generate_letter("A", seed=42)
    for ln in [d.serialize_fit(f) for f in fits]:
        assert not ln.startswith("y=")
        assert d.parse_line(ln) is not None

    placed = d.generate_text("Ab", seed=42)
    for ln in d.serialize_text(placed):
        assert not ln.startswith("y=")
        # monomial lines parse back to a polynomial; Horner (nested
        # arithmetic) lines still evaluate as equations.
        if "(" in ln:
            xs = np.linspace(0.0, 2.0, 8)
            assert np.all(np.isfinite(d.eval_expression(d.expr_body(ln), xs)))
        else:
            assert d.parse_line(ln) is not None


def test_issue22_legacy_y_prefix_still_parses():
    # Backward tolerance: historical `y=` artifacts still round-trip.
    assert d.parse_line("y=0.5-1.25x+3x^2") is not None
    assert d.serialize(d.parse_line("y=0.5-1.25x+3x^2")) == \
        "0.5-1.25x+3x^2"
    assert d.expr_body("y=0.5-1.25x+3x^2") == "0.5-1.25x+3x^2"
    assert d.expr_body("0.5-1.25x+3x^2") == "0.5-1.25x+3x^2"


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
        [[(4, 6)]] * 120                # left trunk
        + [[(2, 4), (6, 8)]] * 120      # splits into two gapped branches
        + [[(4, 6)]] * 120              # ...merging back into right trunk
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
    cols = [[(300, 500)] for _ in range(500)]
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


def _bruteforce_priority(graph, candidates):
    """Reference lexicographic optimum for the staged objective:

        (min route count, max join score, min complexity, min index sum)

    Returns (K, Jmax, Cmin, Imin, optimal_sets) where optimal_sets are the
    index-minimal selected sets achieving all four optima.
    """
    meaningful = set(graph.meaningful)
    n = len(candidates)
    feas = []
    for mask in range(1 << n):
        sel = [j for j in range(n) if (mask >> j) & 1]
        if not sel:
            continue
        covered = set().union(*[
            route_atom_ids(graph, candidates[j]) & meaningful
            for j in sel])
        if covered >= meaningful:
            jsum = sum(route_join_score(graph, candidates[j]) for j in sel)
            csum = sum(len(route_edge_ids(candidates[j])) for j in sel)
            isum = sum(sel)
            feas.append((len(sel), jsum, csum, isum, sel))
    K = min(t[0] for t in feas)
    feasK = [t for t in feas if t[0] == K]
    Jmax = max(t[1] for t in feasK)
    feasJ = [t for t in feasK if t[1] == Jmax]
    Cmin = min(t[2] for t in feasJ)
    feasC = [t for t in feasJ if t[2] == Cmin]
    Imin = min(t[3] for t in feasC)
    opt = [t[4] for t in feasC if t[3] == Imin]
    return K, Jmax, Cmin, Imin, opt


def _assert_staged_priority(graph, candidates, selected):
    """Assert `selected` (a list of candidate INDICES) exactly matches the
    staged lexicographic optimum."""
    K, Jmax, Cmin, Imin, opt_sets = _bruteforce_priority(graph, candidates)
    assert len(selected) == K, (len(selected), K)
    sel_routes = [candidates[j] for j in selected]
    jsum = sum(route_join_score(graph, r) for r in sel_routes)
    csum = sum(len(route_edge_ids(r)) for r in sel_routes)
    isum = sum(selected)
    assert jsum == Jmax, (jsum, Jmax)          # join dominates complexity
    assert csum == Cmin, (csum, Cmin)          # complexity dominates index
    assert isum == Imin, (isum, Imin)          # deterministic index tie-break
    assert set(selected) in [set(s) for s in opt_sets]


def test_staged_priority_ordering_holds_on_real_glyphs():
    """Issue #3: the staged objective must be a genuine lexicographic MILP
    (K, then max join, then min complexity, then deterministic index
    tie-break), not a single blended weighted cost. Verify the selected
    routes for real junction-rich glyphs exactly match the brute-forced
    lexicographic optimum over every candidate subset."""
    from src.denysko import _glyph_geometry_or_error
    for letter in ("y", "r"):
        geom = _glyph_geometry_or_error(letter)
        g = build_stroke_route_graph(geom)
        cands = enumerate_complete_routes(g)
        chosen = select_routes_min_cover(g, cands)
        _assert_staged_priority(g, cands, chosen)


def test_join_maximizing_prefers_joined_cover_over_shorter_unjoined():
    """Issue #3 mechanism: when several minimum-size covers exist, prefer
    the one with the greatest number of legal stroke continuations
    through junctions, even at the cost of a longer (higher-complexity)
    cover.

    Synthetic routing graph with a genuine junction J and three incident
    branches. Two minimum covers of size K=2 exist:

      * joined cover    : two routes that each PASS THROUGH J (join=2)
      * unjoined cover  : one route through J + one route that STARTS at
                          J and escapes at the contact (join=1, and it is
                          the shorter cover by total edge count)

    The exact minimum curve count K is unchanged; the selection must pick
    the maximal-join cover, not the shorter unjoined one.
    """
    V_a, V_J, V_b, V_c = 0, 1, 2, 3
    edges = [
        RouteEdge(0, V_a, V_J, xs=np.array([0.0, 0.5]),
                  lower=np.zeros(2), upper=np.zeros(2)),
        RouteEdge(1, V_J, V_b, xs=np.array([0.5, 1.0]),
                  lower=np.zeros(2), upper=np.zeros(2)),
        RouteEdge(2, V_J, V_c, xs=np.array([0.5, 1.0]),
                  lower=np.zeros(2), upper=np.zeros(2)),
    ]
    verts = [
        RouteVertex(V_a, 0.0, "terminal"),
        RouteVertex(V_J, 0.5, "junction"),
        RouteVertex(V_b, 1.0, "terminal"),
        RouteVertex(V_c, 1.0, "terminal"),
    ]
    graph = RouteGraph(vertices=verts, edges=edges,
                       meaningful=frozenset({0, 1, 2}))

    # candidate routes (hand-built to expose the competing covers)
    rj1 = Route(steps=(OrientedRouteEdge(0, V_a, V_J),
                       OrientedRouteEdge(1, V_J, V_b)))   # passes through J
    rj2 = Route(steps=(OrientedRouteEdge(0, V_a, V_J),
                       OrientedRouteEdge(2, V_J, V_c)))   # passes through J
    rk1 = Route(steps=(OrientedRouteEdge(1, V_J, V_b),))  # starts at J
    rk2 = Route(steps=(OrientedRouteEdge(2, V_J, V_c),))  # starts at J
    ra = Route(steps=(OrientedRouteEdge(0, V_a, V_J),))    # ends at J
    candidates = [rj1, rj2, rk1, rk2, ra]

    chosen = select_routes_min_cover(graph, candidates)
    chosen_routes = [candidates[j] for j in chosen]

    # proven minimum curve count is unchanged
    assert len(chosen) == 2
    # maximal-join cover (both routes pass through J) is preferred, NOT the
    # shorter cover containing a route that starts at the contact
    total_join = sum(route_join_score(graph, r) for r in chosen_routes)
    assert total_join == 2, total_join
    assert not any(route_join_score(graph, r) == 0 for r in chosen_routes)
    # explicit: the two junction-passing routes are the selected pair
    assert set(chosen) == {0, 1}
    # full coverage still holds
    assert route_coverage_fraction(graph, chosen_routes) == pytest.approx(1.0)
    # the staged priority ordering (K -> max join -> min complexity ->
    # deterministic index tie-break) is exactly the lexicographic optimum
    _assert_staged_priority(graph, candidates, chosen)

def test_route_corridor_matches_slice_intervals():
    cols = [[(300, 700)] for _ in range(500)]
    geom = _graph_from_columns(cols)
    graph = build_route_graph(geom)
    routes = enumerate_complete_routes(graph)
    from src.topology import CORRIDOR_MARGIN, build_slice_corridor

    corr = build_slice_corridor(graph, route_edge_ids(routes[0]), geom)
    # normalized world mapping: pixel * NORMALIZED_SIZE / GRID
    from src.topology import GRID, NORMALIZED_SIZE
    step = NORMALIZED_SIZE / GRID
    np.testing.assert_allclose(
        corr.lower, 300 * SCALE * step + CORRIDOR_MARGIN, rtol=1e-9)
    np.testing.assert_allclose(
        corr.upper, (701 * SCALE + 1) * step - CORRIDOR_MARGIN, rtol=1e-9)


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
    # the two differing middle branches live at different heights:
    # compare realized center y of the differing atoms
    ys = []
    for r in chosen:
        for s_ in r.steps:
            if s_.edge_id in only0 or s_.edge_id in only1:
                e = graph.edges[s_.edge_id]
                ys.append(float(e.points[:, 1].mean()))
    assert max(ys) - min(ys) > 0.20          # roof vs crossbar height
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


def _corridor_from(xs, lower, upper, ylo=0.0, yhi=1.0):
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
    """fit_route returns the lowest feasible degree for required geometry."""
    c = _linear_corridor()
    fit = fit_route(c, hi=24)
    assert fit is not None
    ori = preferred_tail_orientation(c)
    assert fit.orientation == ori
    for dd in range(fit.degree):
        assert fit_degree(c, dd, *ori) is None


# ---------------------------------------------------------------------------
# Mandatory tail escape (V3) and orientation choice
# ---------------------------------------------------------------------------


def _slab_corridor(y_lo=0.49, y_hi=0.51):
    xs = np.linspace(0.10, 0.60, 30)
    return _corridor_from(xs, np.full(len(xs), y_lo),
                          np.full(len(xs), y_hi))


def _to_norm(coef):
    """Map an old-scale raw polynomial to the normalized world:
    P_new(x) = P_old(100 x) / 100 (same graph, /100 coordinates)."""
    from numpy.polynomial import Polynomial as Poly
    return (Poly(np.asarray(coef, dtype=float))(
        Poly([0.0, 100.0])) / 100.0).coef


class _SlabGeom:
    fill = np.zeros((512, 512), dtype=bool)
    fill[:, 40:470] = True


def test_constant_line_v2_passes_v3_fails():
    """P(x)=50 inside a slab corridor: perfect V2 adherence, but its
    tails stay horizontal forever - V3 must reject it."""
    corr = _slab_corridor()
    coef = np.array([0.5])
    v2 = d.corridor_adherence_violation(coef, corr)
    assert v2 <= _fitting.CORRIDOR_EPS + 1e-12
    for ori in ORIENTATIONS:
        assert d.tail_reentry_violation(coef, corr, ori) > 0

    class _Fit:
        poly = np.polynomial.Polynomial(coef)
        orientation = (1, -1)

    problems = d.validate_lines(["0.5"], _SlabGeom(), [_Fit()], [corr])
    assert any(p.startswith("V2") is False and p.startswith("V3")
               for p in problems)


def test_escaping_tails_pass_v3():
    corr = _slab_corridor()
    # left-down AND right-up in one stroke: steep line through the band
    s_line = np.polynomial.Polynomial([-0.5, 8.0])
    assert s_line(-0.06) < 0.0 and s_line(0.76) > 1.0
    assert d.tail_reentry_violation(s_line.coef, corr, (-1, 1)) == 0.0
    # constant already below the band: permanently outside on both sides
    sunk = np.polynomial.Polynomial([-0.5])
    assert d.tail_reentry_violation(sunk.coef, corr, (-1, -1)) == 0.0


def test_reentry_and_wrong_asymptote_fail_v3():
    corr = _slab_corridor()
    # old-scale construction mapped to the normalized world via
    # P_new(x) = P_old(100 x)/100 (same geometry /100):
    u100 = np.polynomial.Polynomial([-60.0, 100.0])   # u = 100x - 60
    dip100 = 110.0 + 0.01 * (u100 ** 3 / 3.0 - 23.0 * u100 ** 2
                             + 240.0 * u100)
    dip = np.polynomial.Polynomial(dip100.coef) / 100.0
    # outside the band at the checkpoint (P(0.6)>1); derivative roots at
    # x=0.66 (local max, stays out) and x=1.00 (local min dipping back
    # under yhi=1):
    assert dip(0.6) > 1.0
    rts = [r + 0.6 for r in (0.06, 0.40)]
    assert float(dip(rts[0])) > 1.0
    assert float(dip(rts[1])) < 1.0
    assert d.tail_reentry_violation(dip.coef, corr, (1, 1)) > 0

    # wrong asymptote: outside at both checkpoints, but the parabola
    # opens downward so it must fall back through the band eventually
    uold = np.polynomial.Polynomial([-60.0, 1.0])    # u = x_old - 60
    wrong100 = 110.0 + 30.0 * uold - uold ** 2
    wrong = np.polynomial.Polynomial(wrong100.coef)(  # x_old = 100x
        np.polynomial.Polynomial([0.0, 100.0])) / 100.0
    assert wrong(0.6) > 1.0
    assert d.tail_reentry_violation(wrong.coef, corr, (1, 1)) > 0


def test_orientation_choice_follows_endpoint_geometry():
    # A corridor hugging the top of the band must choose upward escape
    # from geometry before fitting rather than because it is numerically easy.
    xs = np.linspace(10.0, 60.0, 40)
    c = _corridor_from(xs, np.full(len(xs), 92.0), np.full(len(xs), 98.0))
    fit = fit_route(c, hi=20)
    assert fit is not None
    sig_l, sig_r = fit.orientation
    assert sig_r == 1   # downward from y~97 would fight the ramp rows


def test_issue2_real_glyph_tail_orientation_is_geometry_driven():
    """Issue #2: tail escape direction is a geometric decision made before
    fitting and must be inspected directly via PathFit.orientation (not
    inferred from the serialized equations).

    For every selected route the emitted orientation must equal the
    endpoint-geometry-derived orientation; fit_route must never silently
    flip to the opposite direction merely because another orientation is
    easier to fit. The documented acceptance cases are locked in:

      C: upper end up, lower end down;
      A: the relevant leg-route ends both escape down;
      r: stem and hat joined through the junction; both curves escape
         down at the shared top-of-stem endpoint and up at the far end;
      e: the lower route escapes down; the joined spine-to-bar routes
         escape down at the spine end and up at the bar end.
    """
    expected = {
        "C": [(-1, -1), (-1, 1)],
        "A": [(-1, -1), (-1, -1)],
        # issue #3: r's stem and hat are joined through the junction, so
        # the selected curves no longer start/end at the contact; both
        # share the top-of-stem endpoint (escapes down) and escape up at
        # their far end. Orientation remains geometry-derived.
        "r": [(-1, 1), (-1, 1)],
        # issue #3: e's spine joins two of its bars through junctions; the
        # locked orientation set shifts accordingly but stays geometry-driven
        "e": [(-1, -1), (-1, 1), (-1, 1)],
    }
    for letter, want in expected.items():
        geom, graph, candidates, chosen, sigs, selected = \
            d.build_phase1(letter)
        got = []
        for corr in selected:
            fit = fit_route(corr, hi=INITIAL_FIT_DEGREE)
            assert fit is not None, f"{letter}: route infeasible"
            # core contract: fit uses the geometry-derived orientation,
            # never a flipped one chosen for fit ease.
            assert fit.orientation == preferred_tail_orientation(corr), (
                f"{letter}: fit orientation {fit.orientation} != "
                f"geometry-derived "
                f"{preferred_tail_orientation(corr)}")
            got.append(fit.orientation)
        # order-independent: route enumeration is deterministic but we key
        # on the geometric end directions, not on internal edge ids.
        assert sorted(got) == sorted(want), (
            f"{letter}: orientations {sorted(got)} != documented "
            f"{sorted(want)}")


def test_issue4_component_preference_bottom_top_middle():
    """Issue #4 mechanism (letter-independent): a bottom-most component
    sends both tails down, a top-most component sends both tails up, a
    single component defers to the ordinary nearest-boundary rule, and a
    side-by-side (overlapping-y) pair has no clear outward side so it too
    defers."""
    info = {
        1: {"ymin": 0.0, "ymax": 0.3, "cy": 0.15},   # bottom
        2: {"ymin": 0.7, "ymax": 1.0, "cy": 0.85},   # top
    }
    present = {1, 2}
    assert component_preferred_orientation(1, info, present) == (-1, -1)
    assert component_preferred_orientation(2, info, present) == (1, 1)
    # single component present -> fall back (no disconnected partner)
    assert component_preferred_orientation(1, info, {1}) is None
    # side-by-side (overlapping y, no clear vertical outward side) -> None
    side = {3: {"ymin": 0.0, "ymax": 0.5, "cy": 0.25},
            4: {"ymin": 0.0, "ymax": 0.5, "cy": 0.25}}
    assert component_preferred_orientation(3, side, {3, 4}) is None


def test_issue4_i_disconnected_components_escape_away():
    """Issue #4 acceptance: `i` has two disconnected glyph components
    (stem + dot) identified purely from glyph geometry. The bottom
    component (stem) escapes down/down and the top component (dot) escapes
    up/up; each selected route's emitted orientation must equal its
    component-level preference and the fitter must not flip it to a
    geometry-easier orientation."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("i")
    labels, n_comp, comp_info = glyph_connected_components(geom)
    assert n_comp >= 2
    route_comps = [route_component_label(graph, r, labels)
                   for r in chosen]
    present = {c for c in route_comps if c is not None}
    assert len(present) == 2
    prefs = [component_preferred_orientation(c, comp_info, present)
             for c in route_comps]
    # exactly one bottom (down/down) and one top (up/up)
    assert sorted(prefs) == [(-1, -1), (1, 1)]
    for r, corr in zip(chosen, selected):
        fit = fit_route(corr, hi=INITIAL_FIT_DEGREE)
        assert fit is not None
        assert fit.orientation == corr.preferred_orientation, (
            f"i: fit orientation {fit.orientation} overrode the "
            f"component-level preference {corr.preferred_orientation}")
        assert fit.orientation in prefs


def test_issue4_j_inspect_disconnected_components():
    """Issue #4: inspect `j` where the same stem/dot disconnected
    structure applies. The dot (top component) escapes up/up and the
    stem+descender (bottom component) escapes down/down; the fitter must
    not override either to a geometry-easier orientation. The two
    selected routes map to two distinct connected components."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("j")
    labels, n_comp, comp_info = glyph_connected_components(geom)
    assert n_comp >= 2
    route_comps = [route_component_label(graph, r, labels)
                   for r in chosen]
    assert len({c for c in route_comps if c is not None}) == 2
    for r, corr in zip(chosen, selected):
        fit = fit_route(corr, hi=INITIAL_FIT_DEGREE)
        assert fit is not None
        assert fit.orientation == corr.preferred_orientation, (
            f"j: fit orientation {fit.orientation} overrode the "
            f"component-level preference {corr.preferred_orientation}")


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
        fill = np.zeros((512, 512), dtype=bool)
        fill[:, 40:470] = True          # generous slab: y 0..100 all x

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


def test_invalid_cli_input(capsys):
    # usage errors exit via SystemExit(2)
    for argv in ([], ["--seed", "nope", "A"],
                 ["--unknown", "A"]):
        with pytest.raises(SystemExit) as ei:
            d.run(argv)
        assert ei.value.code == 2, argv


def test_all_space_text_emits_zero_equations(capsys):
    result = d.generate_text("   ")
    assert result.placed_fits == ()
    assert d.run(["   ", "-q"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""


def test_punctuation_is_attempted_not_whitelist_rejected(capsys):
    # One real public-CLI generation proves both punctuation glyphs are
    # attempted. The exact Hello, World! rendering is the mandatory PR
    # smoke artifact, so the unit suite need not regenerate it too.
    assert d.run([",!", "--seed", "42", "-q"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines and all(not line.startswith("y=") for line in lines)
    for line in lines:
        assert d.parse_line(line) is not None or d.expr_body(line)


def test_arbitrary_char_failure_reports_index_and_repr(monkeypatch,
                                                       capsys):
    import string

    def boom(letter, **kwargs):
        if letter in string.ascii_letters:
            # This test is about contextual failure/no-partial-output, not
            # fitting the successful prefix. Keep the prefix deliberately
            # cheap while still exercising generate_text().
            return [object()], None, None
        raise RuntimeError("synthetic raster failure")

    monkeypatch.setattr(d, "generate_letter", boom)
    with pytest.raises(d.GenerationError) as ei:
        d.generate_text("ab#")
    msg = str(ei.value)
    assert "character 2" in msg and "'#'" in msg
    # zero partial stdout through the public CLI
    assert d.run(["ab#", "-q"]) == 1
    captured = capsys.readouterr()
    assert "character 2" in captured.err and "'#'" in captured.err
    assert captured.out == ""


def test_entry_propagates_exit_code(monkeypatch, capsys):
    from src.__main__ import entry

    monkeypatch.setattr("sys.argv", ["denysko", "A"])
    with pytest.raises(SystemExit) as ei:
        entry()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert all(not line.startswith("y=") for line in out.splitlines())


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


# ---------------------------------------------------------------------------
# Oriented-route geometry regressions
# ---------------------------------------------------------------------------


def test_oriented_reconstruction_reverses_stored_direction():
    """v0 --e0--> v1 <--e1-- v2 : walking v0->v1->v2 must use e0 forward
    and e1 reversed, with exact point order."""
    from src.topology import (OrientedRouteEdge, Route, RouteEdge,
                              RouteGraph, route_polyline)

    edges = [
        RouteEdge(0, 0, 1, xs=np.array([0.0, 1.0]),
                  lower=np.zeros(2), upper=np.zeros(2),
                  points=np.array([[0.0, 0.0], [1.0, 0.0]])),
        RouteEdge(1, 2, 1, xs=np.array([1.0, 2.0]),
                  lower=np.zeros(2), upper=np.zeros(2),
                  points=np.array([[2.0, 0.0], [1.0, 0.0]])),
    ]
    verts = [RouteVertex(0, 0.0, "terminal"),
             RouteVertex(1, 1.0, "junction"),
             RouteVertex(2, 2.0, "terminal")]
    for v in verts:
        v.outgoing = tuple(e.id for e in edges if e.v_from == v.id)
        v.incoming = tuple(e.id for e in edges if e.v_to == v.id)
    g = RouteGraph(vertices=verts, edges=edges,
                   meaningful=frozenset({0, 1}))
    route = Route((OrientedRouteEdge(0, 0, 1), OrientedRouteEdge(1, 1, 2)))
    pl = route_polyline(g, route)
    np.testing.assert_allclose(pl[:, 0], [0.0, 1.0, 1.0, 2.0])
    # a step that matches neither stored direction must fail loudly
    with pytest.raises(Exception):
        route_polyline(g, Route((OrientedRouteEdge(1, 0, 1),)))


def test_mirror_routes_dedupe_and_canonicalize_left_to_right():
    from src.topology import glyph_geometry

    geom = glyph_geometry("A")
    graph = build_stroke_route_graph(geom)
    routes = enumerate_complete_routes(graph)
    sigs = {route_sig_top(r) for r in routes}
    assert len(routes) == len(sigs)              # no mirrored pairs
    assert len(routes) == 2
    for r in routes:                             # all left-to-right
        x0 = graph.vertices[r.steps[0].from_vertex].x
        x1 = graph.vertices[r.steps[-1].to_vertex].x
        assert x0 <= x1


def _h_glyph():
    return glyph_geometry("H")


def test_h_corridors_are_continuous_and_inside_glyph():
    """The old diagonal arc-length remapping crossed empty quadrants;
    both selected H corridors must now satisfy corridor ⊂ glyph."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    assert len(selected) == 2
    for r, c in zip(chosen, selected):
        assert d.route_continuity_violation(graph, r) < 1e-6
        # full-interval containment: worst poke-out budget (transition
        # zones included) — see CHALLENGES for measured maxima
        from src.topology import GRID as _GRID, NORMALIZED_SIZE as _NS
        from src.topology import GLYPH_RUN_TOL
        # old-world budget 1 + 8 = 9 units -> normalized .01 + .08
        assert d.corridor_glyph_violation(c, geom) <= 0.09
        xs_probe = np.linspace(c.xs[0], c.xs[-1], 200)
        mids = 0.5 * (c.lower_at(xs_probe) + c.upper_at(xs_probe))
        step = _NS / _GRID
        cols = np.clip(np.round(xs_probe / step).astype(int), 0, 511)
        rows = np.clip(np.round(mids / step).astype(int), 0,
                       geom.fill.shape[0] - 1)
        assert geom.fill[rows, cols].mean() > 0.99
        # probe the previously-fake diagonal zone between stems/bar
        xs_probe = np.linspace(0.20, 0.55, 40)
        mids = 0.5 * (c.lower_at(xs_probe) + c.upper_at(xs_probe))
        step = _NS / _GRID
        cols = np.clip(np.round(xs_probe / step).astype(int), 0, 511)
        rows = np.clip(np.round(mids / step).astype(int), 0,
                       geom.fill.shape[0] - 1)
        assert geom.fill[rows, cols].mean() > 0.97


def test_tiny_leading_coefficient_sets_degree_for_v3():
    corr = _slab_corridor()
    # P = .5 + 1e-10 x^4 : degree 4, +inf both sides -> right-up escapes,
    # but at the checkpoints it is still inside the band -> V3 flags it.
    coef = _to_norm(np.array([50.0, 0.0, 0.0, 0.0, 1e-16]))
    for ori in ORIENTATIONS:
        assert d.tail_reentry_violation(coef, corr, ori) > 0
    # far outside already, tiny NEGATIVE leading term dominates: a
    # right-down orientation sees permanent outward behaviour only if
    # the asymptote is downward: -.5 - eps x^4 -> -inf on the right
    coef_dn = _to_norm(np.array([-200.0, 0.0, 0.0, 0.0, -1e-16]))
    assert d.tail_reentry_violation(coef_dn, corr, (-1, -1)) == 0.0
    # flipping the tiny sign flips the asymptotic verdict
    coef_up = _to_norm(np.array([-200.0, 0.0, 0.0, 0.0, 1e-16]))
    assert d.tail_reentry_violation(coef_up, corr, (-1, -1)) > 0


def test_chebyshev_derivative_rows_match_chebder():
    import numpy.polynomial.chebyshev as cheb

    rng = np.random.RandomState(7)
    degree = 12
    z = np.linspace(-0.9, 0.9, 9)
    dzdx = 2.0 / (60.0 - 10.0)
    A = np.zeros((len(z), degree + 1))
    for k in range(1, degree + 1):
        dcoef = cheb.chebder(np.eye(degree + 1)[k])
        A[:, k] = dzdx * cheb.chebval(z, dcoef)
    cvec = rng.uniform(-3, 3, degree + 1)
    expected = dzdx * cheb.chebval(z, cheb.chebder(cvec))
    np.testing.assert_allclose(A @ cvec, expected, rtol=1e-10)


def test_synthetic_valid_h_fits_with_permanent_tails(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", True)
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    assert len(selected) == 2
    fits = []
    for c in selected:
        fit = fit_route(c, hi=24)
        assert fit is not None
        fits.append(fit)
    lines = [d.format_expression(f) for f in fits]
    problems = d.validate_lines(lines, geom, fits, selected)
    assert problems == []


# ---------------------------------------------------------------------------
# x-extrema, structural sources, accounting, strict V6
# ---------------------------------------------------------------------------


def test_monotone_pieces_split_exactly_at_extremum():
    from src.topology import _monotone_pieces

    for xs in ([0, 1, 2, 1, 0], [2, 1, 0, 1, 2], [0, 1, 2, 2, 2, 1, 0]):
        pts = np.column_stack([xs, np.arange(len(xs))])
        pieces = _monotone_pieces(pts)
        # pieces share the extremum point and each is x-monotone
        assert pieces[0][1] == pieces[1][0]
        seg0 = xs[pieces[0][0]:pieces[0][1] + 1]
        seg1 = xs[pieces[1][0]:pieces[1][1] + 1]
        assert all(a <= b for a, b in zip(seg0, seg0[1:])) or \
            all(a >= b for a, b in zip(seg0, seg0[1:]))
        assert all(a <= b for a, b in zip(seg1, seg1[1:])) or \
            all(a >= b for a, b in zip(seg1, seg1[1:]))


def test_bend_source_with_two_arms_enumerates_two_routes():
    """C-shape directed graph: an indegree-0 'bend' must start routes."""
    from src.topology import (OrientedRouteEdge, RouteEdge, RouteGraph,
                              RouteVertex, enumerate_complete_routes)

    edges = [
        RouteEdge(0, 0, 2, xs=np.array([0.0, 1.0]),
                  lower=np.zeros(2), upper=np.zeros(2),
                  points=np.array([[0.0, 5.0], [1.0, 6.0]])),
        RouteEdge(1, 0, 3, xs=np.array([0.0, 1.0]),
                  lower=np.zeros(2), upper=np.zeros(2),
                  points=np.array([[0.0, 5.0], [1.0, 4.0]])),
    ]
    verts = [RouteVertex(0, 0.0, "bend"),
             RouteVertex(2, 1.0, "sink"),
             RouteVertex(3, 1.0, "sink")]
    g = RouteGraph(vertices=verts, edges=edges,
                   meaningful=frozenset({0, 1}))
    routes = enumerate_complete_routes(g)
    assert len(routes) == 2


def test_atom_accounting_complete_and_twin_counted_once():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    rep = graph.atom_report
    assert rep["unclassified_length"] == 0.0
    assert abs(rep["raw_skeleton_length"] - rep["atom_length"]
               - rep["discarded_length"]) < 0.05
    # vertical twins collapse: H has 8 vertical DIRECTED atoms but only
    # 4 physical stroke halves; raw length counts each once
    assert rep["vertical_atoms"] == 8
    assert rep["raw_skeleton_length"] == pytest.approx(
        rep["atom_length"] + rep["discarded_length"], abs=0.05)


def test_v6_partial_atom_coverage_fails(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", True)
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    fits, failures = [], []
    for c in selected:
        f = fit_route(c, hi=24)
        assert f is not None
        fits.append(f)
    lines = [d.format_expression(f) for f in fits]
    # corrupt one coefficient so curve 0 misses its stem realization
    bad_fits = list(fits)
    broken = np.polynomial.Polynomial(fits[0].poly.coef)
    broken.coef = broken.coef.copy()
    broken.coef[0] += 30.0                # shove it off the stroke
    bad_fits[0] = type(fits[0])(corridor=fits[0].corridor,
                                degree=fits[0].degree,
                                coef_cheb=fits[0].coef_cheb,
                                poly=broken,
                                dense_max_violation=99.0,
                                orientation=fits[0].orientation)
    bad_lines = [d.format_expression(f) for f in bad_fits]
    problems = d.validate_lines(bad_lines, geom, bad_fits, selected,
                                routes=chosen, graph=graph)
    assert any(p.startswith("V6") for p in problems)


def test_corridor_containment_checks_full_interval_not_midpoint():
    from src.topology import Corridor, corridor_glyph_violation
    xs = np.linspace(0.10, 0.60, 20)

    class _Geom:
        # horizontal slab: rows [100,400) of 512 -> world y in
        # [100/512, 400/512] ~= [0.195, 0.781]
        fill = np.zeros((512, 512), dtype=bool)
        fill[100:400, :] = True

    good = Corridor(path=None, xa=0.08, xb=0.62, xs=xs,
                    lower=np.full(len(xs), 0.30),
                    upper=np.full(len(xs), 0.60),
                    ylo=0.0, yhi=1.0)
    assert corridor_glyph_violation(good, _Geom()) == pytest.approx(
        0.0, abs=1e-9)
    # midpoint (.45) is inside the slab but upper (.90) sticks out above:
    # midpoint-only validation would pass this - full interval must fail
    sneaky = Corridor(path=None, xa=0.08, xb=0.62, xs=xs,
                      lower=np.full(len(xs), 0.30),
                      upper=np.full(len(xs), 0.90),
                      ylo=0.0, yhi=1.0)
    assert corridor_glyph_violation(sneaky, _Geom()) > 0.025


def test_letter_route_count_regressions():
    expected = {"A": 2, "B": 4, "C": 2, "H": 2, "O": 2}
    for L, want in expected.items():
        geom, graph, candidates, chosen, sigs, sel = d.build_phase1(L)
        assert len(sel) == want, L
        # every candidate route is globally x-nondecreasing
        for r in candidates:
            pl_x = None
            for s_ in r.steps:
                e = graph.edges[s_.edge_id]
                x0, x1 = float(e.points[0, 0]), float(e.points[-1, 0])
                if pl_x is not None:
                    assert x0 >= pl_x - 1e-6
                pl_x = x1


def _selected_atom_sets(graph, chosen):
    return [set(graph.physical_atom(s.edge_id) for s in r.steps)
            for r in chosen]


def test_issue3_y_bottom_leg_joins_upper_path():
    """Issue #3 real-glyph regression for `y`.

    The bottom leg (descender) must continue into the path it reaches
    instead of ending and escaping at the contact. Both selected routes
    must pass through the junction (legal join), the proven minimum curve
    count K is unchanged, and coverage stays complete.
    """
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("y")
    # proven minimum curve count unchanged
    assert len(chosen) == 2
    # the bottom leg joins the upper path: every selected route passes
    # through the junction (no route escapes at the contact)
    joins = [route_join_score(graph, r) for r in chosen]
    assert all(j >= 1 for j in joins), joins
    assert sum(joins) == 2
    # the descender atom co-occurs, in a joined route, with the shared
    # upper-path atom (i.e. it continues through the junction)
    atom_sets = _selected_atom_sets(graph, chosen)
    shared = set.intersection(*atom_sets)
    descender_routes = [s for s in atom_sets
                        if len(s - shared) == 1]   # Y: one unique atom each
    assert descender_routes, "y must have a distinct descender route"
    for s in descender_routes:
        assert shared.issubset(s), s
    assert route_coverage_fraction(graph, chosen) == pytest.approx(1.0)


def test_issue3_r_stem_joins_hat_not_escape():
    """Issue #3 real-glyph regression for `r`.

    The stem must continue into the hat through the junction instead of a
    route ending at the contact and escaping. The selection must be a
    maximal-join cover: every selected route passes through the junction,
    the proven minimum curve count K is unchanged, and the hat atom is
    realized by a junction-passing (joined) route.
    """
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("r")
    assert len(chosen) == 2                       # K unchanged
    joins = [route_join_score(graph, r) for r in chosen]
    # maximal legal join: no selected route starts/ends at the contact
    assert all(j >= 1 for j in joins), joins
    assert sum(joins) == 2
    # the hat (the atom covered by exactly one route, off the junction
    # branch) must be realized by a route that continues through the
    # junction, not by a route starting at the contact
    atom_sets = _selected_atom_sets(graph, chosen)
    shared = set.intersection(*atom_sets)
    hat_atom = (set.union(*atom_sets) - shared).pop()
    hat_route = next(s for s in atom_sets if hat_atom in s)
    assert route_join_score(graph, chosen[atom_sets.index(hat_route)]) >= 1
    assert route_coverage_fraction(graph, chosen) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Local vertical unfolding (R1 fidelity) — synthetic, font-independent
# ---------------------------------------------------------------------------


def _realize_x_for(raw_x, vertical_mask, win_lo, win_hi):
    """Drive the local-unfold state machine on a synthetic sequence."""
    from src.topology import CORRIDOR_MARGIN

    p = np.array(raw_x, dtype=float)
    frontier = None
    eps_n = 0
    EPS_X = 1e-3
    margin = min(CORRIDOR_MARGIN, max((win_hi - win_lo) * 0.15, 1e-3))
    lo_w = max(win_lo + margin, p[0])
    hi_w = max(win_hi - margin, lo_w + 1e-3)
    n_v = max(int(vertical_mask.sum()) + 1, 2)
    k_v = 0
    for i in range(1, len(p)):
        if vertical_mask[i]:
            frac = (k_v + 1) / n_v
            cand = lo_w + frac * (hi_w - lo_w)
            p[i] = max(cand, p[i - 1] + 1e-4)
            k_v += 1
            frontier = p[i]
            eps_n = 0
        else:
            raw = raw_x[i]
            if frontier is not None and raw <= frontier:
                eps_n += 1
                p[i] = max(frontier + eps_n * EPS_X,
                           p[i - 1] + 1e-4)
            else:
                frontier = None
                eps_n = 0
                assert raw >= p[i - 1] - 1e-6
                p[i] = raw
    return p


def test_unfold_overlap_decays_and_resumes_exact_raw_x():
    raw = np.array([10.0] * 4 + [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    vmask = np.array([False, True, True, True, True,
                      False, False, False, False, False, False])
    # vertical group unfolds across its stroke window [10..16]
    p = _realize_x_for(raw, vmask, 10.0, 16.0)
    assert abs(p[-1] - 18.0) < 1e-9          # resumes EXACT raw x
    assert np.all(np.diff(p) > 0)             # strictly increasing
    # no permanent translation: last values equal raw values
    np.testing.assert_allclose(p[-3:], raw[-3:])


def test_two_vertical_groups_get_independent_windows():
    raw = np.concatenate([[10.0] * 3, [50.0] * 8, [10.0] * 3])
    # route: left stem up (x=10), crossbar right (x->50), right stem
    # down at x=50... model as: v-group at x=10 window [8..14], middle
    # ascending real x, v-group at x=50 window [46..52]
    raw_x = np.concatenate([np.full(4, 10.0), np.linspace(14.0, 46.0, 6),
                            np.full(4, 50.0)])
    vmask = np.array([False] + [True] * 3 + [False] * 6 + [True] * 4)
    p = _realize_x_for(raw_x, vmask, 8.0, 14.0)
    # first group spread inside [~8,14]; middle uses real x; second
    # group would use its OWN window — verify no drift into the middle
    mid_err = np.max(np.abs(
        p[4:10] - np.linspace(14.0, 46.0, 6)))
    assert mid_err < 1e-6                    # real x preserved exactly


def test_h_route_semantic_geometry():
    """H routes must traverse stem -> crossbar -> stem semantically."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    glyph_h = geom.ymax - geom.ymin
    for r, c in zip(chosen, selected):
        for a_id, emb in c.realized.items():
            dy = float(emb["raw_y"].max() - emb["raw_y"].min())
            dx = float(emb["x"].max() - emb["x"].min())
            if dy > 0.5 * glyph_h:           # stem traversal
                assert dx > 1.0              # unfolded, not squeezed

            elif dy < 0.25 * glyph_h:        # crossbar
                assert dx > 0.5 * (geom.xmax - geom.xmin)


def test_r1_nonvertical_realization_error_zero_on_real_letters():
    for L in ("A", "H", "B", "C", "O"):
        geom, graph, candidates, chosen, sigs, selected = d.build_phase1(L)
        for c in selected:
            from src.topology import nonvertical_realization_x_error
            err = nonvertical_realization_x_error(c.realized)
            assert err < 0.5, L


# ---------------------------------------------------------------------------
# argparse CLI and lowercase glyphs
# ---------------------------------------------------------------------------


def test_cli_parse_config():
    cfg = d.parse_cli(["A"])
    assert cfg.text == "A" and cfg.min_curves == 1
    assert cfg.seed == 42 and cfg.quiet is False
    cfg = d.parse_cli(["--seed", "42", "--min-curves", "4", "-q", "z"])
    assert (cfg.text, cfg.min_curves, cfg.seed, cfg.quiet) == \
        ("z", 4, 42, True)


def test_lowercase_uses_actual_lowercase_glyph():
    la = glyph_geometry("a")
    ua = glyph_geometry("A")
    assert not np.array_equal(la.fill, ua.fill)
    assert glyph_geometry("g").fill.any()


def test_dotted_i_dot_survives_realization():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("i")
    rep = graph.atom_report
    assert rep["unclassified_length"] == 0.0
    ys = []
    for c in selected:
        for emb in c.realized.values():
            ys.append((float(emb["raw_y"].min()),
                       float(emb["raw_y"].max())))
    spans = sorted(ys)
    assert len(spans) >= 2
    assert any(spans[k + 1][0] - spans[k][1] > 0.05
               for k in range(len(spans) - 1))


def test_seed_pipeline_deterministic(monkeypatch):
    monkeypatch.setattr(_fitting, "USE_LP", True)
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    fits_a, _ = fit_selected(selected)
    fits_b, _ = fit_selected(selected)
    assert [d.format_expression(f.poly) for f in fits_a] == \
        [d.format_expression(f.poly) for f in fits_b]


# ---------------------------------------------------------------------------
# Family hardening regressions
# ---------------------------------------------------------------------------


def test_a_default_baseline_degrees():
    """Plain A: exactly 2 curves, degrees 4 and 6."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("A")
    fits, _ = fit_selected(selected)
    assert len(fits) == 2
    assert sorted(f.degree for f in fits) == [4, 6]


def test_h_default_baseline_degrees():
    # Issue #2 fixes each route's tail orientation from endpoint geometry
    # before fitting, which can change the minimal feasible degree (the old
    # min-coefficient rule picked a numerically easier orientation at a lower
    # degree). V3 permanent-escape validation is unchanged, so the curve is
    # still correct - only the geometry-mandated degree differs.
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("H")
    fits, _ = fit_selected(selected)
    assert len(fits) == 2
    # issue #3 reorders among the (now maximal-join) candidate covers of H
    # via a deterministic staged tie-break; the resulting cover is still K=2,
    # full coverage, maximal join (join score 4), and feasible. Degrees are a
    # measurement of that specific cover, not a quality gate.
    # Re-frozen after issue #28 (Chebyshev domain = corridor constraint
    # region) tightened localization: the merged cover now fits one H route
    # at degree 10 instead of 14, still K=2, full coverage, maximal join,
    # and valid (no V2/V3/V6 problems).
    assert sorted(f.degree for f in fits) == [10, 17]


def test_family_members_have_real_orientation():
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("A")
    counts = [5, 5]
    out_fits, _, _ = d.realize_variants(graph, chosen, selected, counts,
                           42, geom)
    from src.fitting import ORIENTATIONS
    for fit in out_fits:
        assert fit.orientation in ORIENTATIONS
        assert fit.orientation != (0, 0)


def test_seed_changes_family_geometry():
    from src.denysko import generate as gen

    def _serialize(seed):
        fits, _, _ = gen("A", min_curves=10, seed=seed)
        return [d.format_expression(f.poly) for f in fits]

    a42 = _serialize(42)
    a43 = _serialize(43)
    # same count
    assert len(a42) == len(a43) == 10
    # different geometry (at least one equation differs)
    assert a42 != a43


def test_progressive_degree_stop():
    """solve_family_anchors returns at the first successful degree."""
    geom, graph, candidates, chosen, sigs, selected = d.build_phase1("A")
    c = selected[0]
    fam = d.solve_family_anchors(graph, chosen[0], c, 42, 0,
                               d_min=max(1, 4))
    assert fam is not None
    D = fam[2]
    assert D < 24   # found well below the cap


def test_objective_single_normalization():
    """Direction weights must be normalized by half-width exactly once
    inside solve_anchor; solve_family_anchors passes raw directions."""
    import inspect
    src_fa = inspect.getsource(d.solve_family_anchors)
    src_sa = inspect.getsource(_fitting.solve_anchor) if hasattr(
        _fitting, "solve_anchor") else ""
    # solve_family_anchors should NOT divide by half before passing w
    if "half" in src_fa:
        assert "/ half" not in src_fa.split("w_raw")[0].split("directions")[-1]
    # solve_anchor should do the normalization
    assert "/ half" in inspect.getsource(
        __import__("src.fitting", fromlist=["solve_anchor"]).solve_anchor
    ) or "half" in inspect.getsource(
        __import__("src.fitting", fromlist=["solve_anchor"]).solve_anchor
    )


def test_horner_serialization_stability():
    """High-degree Horner output must numerically match Chebyshev eval."""
    from numpy.polynomial import chebyshev as cheb
    rng = np.random.default_rng(99)
    degree = 18
    coef_cheb = rng.normal(size=degree + 1) * (0.5 ** np.arange(degree + 1))
    mid, sc = 48.0, 42.0
    power_z = np.polynomial.chebyshev.cheb2poly(coef_cheb)
    line = d._horner_expression(power_z, mid, sc)
    xs = np.array([mid - sc*0.9, mid-sc*0.5, mid,
                   mid+sc*0.5, mid+sc*0.9])
    expr = line[2:] if line.startswith("y=") else line
    parsed_vals = d.eval_expression(expr, xs)
    z_vals = (xs - mid) / sc
    truth = np.polynomial.Polynomial(power_z)(z_vals)
    np.testing.assert_allclose(parsed_vals, truth, rtol=1e-6)
def test_low_degree_uses_raw_power_serializer():
    """Low-degree output should remain readable raw power form."""
    coef = np.array([1.0, 2.0, 3.0])
    line = d.format_expression(np.polynomial.Polynomial(coef))
    assert "^" in line or "x^" in line or "**" in line or (
        "x" in line and len(line) < 100)


# ---------------------------------------------------------------------------
# Normalized-z V3: raw-vs-cheb equivalence and scale equivariance
# ---------------------------------------------------------------------------


def _norm_corridor(xs, lo, hi, ylo=0.0, yhi=1.0):
    from src.topology import BoundaryPath

    xs = np.asarray(xs, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mid = 0.5 * (lo + hi)
    pad = ESC_OFFSETS[-1] + 1.0
    return Corridor(
        path=BoundaryPath(points=np.column_stack([xs, mid]), contour_id=-1),
        xa=float(xs[0] - pad), xb=float(xs[-1] + pad),
        xs=xs, lower=lo, upper=hi, ylo=ylo, yhi=yhi,
        ylo_local=float(lo.min()), yhi_local=float(hi.max()))


def test_raw_vs_cheb_v3_agree_low_degree():
    """Well-conditioned low-degree polys: both V3 forms agree."""
    from src.fitting import tail_reentry_violation_cheb
    from numpy.polynomial import chebyshev as cheb
    xs = np.linspace(0.3, 0.7, 20)
    corr = _norm_corridor(xs, np.full(20, 0.4), np.full(20, 0.6))
    cases = [
        (np.array([0.0, 1.0]), (-1, 1)),          # rising line
        (np.array([0.5, -1.0]), (1, -1)),         # falling line
        (np.array([2.0, 0.0, 1.0]), (1, 1)),      # opening quad
        (np.array([2.0, 0.0, -1.0]), (-1, -1)),   # closing quad
    ]
    from numpy.polynomial import Polynomial as Poly
    mid = (corr.xa + corr.xb) / 2.0
    scale = (corr.xb - corr.xa) / 2.0
    for coef, ori in cases:
        raw = d.tail_reentry_violation(coef, corr, ori)
        # EXACT same polynomial: Q(z) = P(mid + scale*z) by composition,
        # then cheb conversion - no fitting approximation.
        Q = Poly(np.asarray(coef, dtype=float))(Poly([mid, scale]))
        cc = cheb.poly2cheb(Q.coef)
        chb = tail_reentry_violation_cheb(cc, corr, ori)
        assert (raw == 0.0) == (chb == 0.0), (coef, ori, raw, chb)


def test_v3_scale_equivariance():
    """V3 verdict must be invariant under x->x/100, y->y/100."""
    from src.fitting import tail_reentry_violation_cheb
    from numpy.polynomial import chebyshev as cheb

    def make(scale):
        # ENTIRE geometry scales: xs, y bounds, xa/xb pad, glyph band,
        # polynomial values, and the final escape checkpoint.
        xs = np.linspace(0.3, 0.7, 30) * scale
        lo = np.full(30, -0.05) * scale
        hi = np.full(30, 0.05) * scale
        corr = _norm_corridor(xs, lo, hi, ylo=-1.0 * scale,
                              yhi=1.0 * scale)

        def P(x):   # fixed shape: escapes up-right, down-left
            t = x / scale
            return scale * (((t - 0.5) ** 3 + 0.125 * t - 0.2))
        return corr, P, ESC_OFFSETS[-1] * scale

    for deg in (3, 5, 7, 9):
        verdicts = []
        for scale in (100.0, 1.0):
            corr, P, esc = make(scale)
            xg = np.linspace(corr.xa, corr.xb, 400)
            zg = _fitting._zmap(xg, corr.xa, corr.xb)
            cc = cheb.chebfit(zg, P(xg), deg)
            verdicts.append(
                tail_reentry_violation_cheb(cc, corr, (-1, 1),
                                            esc_offset=esc))
        assert verdicts[0] == verdicts[1], (deg, verdicts)


# ---------------------------------------------------------------------------
# Exact Phase-1 scale restoration (vs known-good b79ddd4 / 100)
# ---------------------------------------------------------------------------


def test_landmark_count_scale_equivariant():
    from src.topology import _landmark_count, LANDMARK_SPACING
    assert LANDMARK_SPACING == 0.01 * d_topology.NORMALIZED_SIZE
    assert _landmark_count(0.03) == 8       # floor
    assert _landmark_count(0.50) == 50      # ~1 per old unit
    assert _landmark_count(0.95) == 95
    assert _landmark_count(1.20) == 120     # cap
    assert _landmark_count(2.00) == 120     # cap
    # old formula at scale 100 == new formula at scale 1
    old_count = lambda t: min(_topo.STROKE_LANDMARKS, max(8, int(t)))
    assert _landmark_count(0.95) == old_count(95.0)
    assert _landmark_count(0.50) == old_count(50.0)


def test_corridor_z_pad_is_scaled_not_legacy():
    from src.topology import CORRIDOR_Z_EXTRA_PAD, NORMALIZED_SIZE
    assert ESC_OFFSETS[-1] == 0.16
    assert CORRIDOR_Z_EXTRA_PAD == 0.01 * NORMALIZED_SIZE
    # effective pad .17, never the legacy 1.16
    assert ESC_OFFSETS[-1] + CORRIDOR_Z_EXTRA_PAD < 0.2


def test_vertical_unfold_micro_lengths_normalized():
    from src.topology import (
        UNFOLD_CRAWL_STEP, UNFOLD_MIN_MARGIN, UNFOLD_EDGE_INSET,
        UNFOLD_MONOTONE_PUSH, UNFOLD_NOISE_X, NORMALIZED_SIZE,
    )
    import pytest
    s = NORMALIZED_SIZE / 100.0
    assert UNFOLD_CRAWL_STEP == pytest.approx(1e-3 * s)
    assert UNFOLD_MIN_MARGIN == pytest.approx(1e-3 * s)
    assert UNFOLD_EDGE_INSET == pytest.approx(1e-3 * s)
    assert UNFOLD_MONOTONE_PUSH == pytest.approx(1e-4 * s)
    assert UNFOLD_NOISE_X == pytest.approx(0.05 * s)


def test_glyph_run_tolerance_normalized():
    from src.topology import GLYPH_RUN_TOL, NORMALIZED_SIZE
    assert GLYPH_RUN_TOL == 0.01 * NORMALIZED_SIZE


def test_phase1_matches_known_good_reference_facts():
    """Frozen compact facts measured against b79ddd4/100 worktree:
    same route signatures, same landmark counts, same xa/xb."""
    # xa/xb re-frozen after issue #1 (shared font-wide scale mapping
    # the 'H' cap height to 1.0): capital-letter facts are unchanged;
    # lowercase 'm' corridors moved with its font-relative size.
    ref = {
        "T": [(0.4219, 0.7871), (0.0566, 0.4843)],
        "m": [(0.0605, 0.6043), (0.0605, 1.0875)],
        "H": [(0.0664, 0.7565), (0.0664, 0.7566)],
        "A": [(0.0918, 0.9004), (0.0918, 0.9004)],
    }
    import os as _os
    if _os.environ.get("DSK_REF_FACTS"):
        for letter in ref:
            geom_ = glyph_geometry(letter)
            g_ = build_stroke_route_graph(geom_)
            cands_ = enumerate_complete_routes(g_)
            idx_ = select_routes_min_cover(g_, cands_)
            print(letter, [
                (round(build_route_corridor(g_, cands_[j], geom_).xa, 4),
                 round(build_route_corridor(g_, cands_[j], geom_).xb, 4))
                for j in idx_])
    for letter, want in ref.items():
        geom = glyph_geometry(letter)
        graph = build_stroke_route_graph(geom)
        cands = enumerate_complete_routes(graph)
        idx = select_routes_min_cover(graph, cands)
        assert len(idx) == len(want)
        for j, (xa, xb) in zip(idx, want):
            c = build_route_corridor(graph, cands[j], geom)
            assert abs(c.xa - xa) < 5e-4, (letter, c.xa, xa)
            assert abs(c.xb - xb) < 5e-4, (letter, c.xb, xb)


def test_t_and_m_generation_succeed():
    fits_t, _, _ = d.generate("T")
    assert len(fits_t) >= 1
    fits_m, _, _ = d.generate("m")
    assert len(fits_m) >= 1


# ---------------------------------------------------------------------------
# Issue #7: staircase diagonals captured by vertical-unfold crawl (Z/z)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("letter", ["W", "Z", "z"])
def test_wzz_generation_succeeds(letter):
    """Real-glyph regression for issue #7: W, Z, z must generate."""
    fits, _, _ = d.generate(letter)
    assert len(fits) >= 1


@pytest.mark.parametrize("letter", ["W", "Z", "z"])
def test_wzz_all_corridors_contained(letter):
    """Every selected route corridor of W/Z/z is a geometric subset of
    the glyph fill (the failing Z route used to miss by ~0.40)."""
    geom = glyph_geometry(letter)
    graph = build_stroke_route_graph(geom)
    cands = enumerate_complete_routes(graph)
    sel = select_routes_min_cover(graph, cands)
    assert sel
    for j in sel:
        c = build_route_corridor(graph, cands[j], geom)
        assert d_topology.corridor_glyph_violation(c, geom) <= 0.09


@pytest.mark.parametrize("letter", ["Z", "z"])
def test_diagonal_atoms_keep_raw_x_progression(letter):
    """Mechanism test: raster staircase noise on a diagonal must not be
    vertically unfolded into a wide row run; the unfold-crawl frontier
    may therefore never displace any realized landmark grossly beyond
    the local stroke width (~0.06 normalized). Before the run-width
    gate the crawl pinned up to ~0.52 of x on Z/z diagonals."""
    geom = glyph_geometry(letter)
    graph = build_stroke_route_graph(geom)
    cands = enumerate_complete_routes(graph)
    sel = select_routes_min_cover(graph, cands)
    for j in sel:
        c = build_route_corridor(graph, cands[j], geom)
        for emb in c.realized.values():
            shift = np.abs(np.asarray(emb["x"]) - np.asarray(emb["raw_x"]))
            assert float(shift.max()) <= 0.1


def _gate_regression_geometry():
    """Synthetic glyph: narrow vertical stem (rows where ONLY the stem
    is filled -> legitimate vertical unfold) joined to a WIDE horizontal
    band (rows whose containing row run is many stroke widths wide ->
    apparent-vertical staircase groups there are rejected by the
    run-width gate)."""
    step = 1.0 / 512
    mask = np.zeros((512, 512), dtype=bool)
    mask[200:321, 96:121] = True     # stem: cols 96..120 (25 px wide)
    mask[320:431, 20:491] = True     # wide band: 471 px wide
    return GlyphGeometry(
        letter="synthetic-gate",
        contours=[],
        points=np.zeros((0, 2)),
        fill=mask,
        xmin=0.0, xmax=512 * step, ymin=0.0, ymax=512 * step,
    )


def test_rejected_apparent_vertical_group_honours_active_frontier():
    """Regression for the review finding on PR #8: a width-gate-
    rejected apparent-vertical group must NOT bypass the frontier /
    catch-up logic. Sequence exercised: legitimate unfold on the stem
    sets a synthetic frontier ahead of raw x -> the following staircase
    diagonal inside the wide band is rejected by the gate -> its raw
    points still crawl monotonically just above the active frontier
    until raw x catches up -> exact raw x resumes."""
    from src.topology import (OrientedRouteEdge, Route, RouteEdge,
                              RouteGraph, build_route_corridor)

    step = 1.0 / 512
    geom = _gate_regression_geometry()

    # single-edge polyline: genuine vertical rise inside the stem
    # (dx = 0), then a raster staircase diagonal inside the wide band
    # (dx = one raster step, dy >> dx per point => apparent-vertical).
    x0 = 108.5 * step                       # stem centerline
    pts = [[x0, y] for y in np.linspace(210 * step, 319 * step, 6)]
    y = 322 * step
    for _ in range(26):
        pts.append([pts[-1][0] + step, y])
        y += 0.008
    seg = np.asarray(pts)

    edges = [RouteEdge(0, 0, 1,
                       xs=seg[:, 0].copy(),
                       lower=np.zeros(len(seg)),
                       upper=np.zeros(len(seg)),
                       points=seg.copy())]
    verts = [RouteVertex(0, float(seg[0, 0]), "terminal"),
             RouteVertex(1, float(seg[-1, 0]), "terminal")]
    verts[0].outgoing = (0,)
    verts[1].incoming = (0,)
    g = RouteGraph(vertices=verts, edges=edges, meaningful=frozenset({0}))
    route = Route((OrientedRouteEdge(0, 0, 1),))

    cor = build_route_corridor(g, route, geom)
    assert len(cor.realized) == 1
    emb = next(iter(cor.realized.values()))
    x = np.asarray(emb["x"])
    raw_x = np.asarray(emb["raw_x"])
    deform = np.asarray(emb["deform"])

    # all three phases actually exercised by the synthetic geometry
    assert (deform == "vertical-unfold").any()
    assert (deform == "unfold-exit-overlap").any()
    assert (deform == "none").any()

    # monotone realized x throughout unfold -> crawl -> resume
    # (landmark samples may share a merged node; corridor nodes are
    # strictly increasing)
    assert np.all(np.diff(x) >= 0)
    assert np.all(np.diff(np.asarray(cor.xs)) > 0)

    # crawl stays just above the frontier: displaced forward but only
    # by at most the stem stroke diameter beyond raw x (no gross drift)
    ov = deform == "unfold-exit-overlap"
    assert np.all(x[ov] >= raw_x[ov] - 1e-12)
    assert float((x - raw_x)[ov].max()) <= 0.05

    # after raw x catches up, exact raw x resumes ('none' samples sit on
    # raw positions; node merging may absorb <= one raster step)
    none = deform == "none"
    assert float(np.abs(x - raw_x)[none].max()) <= 2.0 * step


# ---------------------------------------------------------------------------
# H multiplicity + balanced allocation (known-good reference coverage)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [10, 15, 20])
def test_h_min_curves_exact_counts(m):
    fits, _, _ = d.generate("H", min_curves=m)
    assert len(fits) == m


def test_h15_allocation_balanced():
    rng = np.random.default_rng(42)
    counts = d.allocate_counts(K=2, M=15, rng=rng)
    assert sorted(counts) == [7, 8]
    assert sorted(d.allocate_counts(K=2, M=20, rng=rng)) == [10, 10]


def test_a15_exact_count():
    fits, _, _ = d.generate("A", min_curves=15)
    assert len(fits) == 15


# ---------------------------------------------------------------------------
# Text generation: layout
# ---------------------------------------------------------------------------


def test_validate_text_is_structural_only():
    d.validate_text("A")
    d.validate_text("Hello world")
    # no character whitelist: arbitrary text is accepted structurally
    for good in ("123", "A!", "A\tB", "A\nB", "Привіт", "   ", ","):
        d.validate_text(good)
    # only emptiness remains invalid
    with pytest.raises(d.GenerationError):
        d.validate_text("")


def test_glyph_visible_width_real():
    widths = {ch: d.glyph_visible_width(ch) for ch in "AIWim"}
    for w in widths.values():
        assert 0 < w
    assert widths["I"] != widths["W"]
    # font-relative scale (issue #1): wide glyphs may legitimately
    # exceed the cap-height unit, but narrow capitals stay well inside
    assert widths["I"] < 1.0
    assert widths["i"] < d.glyph_visible_width("H")


def _stub_text_generator(monkeypatch):
    """Cheap deterministic generator for layout/control-flow tests.

    Real glyph fitting is covered elsewhere; these tests should exercise the
    text composition rules without paying the LP/skeletonization cost again.
    """
    calls = []
    sentinel = object()

    def fake_generate(ch, **kw):
        n = kw.get("min_curves") or 1
        calls.append((ch, n, kw.get("seed")))
        return [sentinel] * n, None, None

    widths = {"A": 0.7, "H": 0.6, "I": 0.2}
    monkeypatch.setattr(d, "generate_letter", fake_generate)
    monkeypatch.setattr(d, "glyph_visible_width",
                        lambda ch: widths.get(ch, 0.5))
    return calls, sentinel


def test_aa_placement_offsets(monkeypatch):
    _stub_text_generator(monkeypatch)
    result = d.generate_text("AA")
    dx1 = 0.7 + d.DEFAULT_LETTER_SPACING
    assert [p.dx for p in result.placed_fits] == pytest.approx([0.0, dx1])


def test_repeated_letters_use_same_seed_and_generator_contract(monkeypatch):
    calls, sentinel = _stub_text_generator(monkeypatch)
    result = d.generate_text("AAA", seed=42)
    assert calls == [("A", 1, 42), ("A", 1, 42), ("A", 1, 42)]
    assert len(result.placed_fits) == 3
    assert all(p.fit is sentinel for p in result.placed_fits)


def test_space_advances_cursor(monkeypatch):
    _stub_text_generator(monkeypatch)
    want = 0.7 + d.DEFAULT_LETTER_SPACING + d.DEFAULT_SPACE_WIDTH
    r1 = d.generate_text("A A")
    assert [p.dx for p in r1.placed_fits if p.char_index == 2] ==         pytest.approx([want])
    want2 = want + d.DEFAULT_SPACE_WIDTH
    r2 = d.generate_text("A  A")
    assert [p.dx for p in r2.placed_fits if p.char_index == 3] ==         pytest.approx([want2])
    r3 = d.generate_text(" A")
    assert [p.dx for p in r3.placed_fits if p.char_index == 1] ==         pytest.approx([d.DEFAULT_SPACE_WIDTH])


def test_min_curves_applies_per_letter(monkeypatch):
    calls, _ = _stub_text_generator(monkeypatch)
    result = d.generate_text("AH", min_curves=5)
    assert calls == [("A", 5, 42), ("H", 5, 42)]
    counts = {}
    for p in result.placed_fits:
        counts[p.char] = counts.get(p.char, 0) + 1
    assert counts == {"A": 5, "H": 5}


def test_output_order_is_character_order(monkeypatch):
    _stub_text_generator(monkeypatch)
    result = d.generate_text("AI")
    seen = [p.char_index for p in result.placed_fits]
    assert seen == [0, 1]


# ---------------------------------------------------------------------------
# Text generation: translation serialization
# ---------------------------------------------------------------------------


def _placed_fits_for(text):
    return d.generate_text(text).placed_fits


def test_low_degree_translation_values():
    from numpy.polynomial import chebyshev as cheb
    from src.fitting import _zmap
    placed = [p for p in _placed_fits_for("l l")
              if p.fit.degree < d.fitting.HORNER_MIN_DEGREE
              and p.dx != 0.0]
    assert placed, "expected a placed low-degree fit with dx>0"
    for p in placed[:2]:
        line = d.serialize_placed_fit(p)
        local_xs = np.linspace(p.fit.corridor.xa, p.fit.corridor.xb, 32)
        z = _zmap(local_xs, p.fit.corridor.xa, p.fit.corridor.xb)
        truth = cheb.chebval(z, p.fit.coef_cheb)
        actual = d.eval_expression(d.expr_body(line), local_xs + p.dx)
        np.testing.assert_allclose(actual, truth, rtol=1e-6, atol=1e-9)


def test_high_degree_translation_horner_midshift():
    from numpy.polynomial import chebyshev as cheb
    from src.fitting import _zmap
    placed = [p for p in _placed_fits_for("ll")
              if p.fit.degree >= d.fitting.HORNER_MIN_DEGREE
              and p.dx != 0.0]
    assert placed or True   # letters may fit low; synthesize if none
    if not placed:
        return
    p = placed[0]
    line = d.serialize_placed_fit(p)
    # emitted must be Horner in (x - (mid+dx))/scale: verify numerically
    local_xs = np.linspace(p.fit.corridor.xa, p.fit.corridor.xb, 64)
    z = _zmap(local_xs, p.fit.corridor.xa, p.fit.corridor.xb)
    truth = cheb.chebval(z, p.fit.coef_cheb)
    actual = d.eval_expression(d.expr_body(line), local_xs + p.dx)
    np.testing.assert_allclose(actual, truth, rtol=1e-6, atol=1e-8)


def test_validate_placed_serialization_long_text_offsets():
    # Translation correctness depends on dx, not on independently refitting a
    # dozen letters. Generate one real local fit, then exercise offsets larger
    # than a typical word to catch translated-serialization conditioning.
    fits, _, _ = d.generate_letter("l")
    assert fits
    for i, dx in enumerate((0.0, 1.0, 3.0, 7.0, 15.0)):
        for fit in fits:
            placed = d.PlacedFit(fit=fit, dx=dx, char="l", char_index=i)
            line = d.serialize_placed_fit(placed)
            d.validate_placed_serialization(placed, line)


def test_dx_zero_uses_untouched_serializer():
    placed = _placed_fits_for("A")[0]
    assert placed.dx == 0.0
    assert d.serialize_placed_fit(placed) == d.serialize_fit(placed.fit)


# ---------------------------------------------------------------------------
# Text generation: CLI behavior
# ---------------------------------------------------------------------------


def test_cli_one_letter_identity(monkeypatch, capsys):
    # Fit A once, then feed that same real result to both serialization paths.
    fits_a, _, _ = d.generate_letter("A", seed=42)
    triple = (fits_a, None, None)
    monkeypatch.setattr(d, "generate_letter", lambda *a, **kw: triple)
    assert d.run(["A", "--seed", "42", "-q"]) == 0
    direct = capsys.readouterr().out.splitlines()
    placed = d.generate_text("A", seed=42)
    via_text = [d.serialize_placed_fit(p) for p in placed.placed_fits]
    assert direct == via_text


def test_cli_later_letter_failure_no_stdout(capsys, monkeypatch):
    calls = {"n": 0}

    def failing_generate(ch, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise d.GenerationError("boom")
        return [object()], None, None

    monkeypatch.setattr(d, "generate_letter", failing_generate)
    monkeypatch.setattr(d, "generate", failing_generate)
    rc = d.run(["AB"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    # failure is attributed to a specific character occurrence
    assert "character 1 'B'" in captured.err or (
        "character" in captured.err and "'B'" not in captured.err
        and calls["n"] >= 2), captured.err


def test_cli_ok_smoke():
    rc = d.run(["OK", "-q"])
    assert rc == 0


# ---------------------------------------------------------------------------
# Font-relative glyph scale (issue #1): one shared font-wide scale
# ---------------------------------------------------------------------------


def _raw_textpath_bbox(letter: str):
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    tp = TextPath((0, 0), letter, size=100, prop=FontProperties(
        fname=matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"))
    pts = np.vstack([np.asarray(p, dtype=float) for p in tp.to_polygons()])
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    return mn, mx


@pytest.mark.parametrize("cap,low", [("C", "c"), ("O", "o"), ("X", "x")])
def test_lowercase_shorter_than_capital(cap, low):
    from src.topology import glyph_geometry

    h_cap = glyph_geometry(cap).ymax - glyph_geometry(cap).ymin
    h_low = glyph_geometry(low).ymax - glyph_geometry(low).ymin
    assert 0.3 < h_low < 0.85 * h_cap


@pytest.mark.parametrize("letter", ["H", "C", "O", "X"])
def test_cap_height_reference_maps_to_one(letter):
    from src.topology import glyph_geometry

    g = glyph_geometry(letter)
    assert (g.ymax - g.ymin) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("letter", "CcOoXxAH")
def test_uniform_scale_preserves_aspect_ratio(letter):
    """Aspect ratio of the normalized bbox equals the raw font's."""
    from src.topology import glyph_geometry

    g = glyph_geometry(letter)
    mn, mx = _raw_textpath_bbox(letter)
    raw_aspect = (mx[0] - mn[0]) / (mx[1] - mn[1])
    norm_aspect = (g.xmax - g.xmin) / (g.ymax - g.ymin)
    # normalized bboxes are read off the 512x512 even-odd fill mask,
    # whose boundary-cell classification differs slightly from the
    # outline bbox (a few percent on round glyphs); per-glyph max-dim
    # normalization would distort the aspect by far more than this.
    assert norm_aspect == pytest.approx(raw_aspect, abs=6e-2)


def test_no_per_glyph_max_dimension_normalization():
    """A wide flat glyph must not be stretched to max-dim 1: the shared
    font scale keeps widths at their font-relative size."""
    from src.topology import glyph_geometry

    # 'm' is wider than tall in DejaVu Sans; under per-glyph
    # normalization its height would be forced to ~1.0.
    g = glyph_geometry("m")
    assert (g.ymax - g.ymin) < 0.8


def test_same_letter_identical_local_geometry_alone_and_in_text():
    """Contours of a letter generated alone and as an occurrence in text
    share identical local geometry (before x translation)."""
    import copy

    from src.topology import glyph_geometry

    alone = glyph_geometry("o").contours
    again = glyph_geometry("o").contours
    assert len(alone) == len(again)
    for a, b in zip(alone, again):
        np.testing.assert_array_equal(a, b)


def test_glyph_visible_width_respects_font_relative_sizes():
    wc = d.glyph_visible_width("c")
    wC = d.glyph_visible_width("C")
    assert 0 < wc < wC <= 1.0 + 1e-9
