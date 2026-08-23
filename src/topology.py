"""Phase 1: filled-glyph topology and complete route corridors.

Routes are NOT contour fragments. Phase 1 rasterizes the canonical
even-odd fill mask, sweeps it along x, and builds a layered routing
graph of the connected filled y-intervals (a Reeb-graph-style
decomposition of the filled glyph). After compressing trivial
continuation, graph vertices mark appearance / disappearance / split /
merge events and edges carry the vertical allowed interval across each
stretch. Complete routes are source-to-sink paths through that graph;
their corridors are the union of their slice intervals (with an
interior safety margin), so a corridor boundary may be contributed by
several different font contours.

Font contours are kept for diagnostics only - they define the walls of
the geometry, not the routes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

GRID = 512
SIZE = 100.0

TAU = 2.0
MIN_COVERAGE = 0.95              # diagnostic boundary proximity gate
DEFAULT_MAX_CURVES = 12

MIN_SLICE_ROWS = 2               # drop sub-2-row raster slivers
PINCH_COLS = 2                   # bridge disappearances up to this many cols
MAX_ROUTES = 4096                # enumeration guard
SLIVER_SPAN = 1.0                # route edges shorter than this are slivers

CORRIDOR_MARGIN = 0.4            # interior safety margin (actually applied)
MIN_CORRIDOR_WIDTH = 0.05        # never produce an inverted/empty interval
CORRIDOR_EPS = 0.35              # solver-numerics tolerance (see CHALLENGES)
SELECT_COVERAGE_TARGET = 0.97    # route-edge coverage buffer
ESCAPE_RATE = 2.5                # tail ramp: clearance growth per unit x
ESC_OFFSETS = (3.0, 6.0, 10.0, 16.0)   # ramp checkpoints beyond ends
ESC_SLOPE_MIN = 0.05             # min outward |dP/dx| along the ramp


def _font_path() -> str:
    return os.path.join(
        matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"
    )


def _normalized_polygons(letter: str) -> list[np.ndarray]:
    """Flattened glyph outlines normalized like the canonical raster:
    bundled DejaVuSans at size 100, aspect preserved, filled-bbox
    lower-left mapped to (0, 0), max dimension 100, y-up."""
    tp = TextPath((0, 0), letter, size=100,
                  prop=FontProperties(fname=_font_path()))
    polys = [np.asarray(p, dtype=float).copy() for p in tp.to_polygons()]
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return []
    pts = np.vstack(polys)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    scale = SIZE / max(mx[0] - mn[0], mx[1] - mn[1], 1e-12)
    out = []
    for poly in polys:
        t = np.empty_like(poly)
        t[:, 0] = (poly[:, 0] - mn[0]) * scale
        t[:, 1] = (poly[:, 1] - mn[1]) * scale
        out.append(t)
    return out


def _canonical_fill(polys_norm: list[np.ndarray]):
    """Even-odd rasterization of normalized outlines on the canonical
    GRID x GRID sample grid. Rows increase with y (row r centers at
    y = (r + 0.5) * step)."""
    step = SIZE / GRID
    axis = (np.arange(GRID) + 0.5) * step
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(grid), dtype=bool)
    for ring in polys_norm:
        inside ^= Path(ring).contains_points(grid)
    return inside.reshape(GRID, GRID), step


def _mask_boundary_cloud(fill: np.ndarray, step: float) -> np.ndarray:
    fp = np.pad(fill, 1, constant_values=False)
    interior_filled = (
        fp[1:-1, 1:-1]
        & fp[:-2, 1:-1]
        & fp[2:, 1:-1]
        & fp[1:-1, :-2]
        & fp[1:-1, 2:]
    )
    boundary = fill & ~interior_filled
    iy, ix = np.nonzero(boundary)
    return np.column_stack([(ix + 0.5) * step, (iy + 0.5) * step])


@dataclass
class GlyphGeometry:
    letter: str
    contours: list[np.ndarray]      # diagnostics/reference only
    points: np.ndarray              # boundary cloud (diagnostics)
    fill: np.ndarray                # canonical even-odd fill mask [GRID,GRID]
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def glyph_geometry(letter: str) -> GlyphGeometry:
    contours = _normalized_polygons(letter)
    fill, _step = _canonical_fill(contours)
    points = _mask_boundary_cloud(fill, _step)
    ys, xs_ = np.nonzero(fill)
    step = SIZE / GRID
    return GlyphGeometry(
        letter=letter,
        contours=contours,
        points=points,
        fill=fill,
        xmin=float((xs_.min()) * step),
        xmax=float((xs_.max() + 1) * step),
        ymin=float((ys.min()) * step),
        ymax=float((ys.max() + 1) * step),
    )


# ---------------------------------------------------------------------------
# Vertical-slice routing graph over the filled mask
# ---------------------------------------------------------------------------


@dataclass
class BoundaryPath:
    """Lightweight polyline carrier (route centerline or debug arc)."""

    points: np.ndarray
    contour_id: int = -1
    source_edge_ids: tuple = ()
    arc_points: np.ndarray | None = None
    covered: np.ndarray | None = None


@dataclass
class SliceInterval:
    column: int
    y_lo: float                     # normalized (y-up)
    y_hi: float


@dataclass
class RouteVertex:
    id: int
    x: float
    kind: str                       # 'source' | 'sink' | 'split' | 'merge'
    incoming: tuple = ()
    outgoing: tuple = ()


@dataclass
class RouteEdge:
    id: int
    v_from: int
    v_to: int
    xs: np.ndarray                  # ascending column centers
    lower: np.ndarray               # slice interval bottoms
    upper: np.ndarray               # slice interval tops
    points: np.ndarray | None = None   # (n,2) polyline, stroke-graph edges

    @property
    def span(self) -> float:
        if self.points is not None and len(self.points) > 1:
            seg = np.diff(self.points, axis=0)
            return float(np.hypot(seg[:, 0], seg[:, 1]).sum())
        return float(self.xs[-1] - self.xs[0])

    @property
    def mean_height(self) -> float:
        return float(np.mean(self.upper - self.lower))


@dataclass
class RouteGraph:
    vertices: list[RouteVertex]
    edges: list[RouteEdge]
    meaningful: frozenset           # edge ids that selection must cover

    def outgoing_edges(self, vertex_id: int):
        return [e for e in self.edges if e.v_from == vertex_id]

    def sources(self):
        return [v for v in self.vertices if v.kind == "source"]

    def sinks(self):
        return [v for v in self.vertices if v.kind == "sink"]


@dataclass
class Route:
    edge_ids: tuple                 # ordered source -> sink
    signature: str
    corridor: "Corridor"


def _column_runs(colmask: np.ndarray, min_rows: int):
    """Maximal contiguous True runs as (row_lo, row_hi) inclusive."""
    runs = []
    r = 0
    H = len(colmask)
    while r < H:
        if colmask[r]:
            s = r
            while r < H and colmask[r]:
                r += 1
            if r - s >= min_rows:
                runs.append((s, r - 1))
        else:
            r += 1
    return runs


def build_route_graph(geom: GlyphGeometry) -> RouteGraph:
    """Layered sweep of the canonical fill mask producing a compressed
    routing graph with explicit source/sink/split/merge vertices."""
    fill = geom.fill
    W = fill.shape[1]
    step = SIZE / GRID
    runs_per_col = [
        _column_runs(fill[:, i], MIN_SLICE_ROWS) for i in range(W)
    ]

    def interval(i, run):
        r0, r1 = run
        return SliceInterval(column=i, y_lo=r0 * step, y_hi=(r1 + 1) * step)

    vertices: list[RouteVertex] = []
    edges: list[RouteEdge] = []

    def new_vertex(x, kind, incoming=(), outgoing=()):
        v = RouteVertex(len(vertices), x, kind,
                        tuple(incoming), tuple(outgoing))
        vertices.append(v)
        return v.id

    def open_edge(v_from, i, iv, last_rows):
        e = RouteEdge(
            len(edges), v_from, -1,
            np.asarray([(i + 0.5) * step]),
            np.asarray([iv.y_lo]),
            np.asarray([iv.y_hi]),
        )
        edges.append(e)
        return {"edge": e.id, "last_rows": last_rows,
                "last_col": i, "last_iv": (iv.y_lo, iv.y_hi)}

    def extend(trk, i, iv, last_rows):
        trk["last_rows"] = last_rows
        trk["last_col"] = i
        trk["last_iv"] = (iv.y_lo, iv.y_hi)
        e = edges[trk["edge"]]
        e.xs = np.append(e.xs, (i + 0.5) * step)
        e.lower = np.append(e.lower, iv.y_lo)
        e.upper = np.append(e.upper, iv.y_hi)

    active: dict[int, dict] = {}
    lost: dict[int, dict] = {}
    next_branch = 0

    for i in range(W):
        runs = runs_per_col[i]

        # pinch reconnection; stale branches terminate for good
        still_lost = {}
        for bid, trk in lost.items():
            if i - trk["last_col"] > PINCH_COLS:
                e = edges[trk["edge"]]
                if e.v_to < 0:
                    e.v_to = new_vertex((trk["last_col"] + 0.5) * step,
                                        "sink")
                continue
            a0, a1 = trk["last_rows"]
            rematch = None
            for ri, run in enumerate(runs):
                if a0 <= run[1] + 1 and run[0] <= a1 + 1:
                    rematch = ri
                    break
            gap = i - trk["last_col"]
            if rematch is not None and gap <= PINCH_COLS:
                r0, r1 = runs.pop(rematch)
                for gc in range(trk["last_col"] + 1, i):
                    e = edges[trk["edge"]]
                    e.xs = np.append(e.xs, (gc + 0.5) * step)
                    e.lower = np.append(e.lower, trk["last_iv"][0])
                    e.upper = np.append(e.upper, trk["last_iv"][1])
                extend(trk, i, interval(i, (r0, r1)), (r0, r1))
                active[bid] = trk
            else:
                still_lost[bid] = trk
        lost = still_lost

        # classify claims: branch -> [run idx], run -> [branch id]
        claims: dict[int, list[int]] = {}
        bid_claims: dict[int, list[int]] = {}
        act = sorted(active.items(), key=lambda kv: kv[1]["last_rows"][0])
        for bid, trk in act:
            a0, a1 = trk["last_rows"]
            for ri, run in enumerate(runs):
                if a0 <= run[1] + 1 and run[0] <= a1 + 1:
                    claims.setdefault(ri, []).append(bid)
                    bid_claims.setdefault(bid, []).append(ri)

        new_active: dict[int, dict] = {}
        done: set[int] = set()

        # merges first
        for ri in sorted(claims):
            bids = claims[ri]
            if len(bids) <= 1:
                continue
            iv = interval(i, runs[ri])
            pes = tuple(active[b]["edge"] for b in bids)
            vid = new_vertex((i + 0.5) * step, "merge", incoming=pes)
            for eid in pes:
                edges[eid].v_to = vid
            nbid = next_branch; next_branch += 1
            new_active[nbid] = open_edge(vid, i, iv, runs[ri])
            done.update(bids)

        # splits next
        for bid in sorted(bid_claims):
            if bid in done:
                continue
            ris = bid_claims[bid]
            if len(ris) <= 1:
                continue
            pe = active[bid]["edge"]
            vid = new_vertex((i + 0.5) * step, "split",
                             incoming=(pe,))
            edges[pe].v_to = vid
            done.add(bid)
            for ri in sorted(ris):
                nbid = next_branch; next_branch += 1
                new_active[nbid] = open_edge(
                    vid, i, interval(i, runs[ri]), runs[ri])

        # continuations
        for bid in sorted(bid_claims):
            if bid in done:
                continue
            ri = bid_claims[bid][0]
            trk = active[bid]
            extend(trk, i, interval(i, runs[ri]), runs[ri])
            new_active[bid] = trk

        # sources from unclaimed runs
        for ri, run in enumerate(runs):
            if ri in claims:
                continue
            vid = new_vertex((i + 0.5) * step, "source")
            nbid = next_branch; next_branch += 1
            new_active[nbid] = open_edge(vid, i, interval(i, run), run)

        # disappeared -> pinch hold or sink later
        for bid, trk in active.items():
            if bid in done or bid in new_active:
                continue
            if bid not in bid_claims:
                lost[bid] = trk

        active = new_active

    right_x = (W - 0.5) * step
    for trk in {**active, **lost}.values():
        e = edges[trk["edge"]]
        if e.v_to < 0:
            e.v_to = new_vertex(right_x, "sink")

    for v in vertices:
        v.outgoing = tuple(e.id for e in edges if e.v_from == v.id)
        v.incoming = tuple(e.id for e in edges if e.v_to == v.id)

    meaningful = frozenset(
        e.id for e in edges
        if e.span >= SLIVER_SPAN and e.mean_height >= 1.0
    )
    return RouteGraph(vertices=vertices, edges=edges, meaningful=meaningful)

@dataclass
class Route:
    edge_ids: tuple                 # ordered source -> sink
    signature: str
    corridor: "Corridor"


def enumerate_complete_routes(graph: RouteGraph,
                              cap: int = MAX_ROUTES) -> list[tuple]:
    """Enumerate complete routes as ordered edge-id tuples.

    Routes are undirected edge-simple paths between terminal vertices
    ('source', 'sink' or 'terminal' kinds): a stroke chain may be
    traversed in either x-direction since a corridor constrains
    y-over-x regardless of walk order. Deterministic DFS ordered by
    edge id. The MAX_ROUTES cap guards against blowup: exceeding it
    RAISES rather than silently truncating, so an exact-minimum-cover
    claim is never based on an incomplete candidate set.
    """
    incident: dict[int, list[tuple[int, int]]] = {}
    for e in graph.edges:
        incident.setdefault(e.v_from, []).append((e.id, e.v_to))
        incident.setdefault(e.v_to, []).append((e.id, e.v_from))
    for v in incident:
        incident[v].sort()

    terminals = {v.id for v in graph.vertices
                 if v.kind in ("source", "sink", "terminal")}

    routes: list[tuple] = []
    budget = [cap]

    def dfs(v, seen, path):
        for eid, w in incident.get(v, []):
            if eid in seen:
                continue
            npath = path + (eid,)
            if w in terminals:
                # a terminal always terminates a route: passing THROUGH
                # one would let a single polynomial claim a whole cycle
                routes.append(npath)
                budget[0] -= 1
                if budget[0] < 0:
                    raise RuntimeError(
                        f"route enumeration exceeded MAX_ROUTES={cap}")
                continue
            dfs(w, seen | {eid}, npath)

    for t in sorted(terminals):
        dfs(t, frozenset(), ())
    return sorted(set(routes))   # dedupe mirrored walks, deterministic


def _route_signature(edge_ids):
    return "/".join(f"e{e}" for e in edge_ids)


def select_routes_min_cover(graph: RouteGraph, candidates: list[tuple]):
    """Exact minimum cover of all meaningful graph edges by complete
    routes (deterministic HiGHS MILP; falls back to exhaustive subset
    search for tiny candidate counts).

    Tie-break: fewer routes, then larger geometric coverage, then lower
    total complexity (sum of degrees), then stable signature order.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    meaningful = sorted(graph.meaningful)
    if not meaningful or not candidates:
        return []

    cand_sets = []
    for sig_ids in candidates:
        cand_sets.append(frozenset(sig_ids) & set(meaningful))

    n_r = len(candidates)
    n_e = len(meaningful)

    def total_coverage(sel):
        u = set()
        for r in sel:
            u |= cand_sets[r]
        return len(u)

    def complexity(sel):
        # deterministic proxy: total number of graph edges traversed
        return sum(len(candidates[r]) for r in sel)

    # exact via MILP
    A = np.zeros((n_e, n_r))
    for j, cs in enumerate(cand_sets):
        for eid in cs:
            k = meaningful.index(eid)
            A[k, j] = 1.0
    c = np.ones(n_r)
    # deterministic tie-break: lexicographic route order micro-costs
    for j in range(n_r):
        c[j] += 1e-9 * j / max(1, n_r)
    res = milp(
        c=c,
        constraints=[LinearConstraint(A, lb=np.ones(n_e), ub=np.full(n_e, np.inf))],
        integrality=np.ones(n_r),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError("route cover MILP failed")

    chosen = [j for j in range(n_r) if res.x[j] > 0.5]
    best_count = len(chosen)

    # deterministic tie-break among equal-count covers
    import itertools
    best_key, best_sel = None, None
    idx_sorted = sorted(range(n_r),
                        key=lambda j: (_route_signature(candidates[j])))
    for size in range(best_count, best_count + 1):
        for combo in itertools.combinations(idx_sorted, size):
            u = set()
            for j in combo:
                u |= cand_sets[j]
            if not set(meaningful) <= u:
                continue
            cov = total_coverage(combo)
            cx = complexity(combo)
            key = (-cov, cx, tuple(combo))
            if best_key is None or key < best_key:
                best_key, best_sel = key, combo
        break  # fixed size = proven minimum

    return list(best_sel or chosen)


def route_edge_coverage(graph: RouteGraph,
                        selected: list[tuple]) -> np.ndarray:
    """Boolean coverage vector over all graph edges for the chosen
    routes (used by validation V1 and diagnostics)."""
    n_e = len(graph.edges)
    covered = np.zeros(n_e, dtype=bool)
    for sig_ids in selected:
        for eid in sig_ids:
            covered[eid] = True
    return covered


def route_coverage_fraction(graph: RouteGraph, selected: list[tuple]):
    """Fraction of meaningful edges covered by the selected routes."""
    if not graph.meaningful:
        return 1.0
    hit = set()
    for sig_ids in selected:
        hit |= set(sig_ids)
    return len(graph.meaningful & hit) / len(graph.meaningful)


@dataclass
class Corridor:
    """Allowed region for one complete route.

    Piecewise-linear [lower(x), upper(x)] from the route's slice
    intervals (with interior safety margin applied). (ylo, yhi) is the
    glyph vertical band the tails must escape permanently; escape
    directions are a deterministic fitting-time choice per side.
    """

    path: BoundaryPath
    xa: float
    xb: float
    xs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    ylo: float
    yhi: float

    def lower_at(self, x):
        return np.interp(x, self.xs, self.lower)

    def upper_at(self, x):
        return np.interp(x, self.xs, self.upper)


def build_route_corridor(graph: RouteGraph, edge_ids: tuple,
                         geom: GlyphGeometry) -> Corridor:
    """Corridor for one complete route: concatenated slice intervals of
    its graph edges, with CORRIDOR_MARGIN applied per column (reduced
    deterministically on thin slices; never inverted)."""
    xs_all, lo_all, hi_all = [], [], []
    for eid in edge_ids:
        e = graph.edges[eid]
        xs_all.extend(e.xs)
        lo_all.extend(e.lower)
        hi_all.extend(e.upper)

    xs = np.asarray(xs_all, dtype=float)
    lo = np.asarray(lo_all, dtype=float)
    hi = np.asarray(hi_all, dtype=float)
    keep = np.concatenate([[True], np.diff(xs) > 1e-12])
    xs, lo, hi = xs[keep], lo[keep], hi[keep]

    heights = hi - lo
    mm = np.minimum(CORRIDOR_MARGIN,
                    np.maximum(0.0, (heights - MIN_CORRIDOR_WIDTH)) / 2.0)
    lower = lo + mm
    upper = hi - mm

    center = (lower + upper) / 2.0
    path = BoundaryPath(points=np.column_stack([xs, center]),
                        contour_id=-1)

    pad = ESC_OFFSETS[-1] + 1.0   # window covers all escape checkpoints
    return Corridor(
        path=path,
        xa=float(xs[0] - pad),
        xb=float(xs[-1] + pad),
        xs=xs,
        lower=lower,
        upper=upper,
        ylo=float(geom.ymin),
        yhi=float(geom.ymax),
    )




# ---------------------------------------------------------------------------
# Combined stroke/hole route graph (skeleton-derived)
# ---------------------------------------------------------------------------

STROKE_LANDMARKS = 48        # corridor landmark samples per route
STROKE_RADIUS_GAIN = 1.6     # corridor half-width = gain * stroke radius
STROKE_MIN_HALF = 0.8        # never narrower than this


def _monotone_pieces(pts: np.ndarray):
    """Split a polyline into x-monotone pieces at local x extrema.
    Returns a list of (i0, i1) index ranges into pts."""
    n = len(pts)
    if n < 2:
        return []
    cuts = [0]
    for i in range(1, n - 1):
        dx0 = pts[i, 0] - pts[i - 1, 0]
        dx1 = pts[i + 1, 0] - pts[i, 0]
        if dx0 * dx1 < 0:
            cuts.append(i)
    cuts.append(n - 1)
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]


def build_stroke_route_graph(geom: GlyphGeometry) -> RouteGraph:
    """Route graph from the medial-axis stroke skeleton.

    Vertices: skeleton endpoints ('terminal'), junction clusters and
    x-extremum bends. Ring-shaped components (O) have their leftmost
    vertex retyped 'terminal' so the ring splits into two arcs.
    Edges: x-monotone polyline pieces carrying local thickness.
    """
    from src.skeleton import stroke_graph

    step = SIZE / GRID
    sg = stroke_graph(geom.fill)

    vertices: list[RouteVertex] = []
    edges: list[RouteEdge] = []

    def new_vertex(x, kind):
        v = RouteVertex(len(vertices), x, kind)
        vertices.append(v)
        return v.id

    node_vert = {n.id: new_vertex(n.xy[0] * step,
                                  "terminal" if n.kind == "end"
                                  else "junction")
                 for n in sg.nodes}

    # component membership (union-find over stroke nodes)
    parent = {n.id: n.id for n in sg.nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def add_piece(seg, a_vid, b_vid):
        if len(seg) < 2:
            return
        if seg[-1, 0] < seg[0, 0]:   # right-to-left piece: orient leftward
            seg = seg[::-1]
            a_vid, b_vid = b_vid, a_vid
        edges.append(RouteEdge(
            len(edges), a_vid, b_vid,
            xs=seg[:, 0].copy(),
            lower=np.zeros(len(seg)),
            upper=np.zeros(len(seg)),
            points=seg.copy(),
        ))

    for se in sg.edges:
        pts = se.points * np.array([step, step])   # pixel -> glyph coords
        if (len(pts) > 50
                and float(np.hypot(*(pts[0] - pts[-1]))) <= 2.0 * step):
            # closed ring: rotate to the global x-minimum and split into
            # two arcs between a fresh terminal vertex pair
            i0 = int(np.argmin(pts[:, 0]))
            pts = np.roll(pts, -i0, axis=0)
            mid = len(pts) // 2
            vid = new_vertex(float(pts[0, 0]), "terminal")
            comp_of_vertex_extra = None
            for half in (pts[:mid + 1], pts[mid:]):
                for j0, j1 in _monotone_pieces(half):
                    seg = half[j0:j1 + 1]
                    a_vid = (vid if j0 == 0
                             else new_vertex(float(seg[0, 0]), "bend"))
                    b_vid = (vid if j1 == len(half) - 1
                             else new_vertex(float(seg[-1, 0]), "bend"))
                    add_piece(seg, a_vid, b_vid)
            continue
        for i0, i1 in _monotone_pieces(pts):
            seg = pts[i0:i1 + 1]
            a_vid = (node_vert[se.a] if i0 == 0
                     else new_vertex(float(seg[0, 0]), "bend"))
            b_vid = (node_vert[se.b] if i1 == len(pts) - 1
                     else new_vertex(float(seg[-1, 0]), "bend"))
            add_piece(seg, a_vid, b_vid)
            union(se.a, se.b)

    # rings: components without any terminal node get their leftmost
    # vertex retyped 'terminal' (the O cut point)
    comp_of_vertex = {node_vert[n.id]: find(n.id) for n in sg.nodes}
    comp_min_x: dict[int, float] = {}
    comp_min_vert: dict[int, int] = {}
    for v in vertices:
        r = comp_of_vertex.get(v.id)
        if r is None or v.kind == "terminal":
            continue
        if r not in comp_min_x or v.x < comp_min_x[r]:
            comp_min_x[r] = v.x
            comp_min_vert[r] = v.id
    has_terminal = {find(n.id) for n in sg.nodes if n.kind == "end"}
    for r, vid in comp_min_vert.items():
        if r not in has_terminal:
            vertices[vid].kind = "terminal"

    for v in vertices:
        v.outgoing = tuple(e.id for e in edges if e.v_from == v.id)
        v.incoming = tuple(e.id for e in edges if e.v_to == v.id)

    meaningful = frozenset(e.id for e in edges if e.span >= SLIVER_SPAN)
    return RouteGraph(vertices=vertices, edges=edges, meaningful=meaningful)


def _route_corridor_from_stroke(graph: RouteGraph, edge_ids: tuple,
                                geom: GlyphGeometry) -> Corridor:
    """Corridor for one complete skeleton route.

    Landmarks are sampled evenly along the concatenated route polyline;
    each carries a vertical band around the skeleton point (local
    stroke region clamped to the fill). Constraint positions follow the
    landmarks' real x where the route progresses horizontally and spread
    deterministically across the window where it moves vertically, so
    traversal of every major feature is forced while one route always
    maps x to one connected interval.
    """
    from scipy import ndimage  # lazy

    step = SIZE / GRID
    pts_all = []
    tail = None
    for eid in edge_ids:
        p = graph.edges[eid].points
        if tail is not None:
            d_head = float(np.hypot(p[0, 0] - tail[0], p[0, 1] - tail[1]))
            d_last = float(np.hypot(p[-1, 0] - tail[0], p[-1, 1] - tail[1]))
            if d_last < d_head:
                p = p[::-1]
        pts_all.append(p)
        tail = pts_all[-1][-1]
    route_pts = np.vstack(pts_all)

    seg = np.hypot(*np.diff(route_pts, axis=0).T)
    s_arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_arc[-1])
    n_lm = min(STROKE_LANDMARKS, max(8, int(total)))
    targets = np.linspace(0.0, total, n_lm)
    lam = np.column_stack([
        np.interp(targets, s_arc, route_pts[:, 0]),
        np.interp(targets, s_arc, route_pts[:, 1]),
    ])

    radius = ndimage.distance_transform_edt(geom.fill)
    lo_list, hi_list = [], []
    for x_g, y_g in lam:
        col = int(round(x_g / step))
        row = int(round(y_g / step))
        col = min(max(col, 0), geom.fill.shape[1] - 1)
        row = min(max(row, 0), geom.fill.shape[0] - 1)
        half = max(STROKE_RADIUS_GAIN * float(radius[row, col]) * step,
                   STROKE_MIN_HALF)
        lo_list.append(y_g - half)
        hi_list.append(y_g + half)
    lo_arr = np.asarray(lo_list)
    hi_arr = np.asarray(hi_list)

    # Constraint positions: map arc fraction onto the route's own
    # x-range. Horizontal stretches land at their true x; vertical
    # stretches spread deterministically across the (narrow) window,
    # forcing traversal while keeping every position inside the glyph.
    x_lo = float(lam[:, 0].min())
    x_hi = float(lam[:, 0].max())
    if x_hi - x_lo < 1e-6:
        x_hi = x_lo + 1e-6
    p = x_lo + (x_hi - x_lo) * (targets / max(total, 1e-9))

    heights = hi_arr - lo_arr
    mm = np.minimum(CORRIDOR_MARGIN,
                    np.maximum(0.0, (heights - MIN_CORRIDOR_WIDTH)) / 2.0)
    lower = lo_arr + mm
    upper = hi_arr - mm

    center = (lower + upper) / 2.0
    path = BoundaryPath(points=np.column_stack([p, center]),
                        contour_id=-1)
    pad = ESC_OFFSETS[-1] + 1.0
    return Corridor(
        path=path,
        xa=float(p[0] - pad),
        xb=float(p[-1] + pad),
        xs=p,
        lower=lower,
        upper=upper,
        ylo=float(geom.ymin),
        yhi=float(geom.ymax),
    )
