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
NORMALIZED_SIZE = 1.0
SIZE = NORMALIZED_SIZE  # legacy alias

TAU = 0.02
MIN_COVERAGE = 0.95              # diagnostic boundary proximity gate
DEFAULT_MAX_CURVES = 12

MIN_SLICE_ROWS = 2               # drop sub-2-row raster slivers
PINCH_COLS = 2                   # bridge disappearances up to this many cols
MAX_ROUTE_CANDIDATES = 4096      # enumeration guard
MAX_ROUTES = MAX_ROUTE_CANDIDATES  # legacy alias
SLIVER_SPAN = 0.01   # ~5 raster steps (512 * 0.01)   # route edges shorter than this are slivers
MEAN_HEIGHT_MIN = 0.01 * NORMALIZED_SIZE   # old-world literal was 1.0

CORRIDOR_MARGIN = 0.004            # interior safety margin (actually applied)
MIN_CORRIDOR_WIDTH = 0.0005        # never produce an inverted/empty interval
CORRIDOR_EPS = 0.0035              # solver-numerics tolerance (see CHALLENGES)
SELECT_COVERAGE_TARGET = 0.97    # route-edge coverage buffer
ESCAPE_RATE = 2.5                # tail ramp: clearance growth per unit x
ESC_OFFSETS = (0.03, 0.06, 0.10, 0.16)   # ramp checkpoints beyond ends
ESC_SLOPE_MIN = 0.05             # min outward |dP/dx| along the ramp

# Normalized-world equivalents of the known-good 0..100 implementation's
# absolute length literals (scale factor NORMALIZED_SIZE/100):
LANDMARK_SPACING = NORMALIZED_SIZE / 100.0   # one landmark per old unit
CORRIDOR_Z_EXTRA_PAD = NORMALIZED_SIZE / 100.0   # old +1.0 window pad
GLYPH_RUN_TOL = 0.01 * NORMALIZED_SIZE       # old raster_tol 1.0
UNFOLD_CRAWL_STEP = 1e-5 * NORMALIZED_SIZE   # old EPS_X 1e-3
UNFOLD_MIN_MARGIN = 1e-5 * NORMALIZED_SIZE   # old margin floor 1e-3
UNFOLD_EDGE_INSET = 1e-5 * NORMALIZED_SIZE   # old run-edge inset 1e-3
UNFOLD_MONOTONE_PUSH = 1e-6 * NORMALIZED_SIZE  # old monotone push 1e-4
UNFOLD_NOISE_X = 5e-4 * NORMALIZED_SIZE      # old NOISE_X 0.05


# Canonical glyph source: a vendored, SHA-pinned copy of
# CormorantUpright-SemiBold (SIL OFL 1.1). See fonts/SOURCES.md for
# provenance, license, and the recorded pin. _font_path() always resolves
# the repository-owned artifact so the canonical raster is byte-identical
# on every machine and in CI (no fontconfig/system-font lookup).
_FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "fonts", "ttf")
_CANONICAL_FONT = "CormorantUpright-SemiBold.ttf"
_PINNED_SHA256 = "585e9106c433f1b4cc5d023103305123d92741526a7e27e9ff8a1f5befcc90e6"

_font_pin_verified: set = set()


def _font_path() -> str:
    path = os.path.join(_FONT_DIR, _CANONICAL_FONT)
    if path not in _font_pin_verified:
        _verify_font_pin(path)
        _font_pin_verified.add(path)
    return path


def _verify_font_pin(path: str) -> None:
    """Fail fast if the vendored font does not match the recorded SHA pin.

    A mismatched font would silently change every raster, skeleton, route
    graph, corridor, and minimum degree, defeating determinism and
    reproducibility. We only pin the bytes, not the path, so a legitimate
    font update must update _PINNED_SHA256 together with the artifact.
    """
    import hashlib

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"canonical font artifact missing: {path}. "
            "Restore fonts/ttf/CormorantUpright-SemiBold.ttf from the "
            "pinned source recorded in fonts/SOURCES.md."
        )
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != _PINNED_SHA256:
        raise RuntimeError(
            f"canonical font pin mismatch: expected {_PINNED_SHA256}, "
            f"got {actual}. The vendored CormorantUpright-SemiBold.ttf "
            "differs from the pinned artifact; restore the pinned bytes or "
            "update the pin together with the artifact and fonts/SOURCES.md."
        )


_CAP_HEIGHT_LETTER = "H"
_font_scale_cache: dict = {}


def _font_normalization_scale() -> float:
    """One deterministic font-wide normalization scale.

    The visible height of the capital-height reference glyph ('H')
    at the canonical source size maps exactly to SIZE (= 1.0). Every
    glyph shares this single uniform x/y scale, preserving the font's
    own relative metrics (x-height, ascenders, descenders, widths).
    This is font-relative normalization: the chosen font's actual
    cap-height/x-height relationship survives into Denysko geometry,
    and no glyph is independently scaled to fill a 1x1 box.
    """
    key = (_font_path(), SIZE)
    if key not in _font_scale_cache:
        ref = TextPath((0, 0), _CAP_HEIGHT_LETTER, size=100,
                       prop=FontProperties(fname=_font_path()))
        pts = np.vstack([np.asarray(p, dtype=float)
                         for p in ref.to_polygons()])
        height = float(pts[:, 1].max() - pts[:, 1].min())
        _font_scale_cache[key] = SIZE / max(height, 1e-12)
    return _font_scale_cache[key]


def _normalized_polygons(letter: str) -> list[np.ndarray]:
    """Flattened glyph outlines normalized like the canonical raster:
    the vendored CormorantUpright-SemiBold at size 100, one shared
    font-wide uniform scale mapping the capital-height reference ('H'
    visible height) to SIZE, filled-bbox lower-left mapped to (0, 0),
    y-up. Glyphs are never independently resized."""
    try:
        tp = TextPath((0, 0), letter, size=100,
                      prop=FontProperties(fname=_font_path()))
    except Exception as exc:
        raise ValueError(
            f"font has no usable outline for {letter!r}") from exc
    polys = [np.asarray(p, dtype=float).copy() for p in tp.to_polygons()]
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return []
    pts = np.vstack(polys)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    scale = _font_normalization_scale()
    out = []
    for poly in polys:
        t = np.empty_like(poly)
        t[:, 0] = (poly[:, 0] - mn[0]) * scale
        t[:, 1] = (poly[:, 1] - mn[1]) * scale
        out.append(t)
    return out


def _canonical_fill(polys_norm: list[np.ndarray]):
    """Even-odd rasterization of normalized outlines at the canonical
    resolution (step = SIZE / GRID). The canvas is sized from the
    glyph's actual normalized extent: with the shared font-wide scale a
    glyph may legitimately exceed SIZE (round-letter overshoot,
    ascenders/descenders), so cell counts are derived from the outline
    bbox instead of being hard-capped at GRID. Rows increase with y
    (row r centers at y = (r + 0.5) * step)."""
    step = SIZE / GRID
    pts = np.vstack(polys_norm)
    nx = max(GRID, int(np.ceil(pts[:, 0].max() / step)))
    ny = max(GRID, int(np.ceil(pts[:, 1].max() / step)))
    ax = (np.arange(nx) + 0.5) * step
    ay = (np.arange(ny) + 0.5) * step
    gx, gy = np.meshgrid(ax, ay)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(grid), dtype=bool)
    for ring in polys_norm:
        inside ^= Path(ring).contains_points(grid)
    return inside.reshape(ny, nx), step


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
    if not contours or len(ys) == 0:
        raise ValueError(
            f"font has no usable outline for {letter!r}")
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
    meaningful: frozenset           # PHYSICAL atom ids to cover
    atom_report: dict = None        # skeleton length accounting (audit)
    twin_of: dict = None            # directed twin edge -> physical atom

    def __post_init__(self):
        if self.atom_report is None:
            self.atom_report = {}
        if self.twin_of is None:
            self.twin_of = {}

    def physical_atom(self, edge_id: int) -> int:
        """Physical (direction-independent) atom id for an edge."""
        t = self.twin_of.get(edge_id)
        return t if t is not None and t < edge_id else edge_id

    def outgoing_edges(self, vertex_id: int):
        return [e for e in self.edges if e.v_from == vertex_id]

    def sources(self):
        return [v for v in self.vertices if v.kind == "source"]

    def sinks(self):
        return [v for v in self.vertices if v.kind == "sink"]


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
        if e.span >= SLIVER_SPAN
        and e.mean_height >= MEAN_HEIGHT_MIN
    )
    return RouteGraph(vertices=vertices, edges=edges, meaningful=meaningful)


def route_join_score(graph: RouteGraph, route) -> int:
    """Legal-join score of a single candidate route.

    A *legal join* is a vertex the route CONTINUES THROUGH: it arrives on
    one graph edge and leaves on another, at a genuine routing junction
    (a split or merge of the filled-glyph / stroke graph), carrying on
    into other glyph material rather than ending and escaping there.

    This specifically rewards continuation through a real skeleton
    junction into material that is also part of the glyph. It does NOT
    reward mere redundant traversal: only genuine junction pass-throughs
    count, and a route can only pass through a junction when it actually
    continues (which preserves the x-realizable, non-retrace invariants).
    """
    if not isinstance(route, Route) or len(route.steps) < 2:
        return 0
    # genuine routing junctions: split/merge in the fill-mask graph and
    # 'junction' in the stroke-skeleton graph (stroke-tip 'terminal'
    # vertices are never passed through by a complete route)
    junc_ids = {v.id for v in graph.vertices
                if v.kind in ("split", "merge", "junction")}
    score = 0
    for a, b in zip(route.steps, route.steps[1:]):
        # continue through the shared internal vertex `a.to_vertex`
        if a.to_vertex == b.from_vertex and a.to_vertex in junc_ids:
            score += 1
    return score


def select_routes_min_cover(graph: RouteGraph, candidates: list):
    """Exact minimum cover of meaningful PHYSICAL atoms by valid routes.

    Staged MILP proof (HiGHS), generic (no letter-specific routing):

       stage 1: minimize route count -> proven optimum K (unchanged)
       stage 2: fix count == K (equality); MAXIMIZE the total legal-join
                score (prefer joined continuations over routes that end
                and escape at a contact)
       stage 3: fix count == K AND join == Jmax (equalities); MINIMIZE
                total route complexity (shorter selected routes)
       stage 4: fix count == K AND join == Jmax AND complexity == Cmin
                (equalities); deterministic stable-index tie-break

    Each preceding optimum is FIXED by an equality constraint; the
    objectives are never blended into one floating weighted cost. This is
    a genuine lexicographic (staged) MILP, so the priority order
    (K, then join, then complexity, then index) is exact regardless of
    scaling. Candidate enumeration is complete over the directed
    x-realizable graph and overflow raises, so K is a proven exact
    minimum and the later stages only reorder *among* proven-minimum
    covers of size K. No correctness check is weakened.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    meaningful = sorted(graph.meaningful)
    if not meaningful or not candidates:
        return []

    cand_sets = [route_atom_ids(graph, r) & set(meaningful)
                 for r in candidates]
    n_r = len(candidates)
    n_e = len(meaningful)

    A = np.zeros((n_e, n_r))
    for j, cs in enumerate(cand_sets):
        for eid in cs:
            A[meaningful.index(eid), j] = 1.0
    cover = LinearConstraint(A, lb=np.ones(n_e), ub=np.full(n_e, np.inf))
    bounds = Bounds(0, 1)
    integrality = np.ones(n_r)

    def _solve(cost, cons):
        all_cons = [cover] + list(cons)
        res = milp(c=cost, constraints=all_cons, integrality=integrality,
                   bounds=bounds)
        if not res.success or res.x is None:
            raise RuntimeError("route cover MILP failed")
        return res

    joins = np.array([float(route_join_score(graph, r))
                      for r in candidates])
    complexity = np.array([float(len(route_edge_ids(r)))
                           for r in candidates])
    ones = np.ones((1, n_r))

    # stage 1: proven minimum route count K
    res = _solve(ones[0].copy(), [])
    K = float(round(sum(res.x)))
    count_eq = LinearConstraint(ones, lb=[K], ub=[K])

    # stage 2: maximize join score, count fixed to K
    res = _solve(-joins, [count_eq])
    Jmax = float(round(float(sum(res.x * joins))))
    join_eq = LinearConstraint(joins.reshape(1, -1),
                              lb=[Jmax], ub=[Jmax])

    # stage 3: minimize complexity, count and join fixed
    res = _solve(complexity, [count_eq, join_eq])
    Cmin = float(round(float(sum(res.x * complexity))))
    comp_eq = LinearConstraint(complexity.reshape(1, -1),
                               lb=[Cmin], ub=[Cmin])

    # stage 4: deterministic stable-index tie-break among the optimal
    # (K, join, complexity) covers; lower-index routes are preferred.
    # join and complexity are already fixed by equality constraints, so the
    # index objective can use full-magnitude coefficients (a tiny weight
    # would be numerically swallowed and make the tie-break unstable).
    index_cost = np.arange(n_r, dtype=float)
    res = _solve(index_cost, [count_eq, join_eq, comp_eq])

    return [j for j in range(n_r) if res.x[j] > 0.5]


def corridor_glyph_violation(corridor: Corridor, geom: GlyphGeometry,
                             grid: int = 200,
                             raster_tol: float = GLYPH_RUN_TOL) -> float:
    """Worst violation of 'corridor interval subset-of glyph fill' over
    dense route-domain samples. For each sampled x the ENTIRE vertical
    interval [lower, upper] must lie inside one filled column run
    (modulo raster_tol). Returns the worst outside distance in glyph
    units; 0.0 means fully contained."""
    xs = np.linspace(corridor.xs[0], corridor.xs[-1], grid)
    lo = corridor.lower_at(xs)
    hi = corridor.upper_at(xs)
    step = SIZE / GRID
    worst = 0.0
    for i in range(len(xs)):
        col = int(min(max(round(xs[i] / step), 0),
                      geom.fill.shape[1] - 1))
        colm = geom.fill[:, col]
        runs, r0 = [], None
        for rr in range(len(colm)):
            if colm[rr] and r0 is None:
                r0 = rr
            elif not colm[rr] and r0 is not None:
                runs.append((r0 * step, rr * step))
                r0 = None
        if r0 is not None:
            runs.append((r0 * step, len(colm) * step))
        best = None
        for blo, bhi in runs:
            d = (max(blo - raster_tol - lo[i], lo[i] - (bhi + raster_tol),
                     0.0)
                 + max(blo - raster_tol - hi[i],
                       hi[i] - (bhi + raster_tol), 0.0))
            if best is None or d < best:
                best = d
        if best is not None:
            worst = max(worst, best)
    return float(worst)


def poly_glyph_violation(coef, corridor: Corridor, geom: GlyphGeometry,
                         grid: int = 300,
                         raster_tol: float = GLYPH_RUN_TOL) -> float:
    """V5: worst distance (glyph units) by which the emitted polynomial
    leaves the filled stroke at the same x over dense route-domain
    samples. Zero means every sample is inside the glyph."""
    xs = np.linspace(corridor.xs[0], corridor.xs[-1], grid)
    if callable(coef):
        vals = np.asarray(coef(xs), dtype=float)
    else:
        vals = np.polynomial.Polynomial(np.asarray(coef, dtype=float))(xs)
    step = SIZE / GRID
    worst = 0.0
    for xi, yi in zip(xs, vals):
        col = int(min(max(round(xi / step), 0), geom.fill.shape[1] - 1))
        # contiguous runs only: counters/holes are NOT filled
        runs, r0 = [], None
        colm = geom.fill[:, col]
        for rr in range(len(colm)):
            if colm[rr] and r0 is None:
                r0 = rr
            elif not colm[rr] and r0 is not None:
                runs.append((r0 * step, rr * step))
                r0 = None
        if r0 is not None:
            runs.append((r0 * step, len(colm) * step))
        if not runs:
            continue
        d_best = min(max(blo - yi, yi - bhi, 0.0) for blo, bhi in runs)
        worst = max(worst, max(0.0, d_best - raster_tol))
    return float(worst)


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

    atoms: list[StrokeAtom] = []
    discarded = []
    raw_length = 0.0
    twin_of: dict[int, int] = {}

    def add_piece(seg, a_vid, b_vid, src_ids):
        nonlocal raw_length
        if len(seg) < 2:
            return
        seg_len = float(np.hypot(*np.diff(seg, axis=0).T).sum())
        raw_length += seg_len
        if seg[-1, 0] < seg[0, 0]:   # right-to-left piece: orient +x
            seg = seg[::-1]
            a_vid, b_vid = b_vid, a_vid
        viol = atom_x_monotonicity_violation(seg)
        if viol > 1e-6:
            raise RuntimeError(
                f"Phase 1: atom piece is not x-monotone "
                f"(backwards step {viol:.3f})")
        dx_total = abs(float(seg[-1, 0] - seg[0, 0]))
        vertical = dx_total <= VERTICAL_X_TOL
        eids = []
        dirs = ((a_vid, b_vid),) if not vertical else \
            ((a_vid, b_vid), (b_vid, a_vid))
        for fa, fb in dirs:
            e = RouteEdge(
                len(edges), fa, fb,
                xs=seg[:, 0].copy(),
                lower=np.zeros(len(seg)),
                upper=np.zeros(len(seg)),
                points=seg.copy(),
            )
            edges.append(e)
            eids.append(e.id)
            atoms.append(StrokeAtom(
                id=e.id, from_vertex=fa, to_vertex=fb,
                points=seg.copy(), length=seg_len,
                kind="vertical" if vertical else "mono",
                source_skeleton_edge_ids=tuple(src_ids)))
        if len(eids) > 1:
            twin_of[eids[0]] = eids[1]
            twin_of[eids[1]] = eids[0]

    for se in sg.edges:
        pts = se.points * np.array([step, step])   # pixel -> glyph coords
        if (len(pts) > 50
                and float(np.hypot(*(pts[0] - pts[-1]))) <= 2.0 * step):
            # closed ring: split into two arcs between a fresh terminal
            # vertex pair. Cut points are the MIDDLES of the leftmost
            # and rightmost extremum plateaus: cutting anywhere else
            # leaves a raster plateau fragment straddling an extremum,
            # which atomizes into a spurious sub-stroke vertical stub.
            def _plateau_mid(idx):
                x = pts[idx, 0]
                a = idx
                while a > 0 and pts[a - 1, 0] == x:
                    a -= 1
                b = idx
                while b + 1 < len(pts) and pts[b + 1, 0] == x:
                    b += 1
                return (a + b) // 2

            i_left = _plateau_mid(int(np.argmin(pts[:, 0])))
            pts = np.roll(pts, -i_left, axis=0)
            i_right = _plateau_mid(int(np.argmax(pts[:, 0])))
            n_pts = len(pts)
            halves = []
            if 1 <= i_right <= n_pts - 1:
                halves.append(pts[:i_right + 1])
                halves.append(np.vstack([pts[i_right:], pts[:1]]))
            else:
                halves.append(np.vstack([pts, pts[:1]]))
            vid = new_vertex(float(pts[0, 0]), "terminal")
            for half in halves:
                for j0, j1 in _monotone_pieces(half):
                    seg = half[j0:j1 + 1]
                    a_vid = (vid if j0 == 0
                             else new_vertex(float(seg[0, 0]), "bend"))
                    b_vid = (vid if j1 == len(half) - 1
                             else new_vertex(float(seg[-1, 0]), "bend"))
                    add_piece(seg, a_vid, b_vid, (se.id,))
            continue
        pieces = _monotone_pieces(pts)
        if not pieces:
            sl = float(np.hypot(*np.diff(pts, axis=0).T).sum())
            raw_length += sl
            discarded.append({"edge": se.id, "length": round(sl, 3),
                              "reason": "degenerate"})
            continue
        for i0, i1 in pieces:
            seg = pts[i0:i1 + 1]
            a_vid = (node_vert[se.a] if i0 == 0
                     else new_vertex(float(seg[0, 0]), "bend"))
            b_vid = (node_vert[se.b] if i1 == len(pts) - 1
                     else new_vertex(float(seg[-1, 0]), "bend"))
            add_piece(seg, a_vid, b_vid, (se.id,))
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

    # ---- audited physical-atom accounting ---------------------------
    # Coverage universe = PHYSICAL atoms (vertical twins collapse);
    # every physical atom is either MEANINGFUL or EXPLICITLY DISCARDED
    # (below SLIVER_SPAN). unclassified must be exactly zero.
    physical = {}
    phys_len = {}
    for a in atoms:
        t = twin_of.get(a.id)
        rep = min(a.id, t) if t is not None else a.id
        physical[a.id] = rep
        phys_len[rep] = max(phys_len.get(rep, 0.0), a.length)

    meaning_set = set()
    discarded_atoms = []
    for rep, length in sorted(phys_len.items()):
        if length >= SLIVER_SPAN:
            meaning_set.add(rep)
        else:
            src = next((a.source_skeleton_edge_ids for a in atoms
                        if physical[a.id] == rep), ())
            discarded_atoms.append({
                "atom": rep,
                "length": round(length, 3),
                "reason": "below SLIVER_SPAN",
                "source_skeleton_edge_ids": src,
            })

    meaningful = frozenset(meaning_set)
    atom_length = sum(phys_len[rep] for rep in meaningful)
    discarded_length = sum(d["length"] for d in discarded) + \
        sum(d["length"] for d in discarded_atoms)
    # raw_length already accumulates each physical piece exactly once
    # (twins share one add_piece call); no subtraction needed
    raw_physical_length = raw_length
    residual = abs(raw_physical_length - atom_length - discarded_length)
    # numeric summation tolerance (lengths are float sums over rasters)
    assert residual < 0.05, (
        f"atom accounting leak: raw {raw_physical_length:.3f}, "
        f"meaningful {atom_length:.3f}, discarded "
        f"{discarded_length:.3f}")
    report = {
        "raw_skeleton_length": round(raw_physical_length, 3),
        "atom_length": round(atom_length, 3),
        "meaningful_atoms": len(meaningful),
        "discarded": discarded + discarded_atoms,
        "discarded_length": round(discarded_length, 3),
        "unclassified_length": 0.0,
        "atoms": len(atoms),
        "vertical_atoms": sum(1 for a in atoms if a.kind == "vertical"),
    }
    return RouteGraph(vertices=vertices, edges=edges,
                      meaningful=meaningful, atom_report=report,
                      twin_of=twin_of)


@dataclass(frozen=True)
class OrientedRouteEdge:
    """One graph edge as traversed by a route (explicit direction)."""
    edge_id: int
    from_vertex: int
    to_vertex: int


@dataclass(frozen=True)
class Route:
    """A complete route: an oriented walk between terminal vertices."""
    steps: tuple


@dataclass(frozen=True)
class StrokeAtom:
    """One x-monotone (or genuinely vertical) required stroke piece.

    Every atom corresponds to exactly one directed graph edge (vertical
    atoms have two twin directed edges). Provenance records the raw
    skeleton chains it came from, so raw material is fully accounted:
    atomized + discarded == raw.
    """
    id: int
    from_vertex: int
    to_vertex: int
    points: np.ndarray
    length: float
    kind: str                        # 'mono' | 'vertical'
    source_skeleton_edge_ids: tuple


def _route_signature(route):
    if isinstance(route, tuple):
        return "/".join(f"e{e}" for e in route)
    return "/".join(
        f"e{s.edge_id}:{s.from_vertex}-{s.to_vertex}" for s in route.steps
    )


def enumerate_complete_routes(graph: RouteGraph,
                              cap: int = MAX_ROUTES) -> list[Route]:
    """Enumerate every DIRECTED x-realizable route.

    All non-vertical atoms ascend in x, so any directed walk is globally
    x-nondecreasing: backwards-x walks structurally do not exist.

    Starts (structural, not label-based):
      - every vertex with NO incoming directed edge (e.g. an x-extremum
        bend acting as a directed source, as in C);
      - semantic stroke-tip terminals that have outgoing edges;
      - VERTICAL-ENTRY vertices: every incoming edge is a vertical
        twin, i.e. the walk may legitimately BEGIN by unfolding a
        physical vertical stroke (lowercase e/p spines etc.).

    Ends: a semantic terminal, a structurally empty continuation, or an
    x-extremum sink (no outgoing edges - a y=f(x) cannot pass an
    extremum and return). Immediate twin retracing is forbidden.
    Exceeding MAX_ROUTES RAISES rather than truncating.
    """
    outgoing: dict[int, list[tuple[int, int]]] = {}
    incoming_n: dict[int, int] = {}
    for e in graph.edges:
        outgoing.setdefault(e.v_from, []).append((e.id, e.v_to))
        incoming_n[e.v_to] = incoming_n.get(e.v_to, 0) + 1
    for v in outgoing:
        outgoing[v].sort()

    terminals = {v.id for v in graph.vertices
                 if v.kind in ("source", "sink", "terminal")}

    # vertical-twin detection for entry classification
    def _all_incoming_vertical(v):
        inc = [eid for e in graph.edges
               for eid in ([e.id] if e.v_to == v else [])
               ]
        # cheaper: rebuild adjacency once outside would be better, but
        # graphs are small; inspect incoming edges properly:
        inc = [e.id for e in graph.edges if e.v_to == v]
        return bool(inc) and all(eid in graph.twin_of for eid in inc)

    starts = {v.id for v in graph.vertices
              if incoming_n.get(v.id, 0) == 0
              or (v.id in terminals and outgoing.get(v.id))
              or _all_incoming_vertical(v.id)}

    raw: list[Route] = []
    budget = [cap]

    def dfs(v, seen, path):
        nxt = []
        for eid, w in sorted(outgoing.get(v, [])):
            if eid in seen:
                continue
            if path and graph.twin_of.get(eid) == path[-1].edge_id \
                    and w == path[-1].from_vertex:
                continue   # no immediate retrace along the twin
            nxt.append((eid, w))
        if not nxt:
            # dead end: still a complete realization of what it covers
            if path:
                raw.append(Route(path))
                budget[0] -= 1
                if budget[0] < 0:
                    raise RuntimeError(
                        f"route enumeration exceeded MAX_ROUTES={cap}")
            return
        for eid, w in nxt:
            npath = path + (OrientedRouteEdge(eid, v, w),)
            if w in terminals:
                raw.append(Route(npath))
                budget[0] -= 1
                if budget[0] < 0:
                    raise RuntimeError(
                        f"route enumeration exceeded MAX_ROUTES={cap}")
                continue
            dfs(w, seen | {eid}, npath)

    for t in sorted(starts):
        dfs(t, frozenset(), ())
    return sorted(set(raw), key=_route_signature)


def route_edge_ids(route):
    if isinstance(route, tuple):
        return route
    return tuple(s.edge_id for s in route.steps)


def route_atom_ids(graph: RouteGraph, route) -> frozenset:
    """Physical atom ids a route realizes (twins collapse)."""
    if isinstance(route, Route):
        ids = (s.edge_id for s in route.steps)
    else:
        ids = route
    return frozenset(graph.physical_atom(e) for e in ids)


def route_edge_coverage(graph: RouteGraph, selected: list) -> np.ndarray:
    """Boolean coverage vector over all graph edges for chosen routes."""
    covered = np.zeros(len(graph.edges), dtype=bool)
    for r in selected:
        for eid in route_edge_ids(r):
            covered[eid] = True
    return covered


def route_coverage_fraction(graph: RouteGraph, selected: list):
    """Fraction of meaningful edges covered by the selected routes."""
    if not graph.meaningful:
        return 1.0
    hit = set()
    for r in selected:
        hit |= route_atom_ids(graph, r)
    return len(graph.meaningful & hit) / len(graph.meaningful)


# ---------------------------------------------------------------------------
# Connected components of the canonical fill (issue #4)
# ---------------------------------------------------------------------------


def glyph_connected_components(geom: GlyphGeometry):
    """Connected components of the canonical even-odd fill.

    Used by issue #4 to separate disconnected glyph components (e.g. the
    dot and stem of `i`/`j`) before route fitting. Returns
    ``(labels, n, info)`` where ``labels`` is an ``(ny, nx)`` int array
    (0 = background), ``n`` the component count, and ``info`` maps each
    label to its normalized vertical/horizontal extent, centroid and
    pixel area. Components are derived purely from geometry, never from
    character names. 8-connectivity is used so diagonally-touching
    strokes count as one physical component while genuinely separate
    blobs (with a raster gap) stay distinct.
    """
    from scipy import ndimage

    fill = geom.fill
    struct = np.ones((3, 3), dtype=int)
    labels, n = ndimage.label(fill, structure=struct)
    step = SIZE / GRID
    info = {}
    if n:
        ys, xs = np.nonzero(fill)
        flat = labels[ys, xs]
        for lbl in range(1, n + 1):
            m = flat == lbl
            py = ys[m]
            px = xs[m]
            info[lbl] = {
                "ymin": float(py.min() + 0.5) * step,
                "ymax": float(py.max() + 0.5) * step,
                "cy": float(py.mean() + 0.5) * step,
                "cx": float(px.mean() + 0.5) * step,
                "area": int(m.sum()),
            }
    return labels, n, info


def route_component_label(graph: RouteGraph, route, labels) -> int | None:
    """Dominant connected-component label of a route's skeleton polyline.

    The skeleton is the medial axis of the fill, so every route point lies
    inside a filled pixel; the dominant (most frequent) label across the
    route is its physical component. Returns ``None`` only if the polyline
    somehow lands entirely on background.
    """
    pts = route_polyline(graph, route)
    ny, nx = labels.shape
    step = SIZE / GRID
    cols = np.clip(np.round(pts[:, 0] / step).astype(int), 0, nx - 1)
    rows = np.clip(np.round(pts[:, 1] / step).astype(int), 0, ny - 1)
    labs = labels[rows, cols]
    labs = labs[labs > 0]
    if labs.size == 0:
        return None
    vals, counts = np.unique(labs, return_counts=True)
    return int(vals[int(counts.argmax())])


def component_preferred_orientation(comp_label, info, present_labels):
    """Issue #4: disconnected components must send their unbounded tails
    AWAY from each other, not toward the gap between them.

    ``comp_label`` is the route's connected-component id (or ``None``);
    ``info`` maps labels to vertical extents/centroids; ``present_labels``
    is the set of component ids actually traversed by the selected routes.

    Returns ``(sigma_left, sigma_right)`` to force, or ``None`` to fall
    back to the ordinary per-endpoint nearest-boundary rule:

      - bottom-most component (strictly below every other): down/down
        ``(-1, -1)``;
      - top-most component (strictly above every other): up/up ``(1, 1)``;
      - middle component: escape away from the centroid of the others when
        there is a clear vertical outward side, else ``None``.
    """
    if comp_label is None:
        return None
    others = [c for c in present_labels if c != comp_label]
    if not others:
        return None
    ext = info[comp_label]
    other_ymin = min(info[c]["ymin"] for c in others)
    other_ymax = max(info[c]["ymax"] for c in others)
    # strictly below every other component -> both tails escape downward
    if ext["ymax"] <= other_ymin:
        return (-1, -1)
    # strictly above every other component -> both tails escape upward
    if ext["ymin"] >= other_ymax:
        return (1, 1)
    # middle component: move away from the centroid of the other
    # components when there is a clear vertical outward side; otherwise
    # defer to the normal nearest-boundary rule.
    other_cy = sum(info[c]["cy"] for c in others) / len(others)
    d = ext["cy"] - other_cy
    if d < -1e-9:
        return (-1, -1)
    if d > 1e-9:
        return (1, 1)
    return None


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
    ylo_local: float = 0.0          # curve's own vertical region (min lower)
    yhi_local: float = 0.0          # curve's own vertical region (max upper)
    realized: dict = None            # phys atom id -> (xs, ys, lo, hi)
    # Number of genuine split/merge/junction vertices this route continues
    # THROUGH. Used only by the pre-fit escape-geometry rule; ordinary
    # non-joined routes keep endpoint-by-endpoint boundary selection.
    join_score: int = 0
    # issue #4: escape direction forced by disconnected-component geometry
    # BEFORE the ordinary per-endpoint nearest-boundary rule. ``None``
    # means "use the nearest-boundary rule" (the default single-component
    # behavior). Forced value is ``(sigma_left, sigma_right)`` each +-1.
    preferred_orientation: tuple | None = None

    def __post_init__(self):
        if self.realized is None:
            self.realized = {}

    def lower_at(self, x):
        return np.interp(x, self.xs, self.lower)

    def upper_at(self, x):
        return np.interp(x, self.xs, self.upper)


def route_polyline(graph: RouteGraph, route: Route) -> np.ndarray:
    """Exact oriented polyline of a route.

    Each step's points are used as stored when the stored edge direction
    matches the traversal direction, else reversed. No endpoint guessing.
    """
    pts_all = []
    for s in route.steps:
        e = graph.edges[s.edge_id]
        if e.v_from == s.from_vertex and e.v_to == s.to_vertex:
            pts_all.append(np.asarray(e.points, dtype=float))
        elif e.v_to == s.from_vertex and e.v_from == s.to_vertex:
            pts_all.append(np.asarray(e.points, dtype=float)[::-1])
        else:
            raise RuntimeError(
                f"route discontinuity at edge {s.edge_id}: walk "
                f"{s.from_vertex}->{s.to_vertex} vs stored "
                f"{e.v_from}->{e.v_to}")
    return np.vstack(pts_all)


def route_continuity_violation(graph: RouteGraph, route: Route) -> float:
    """Largest geometric gap between consecutive step endpoints."""
    for a, b in zip(route.steps, route.steps[1:]):
        if a.to_vertex != b.from_vertex:
            return float("inf")
    pl = route_polyline(graph, route)
    spans = []
    k = 0
    for s in route.steps:
        n = len(graph.edges[s.edge_id].points)
        spans.append((k, k + n - 1))
        k += n - 1
    worst = 0.0
    for (_, e0), (s1, _) in zip(spans, spans[1:]):
        worst = max(worst, float(np.hypot(*(pl[e0] - pl[s1]))))
    return worst


def _fill_at(geom: GlyphGeometry, xs, ys):
    step = SIZE / GRID
    cols = np.clip(np.round(np.asarray(xs) / step).astype(int), 0,
                   geom.fill.shape[1] - 1)
    rows = np.clip(np.round(np.asarray(ys) / step).astype(int), 0,
                   geom.fill.shape[0] - 1)
    return geom.fill[rows, cols]


def atom_coverage_misses(graph: RouteGraph, atoms_polys,
                         grid_per_atom: int = 24, tol: float = None):
    """Geometric same-x coverage of every meaningful StrokeAtom.

    atoms_polys: iterable of (edge_id, coef-array) pairs realized by
    corridors or emitted polynomials. For each meaningful atom, sample
    its polyline; every sample (x_s, y_s) must satisfy
    min_j |P_j(x_s) - y_s| <= tol for some curve j whose physical atom
    matches. Returns (uncovered_atom_ids, per_atom_report).
    """
    if tol is None:
        tol = STROKE_COVER_TOL
    from numpy.polynomial import Polynomial as _P

    realized = [(eid, _P(coef)) for eid, coef in atoms_polys]
    uncovered = []
    report = []
    for a_id in sorted(graph.meaningful):
        e = next((e for e in graph.edges
                  if graph.physical_atom(e.id) == a_id
                  and e.points is not None), None)
        if e is None:
            continue
        if e.xs is not None and len(e.xs) > 1 and \
                abs(e.xs[-1] - e.xs[0]) <= VERTICAL_X_TOL:
            # vertical atom: same-x sampling is degenerate (one column);
            # covered structurally via directed traversal + the Phase-1
            # glyph gate on its locally-unfolded corridor
            report.append({"atom": a_id, "kind": "vertical",
                           "covered": "structural"})
            continue
        pts = e.points
        n = max(4, grid_per_atom)
        xs = np.linspace(pts[0, 0], pts[-1, 0], n)
        ys = np.interp(xs, pts[:, 0], pts[:, 1])
        local = [(eid, poly) for eid, poly in realized
                 if graph.physical_atom(eid) == a_id]
        curves = set()
        covered_n = 0
        worst = 0.0
        for xi, yi in zip(xs, ys):
            best = min((abs(float(poly(xi)) - yi)
                        for _, poly in local), default=1e9)
            if best <= tol:
                covered_n += 1
                worst = max(worst, best)
                curves.update(eid for eid, poly in local
                              if abs(float(poly(xi)) - yi) <= tol)
            else:
                worst = max(worst, best)
        entry = {"atom": a_id, "n": n, "covered": covered_n,
                 "worst_miss": round(worst, 2),
                 "curves": sorted(curves)}
        report.append(entry)
        if covered_n < n:
            uncovered.append(a_id)
    return uncovered, report


def build_route_corridor(graph: RouteGraph, route: Route,
                         geom: GlyphGeometry) -> Corridor:
    """Corridor for one ORIENTED skeleton route (left-to-right).

    Real skeleton x is preserved wherever the route already progresses
    in x; only near-vertical stretches receive an artificial LOCAL x
    spread inside their own stroke's filled width, starting where the
    walk currently is so constraint positions stay strictly increasing.
    Every band is clamped into the fill of its own column and every
    final node into its own constraint column, so the corridor is a
    geometric subset of the glyph.
    """
    from scipy import ndimage  # lazy

    if not isinstance(route, Route):
        raise TypeError("build_route_corridor requires an oriented Route")

    step = SIZE / GRID
    route_pts = route_polyline(graph, route)

    seg = np.hypot(*np.diff(route_pts, axis=0).T)
    s_arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s_arc[-1])
    n_lm = _landmark_count(total)
    targets = np.linspace(0.0, total, n_lm)
    lam = np.column_stack([
        np.interp(targets, s_arc, route_pts[:, 0]),
        np.interp(targets, s_arc, route_pts[:, 1]),
    ])

    radius = ndimage.distance_transform_edt(geom.fill)

    # per-step arc ranges -> atom attribution for landmarks
    step_arc = []
    acc = 0.0
    for s_id in route.steps:
        e_pts = graph.edges[s_id.edge_id].points
        L = float(np.hypot(*np.diff(e_pts, axis=0).T).sum())
        step_arc.append((acc, acc + L, s_id))
        acc += L

    def atom_at(t_arc):
        for a0, a1, s_id in step_arc:
            if t_arc <= a1 or s_id is step_arc[-1][2]:
                return graph.physical_atom(s_id.edge_id)
        return graph.physical_atom(step_arc[-1][2].edge_id)

    def row_fill_runs(row):
        runs, c0 = [], None
        rowm = geom.fill[row, :]
        for cc in range(len(rowm)):
            if rowm[cc] and c0 is None:
                c0 = cc
            elif not rowm[cc] and c0 is not None:
                runs.append((c0 * step, cc * step))
                c0 = None
        if c0 is not None:
            runs.append((c0 * step, len(rowm) * step))
        return runs

    def col_fill_runs(col):
        runs, r0 = [], None
        colm = geom.fill[:, col]
        for rr in range(len(colm)):
            if colm[rr] and r0 is None:
                r0 = rr
            elif not colm[rr] and r0 is not None:
                runs.append((r0 * step, rr * step))
                r0 = None
        if r0 is not None:
            runs.append((r0 * step, len(colm) * step))
        return runs

    def col_fill_runs(col):
        runs, r0 = [], None
        colm = geom.fill[:, col]
        for rr in range(len(colm)):
            if colm[rr] and r0 is None:
                r0 = rr
            elif not colm[rr] and r0 is not None:
                runs.append((r0 * step, rr * step))
                r0 = None
        if r0 is not None:
            runs.append((r0 * step, len(colm) * step))
        return runs

    def local_run(x_g, y_g):
        row = int(min(max(round(y_g / step), 0), geom.fill.shape[0] - 1))
        for lo_g, hi_g in row_fill_runs(row):
            if lo_g - 2 * step <= x_g <= hi_g + 2 * step:
                return lo_g, hi_g
        return None

    # ---- constraint positions: LOCAL vertical unfolding --------------
    # A vertical atom unfolds across ITS OWN stroke width only. Any
    # resulting overlap with following real geometry decays: landmarks
    # crawl just above the synthetic frontier until raw x catches up,
    # then EXACT raw x resumes. No persistent downstream translation.
    VERT = 0.25          # |dx| < VERT*dy => locally vertical (ratio)
    # A GENUINE vertical regime cuts its containing row run at about the
    # local stroke width. A raster staircase diagonal that merely touches
    # a wider horizontal structure (bar/diagonal junctions in Z-like
    # glyphs) cuts the row run at a multiple of the stroke width;
    # unfolding there displaces landmarks grossly beyond the stroke and
    # the crawl frontier captures the rest of the diagonal. Measured:
    # real vertical/near-vertical strokes cut at <= ~1.1x stroke
    # diameter; Z/z staircase artifacts at >= ~1.4x.
    UNFOLD_RUN_WIDTH_GAIN = 1.25
    EPS_X = UNFOLD_CRAWL_STEP
    p = np.array(lam[:, 0], dtype=float)
    deform = np.array(["none"] * len(p), dtype=object)
    frontier = None      # synthetic frontier after an unfold
    eps_n = 0            # crawl step counter inside overlap
    unfold_windows = []  # audit: (win_lo, win_hi) per vertical group

    # One raw-point placement path shared by ordinary nonvertical
    # landmarks AND by width-gate-rejected apparent-vertical groups:
    # after a prior legitimate unfold the active synthetic frontier
    # still applies (overlap-exit crawl until raw x catches up, then
    # exact raw x resumes), and genuine backwards x remains an error.
    NOISE_X = UNFOLD_NOISE_X   # covers multi-step crawl overshoot;
                     # genuine reversals are stroke-width scale

    def _place_raw_point(idx):
        nonlocal frontier, eps_n
        raw = lam[idx, 0]
        if frontier is not None and raw <= frontier:
            # overlap-exit crawl: microscopic increasing steps just
            # above the frontier until raw x catches up
            eps_n += 1
            p[idx] = max(frontier + eps_n * EPS_X,
                         p[idx - 1] + UNFOLD_MONOTONE_PUSH)
            deform[idx] = "unfold-exit-overlap"
        else:
            if frontier is not None:
                pass     # raw x caught up: resume exact raw x
            frontier = None
            eps_n = 0
            if raw < p[idx - 1] - NOISE_X:
                raise RuntimeError(
                    "Phase 1 bug: genuine backwards x on a "
                    f"nonvertical section ({raw:.3f} after "
                    f"{p[idx - 1]:.3f})")
            p[idx] = max(raw, p[idx - 1] + UNFOLD_MONOTONE_PUSH)

    i = 1
    while i < len(p):
        dx = lam[i, 0] - lam[i - 1, 0]
        dy = abs(lam[i, 1] - lam[i - 1, 1])
        if abs(dx) < VERT * dy:
            j = i
            x_lo = x_hi = lam[i, 0]
            while j + 1 < len(p):
                dx2 = lam[j + 1, 0] - lam[j, 0]
                dy2 = abs(lam[j + 1, 1] - lam[j, 1])
                nx_lo = min(x_lo, lam[j + 1, 0])
                nx_hi = max(x_hi, lam[j + 1, 0])
                # a group is a PHYSICAL vertical stroke: its total
                # x-span must stay within the stroke-width tolerance.
                # Steep-but-slanted strokes (z diagonals) keep their
                # real x and stay monotone atoms instead.
                if abs(dx2) >= VERT * dy2 or \
                        nx_hi - nx_lo > VERTICAL_X_TOL:
                    break
                x_lo, x_hi = nx_lo, nx_hi
                j += 1
            wins = [local_run(lam[k, 0], lam[k, 1])
                    or (lam[k, 0] - STROKE_MIN_HALF,
                        lam[k, 0] + STROKE_MIN_HALF)
                    for k in range(i, j + 1)]
            lo_r, hi_r = min(wins, key=lambda w: w[1] - w[0])
            # vertical-regime gate: the containing row run must be
            # commensurate with the local stroke width; otherwise this
            # is raster staircase noise inside a wider structure and
            # raw x progression is preserved (no unfold, no frontier).
            r_local = 0.0
            for k in range(i, j + 1):
                rr_ = int(min(max(round(lam[k, 1] / step), 0),
                              geom.fill.shape[0] - 1))
                cc_ = int(min(max(round(lam[k, 0] / step), 0),
                              geom.fill.shape[1] - 1))
                r_local = max(r_local, float(radius[rr_, cc_]) * step)
            if hi_r - lo_r > \
                    UNFOLD_RUN_WIDTH_GAIN * 2.0 * max(r_local,
                                                      STROKE_MIN_HALF):
                # rejected: raster staircase noise inside a wider
                # structure. Keep raw arc x, but still run every point
                # of the group through the normal frontier/catch-up
                # placement so a prior legitimate unfold's active
                # frontier is honoured (crawl until raw catches up).
                for k in range(i, j + 1):
                    _place_raw_point(k)
                i = j + 1
                continue
            margin = min(CORRIDOR_MARGIN, max((hi_r - lo_r) * 0.15,
                                         UNFOLD_MIN_MARGIN))
            win_lo = max(lo_r + margin, p[i - 1])
            win_hi = max(hi_r - margin, win_lo + STROKE_MIN_HALF)
            n_v = j - i + 2
            for k in range(i, j + 1):
                frac = (k - i + 1) / n_v
                cand = win_lo + frac * (win_hi - win_lo)
                cand = min(max(cand, lo_r + UNFOLD_EDGE_INSET),
                           hi_r - UNFOLD_EDGE_INSET)
                p[k] = max(cand, p[k - 1] + UNFOLD_MONOTONE_PUSH)
                deform[k] = "vertical-unfold"
            unfold_windows.append((round(win_lo, 3), round(win_hi, 3)))
            frontier = p[j]
            eps_n = 0
            i = j + 1
        else:
            _place_raw_point(i)
            i += 1


    # ---- node bands: continuity-tracked fill intervals ---------------
    # Each node's band lives inside a filled run of ITS OWN constraint
    # column. Run identity is tracked continuously: a node picks the run
    # overlapping its predecessor's band (same physical stroke), only
    # falling back to the skeleton-point run at route start/junctions.
    # This prevents silent snaps between nearby branches (e.g. B bowls).
    lo_list, hi_list = [], []
    atom_of_landmark = []
    prev_band = None
    for i2 in range(len(p)):
        y_g = lam[i2, 1]
        row = int(min(max(round(y_g / step), 0), geom.fill.shape[0] - 1))
        scol = int(min(max(round(lam[i2, 0] / step), 0),
                       geom.fill.shape[1] - 1))
        ccol = int(min(max(round(p[i2] / step), 0),
                       geom.fill.shape[1] - 1))
        half = max(STROKE_RADIUS_GAIN * float(radius[row, scol]) * step,
                   STROKE_MIN_HALF)
        cand_lo, cand_hi = y_g - half, y_g + half

        runs_c = col_fill_runs(ccol)
        chosen = None
        if prev_band is not None:
            pb_lo, pb_hi = prev_band
            best_ov = 0.0
            for blo, bhi in runs_c:
                ov = min(pb_hi, bhi) - max(pb_lo, blo)
                if ov > best_ov:
                    best_ov = ov
                    chosen = (blo, bhi)
        if chosen is None:
            # skeleton-provenance: run at the skeleton column containing
            # (or nearest) the skeleton point
            best_d = None
            for blo, bhi in col_fill_runs(scol):
                d = max(blo - step - y_g, y_g - (bhi + step), 0.0)
                if d == 0.0:
                    chosen = (blo, bhi)
                    break
                if best_d is None or d < best_d:
                    best_d = d
                    chosen = (blo, bhi)
        if chosen is None:
            chosen = (y_g - STROKE_MIN_HALF, y_g + STROKE_MIN_HALF)

        blo, bhi = chosen
        lo_y = max(cand_lo, blo)
        hi_y = min(cand_hi, bhi)
        # honour the skeleton point whenever the stroke does
        if not (lo_y <= y_g <= hi_y) and blo - step <= y_g <= bhi + step:
            lo_y = min(lo_y, max(y_g, blo))
            hi_y = max(hi_y, min(y_g, bhi))
            lo_y = max(lo_y, blo)
            hi_y = min(hi_y, bhi)
        if hi_y - lo_y < MIN_CORRIDOR_WIDTH:
            mid_y = 0.5 * (lo_y + hi_y)
            lo_y, hi_y = mid_y - MIN_CORRIDOR_WIDTH / 2, \
                mid_y + MIN_CORRIDOR_WIDTH / 2
        lo_list.append(lo_y)
        hi_list.append(hi_y)
        atom_of_landmark.append(atom_at(float(targets[i2])))
        prev_band = (lo_y, hi_y)
    lo_arr = np.asarray(lo_list)
    hi_arr = np.asarray(hi_list)

    # drop leading/trailing nodes whose constraint column is empty
    # (unfold overshoot past a thin tip); interior nodes stay
    def _col_has(col_i):
        return any(True for _ in col_fill_runs(col_i))

    lo_keep = 0
    while lo_keep < len(p) - 2 and not _col_has(int(round(p[lo_keep] / step))):
        lo_keep += 1
    hi_keep = len(p) - 1
    while hi_keep > lo_keep + 1 and \
            not _col_has(int(round(p[hi_keep] / step))):
        hi_keep -= 1
    p, lower0, upper0 = p[lo_keep:hi_keep + 1], \
        lo_arr[lo_keep:hi_keep + 1].copy(), hi_arr[lo_keep:hi_keep + 1]

    # Merge stacked landmarks: vertical-unfold transitions collapse many
    # consecutive route positions into (nearly) one constraint column,
    # producing multi-valued bounds at a single x - a vertical wall no
    # single-valued polynomial can thread. The geometrically faithful
    # replacement keeps ONE single-valued band per raster column: chain
    # adjacent stacked bands, clip to the column fill, then re-sample
    # any chord whose interpolation crosses unfilled space through the
    # actual glyph fill runs.
    orig_to_merged = {}
    if len(p) > 2:
        merged_p, merged_lo, merged_hi = [], [], []
        node_ks = []      # per node: tuple of original landmark indices
        i_m = 0
        n0 = len(p)
        while i_m < n0:
            j_m = i_m + 1
            while j_m < n0 and abs(p[j_m] - p[i_m]) <= step:
                j_m += 1
            bands = sorted((lower0[k], upper0[k])
                           for k in range(i_m, j_m))
            chains = []
            for blo, bhi in bands:
                if chains and blo <= chains[-1][1] + step:
                    chains[-1][1] = max(chains[-1][1], bhi)
                    chains[-1][0] = min(chains[-1][0], blo)
                else:
                    chains.append([blo, bhi])
            best = max(chains, key=lambda c: c[1] - c[0])
            col_i = int(min(max(round(p[i_m] / step), 0),
                            geom.fill.shape[1] - 1))
            clipped = None
            for r_lo, r_hi in col_fill_runs(col_i):
                cb_lo = max(best[0], r_lo)
                cb_hi = min(best[1], r_hi)
                if cb_hi - cb_lo > MIN_CORRIDOR_WIDTH and (
                        clipped is None
                        or cb_hi - cb_lo > clipped[1] - clipped[0]):
                    clipped = [cb_lo, cb_hi]
            if clipped is not None:
                best = clipped
            else:
                best = list(max(bands, key=lambda c: c[1] - c[0]))
            merged_p.append(p[i_m])
            merged_lo.append(best[0])
            merged_hi.append(best[1])
            node_ks.append(tuple(range(i_m, j_m)))
            i_m = j_m

        # Repair pass: a merged stack can leave a long chord whose
        # linear interpolation crosses unfilled space (e.g. a diagonal
        # traversed while realized x crawled at one frontier column).
        # Insert fill-clipped intermediate nodes along such chords.
        i_r = 0
        while i_r < len(merged_p) - 1:
            ax, bx_ = merged_p[i_r], merged_p[i_r + 1]
            alo, ahi = merged_lo[i_r], merged_hi[i_r]
            blo, bhi_ = merged_lo[i_r + 1], merged_hi[i_r + 1]
            c_a = int(round(ax / step))
            c_b = int(round(bx_ / step))
            insertions = []
            for c in range(c_a + 1, c_b):
                xc = c * step
                t = (xc - ax) / (bx_ - ax)
                lo_t = alo + t * (blo - alo)
                hi_t = ahi + t * (bhi_ - ahi)
                rr = col_fill_runs(c)
                if any(r_lo - step <= lo_t and hi_t <= r_hi + step
                       for r_lo, r_hi in rr):
                    continue
                best_run, best_ov = None, MIN_CORRIDOR_WIDTH
                for r_lo, r_hi in rr:
                    ov = min(hi_t, r_hi) - max(lo_t, r_lo)
                    if ov > best_ov:
                        best_ov, best_run = ov, (r_lo, r_hi)
                if best_run is not None:
                    insertions.append(
                        (xc, max(lo_t, best_run[0]),
                         min(hi_t, best_run[1])))
            for off, (xc, nb_lo, nb_hi) in enumerate(insertions):
                merged_p.insert(i_r + 1 + off, xc)
                merged_lo.insert(i_r + 1 + off, nb_lo)
                merged_hi.insert(i_r + 1 + off, nb_hi)
                node_ks.insert(i_r + 1 + off, ())
            i_r += 1 + len(insertions)

        p = np.asarray(merged_p)
        lower0 = np.asarray(merged_lo)
        upper0 = np.asarray(merged_hi)
        for pos_, ks_ in enumerate(node_ks):
            for k_ in ks_:
                orig_to_merged[k_] = pos_
    else:
        for k_ in range(len(p)):
            orig_to_merged[k_] = k_

    lam_kept = lam[lo_keep:hi_keep + 1]

    heights = upper0 - lower0
    mm = np.minimum(CORRIDOR_MARGIN,
                    np.maximum(0.0, (heights - MIN_CORRIDOR_WIDTH)) / 2.0)
    lower = lower0 + mm
    upper = upper0 - mm

    center = (lower + upper) / 2.0
    path = BoundaryPath(points=np.column_stack([p, center]),
                        contour_id=-1)

    # realized embedding per physical atom (strict V6 / R1 fidelity),
    # restricted to the kept node range. Provenance per sample:
    # raw_x, raw_y, realized_x, center_y, lower, upper, deformation.
    realized = {}
    for i3 in range(lo_keep, hi_keep + 1):
        j_m = orig_to_merged[i3 - lo_keep]
        a_id = atom_of_landmark[i3]
        r = realized.setdefault(a_id, {
            "raw_x": [], "raw_y": [], "x": [], "y": [],
            "lower": [], "upper": [], "deform": []})
        r["raw_x"].append(lam[i3, 0])
        r["raw_y"].append(lam[i3, 1])
        r["x"].append(p[j_m])
        r["y"].append(center[j_m])
        r["lower"].append(lower[j_m])
        r["upper"].append(upper[j_m])
        r["deform"].append(deform[i3])
    realized = {k: {kk: np.asarray(vv) for kk, vv in v4.items()}
                for k, v4 in realized.items()}


    # The Chebyshev normalization domain is the corridor's REAL constraint
    # region (the slice-interval x-range `xs`), NOT a window padded wider
    # than it. Padding `xa/xb` beyond `xs` collapses the meaningful
    # interior into a tiny central slice of the z in [-1,1] while leaving
    # the escape tails in the same central region - the polynomial then
    # has almost no z-resolution across the actual stroke and swings
    # wildly (spanning far more vertical space than the local stroke
    # occupies). Keeping `xa/xb == xs` puts the stroke at z in [-1,1] and
    # the escape tails just outside it, so every curve preserves its local
    # vertical occupancy instead of being forced across the whole glyph.
    return Corridor(
        path=path,
        xa=float(p[0]),
        xb=float(p[-1]),
        xs=p,
        lower=lower,
        upper=upper,
        ylo=float(geom.ymin),
        yhi=float(geom.ymax),
        ylo_local=float(lower.min()),
        yhi_local=float(upper.max()),
        realized=realized,
        join_score=route_join_score(graph, route),
    )



def nonvertical_realization_x_error(realized: dict,
                                    raster_step: float = SIZE / GRID,
                                    tol: float = 1.0):
    """R1: max |realized_x - raw_x| over NONVERTICAL-atom samples that
    are not inside a declared unfold-exit overlap. Must be within tiny
    tolerance: vertical deformation may not propagate into other atoms."""
    worst = 0.0
    for a_id, r in realized.items():
        # only UNDEFORMED nonvertical samples count: 'unfold-exit-
        # overlap' and 'vertical-unfold' samples are declared synthetic
        mask = np.asarray(r["deform"]) == "none"
        if not mask.any():
            continue
        d = np.abs(np.asarray(r["x"])[mask]
                   - np.asarray(r["raw_x"])[mask])
        worst = max(worst, float(d.max()))
    return worst


def build_slice_corridor(graph: RouteGraph, edge_ids, geom) -> Corridor:
    if isinstance(edge_ids, Route):
        edge_ids = route_edge_ids(edge_ids)
    elif edge_ids and hasattr(edge_ids[0], "edge_id"):
        edge_ids = route_edge_ids(edge_ids)
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

    # Chebyshev domain == the slice-interval x-range (see
    # build_route_corridor for the rationale); do NOT pad past `xs`.
    return Corridor(
        path=path,
        xa=float(xs[0]),
        xb=float(xs[-1]),
        xs=xs,
        lower=lower,
        upper=upper,
        ylo=float(geom.ymin),
        yhi=float(geom.ymax),
        ylo_local=float(lower.min()),
        yhi_local=float(upper.max()),
    )




# ---------------------------------------------------------------------------
# Combined stroke/hole route graph (skeleton-derived)
# ---------------------------------------------------------------------------

STROKE_COVER_TOL = 0.10         # >= max corridor half-width (TAU +
                               # STROKE_RADIUS_GAIN * max radius): a
                               # poly inside its corridor can miss the
                               # skeleton by at most this much
VERTICAL_X_TOL = 0.01          # |dx_total| <= this => vertical atom
STROKE_LANDMARKS = 120        # corridor landmark samples per route


def _landmark_count(total_arc_length: float) -> int:
    """Landmark count for a route of given arc length.

    Scale-equivariant conversion of LENGTH to COUNT: approximately one
    landmark per REFERENCE_UNIT (= NORMALIZED_SIZE / 100), floor 8,
    cap STROKE_LANDMARKS. Mirrors the known-good 0..100 implementation,
    where a plain int(total) happened to be correct.
    """
    return min(STROKE_LANDMARKS,
               max(8, int(total_arc_length / LANDMARK_SPACING)))
STROKE_RADIUS_GAIN = 1.6     # corridor half-width = gain * stroke radius
STROKE_MIN_HALF = 0.008        # never narrower than this


def _monotone_pieces(pts: np.ndarray):
    """Split a polyline into x-monotone pieces at GENUINE local x
    extrema. Zero dx (locally vertical motion) continues the current
    direction: an extremum is where the SIGN OF NONZERO dx flips.
    Returns a list of (i0, i1) index ranges into pts."""
    n = len(pts)
    if n < 2:
        return []
    dx = np.diff(pts[:, 0])
    cuts = [0]
    last_sign = 0
    for i in range(len(dx)):
        sgn = 1 if dx[i] > 0 else (-1 if dx[i] < 0 else 0)
        if sgn == 0:
            continue
        if last_sign == 0:
            last_sign = sgn
        elif sgn != last_sign:
            # the extremum POINT is shared by both pieces. Raster
            # skeletons flatten smooth extrema into multi-point
            # plateaus; cutting at the first plateau point orphans the
            # plateau remainder as a pure-vertical sub-stroke fragment.
            # Cut at the MIDDLE of the zero-dx plateau instead so both
            # pieces leave the extremum symmetrically.
            j = i
            while j > 0 and dx[j - 1] == 0:
                j -= 1
            cuts.append((j + i) // 2)
            last_sign = sgn
    if cuts[-1] != n - 1:
        cuts.append(n - 1)
    return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)
            if cuts[k] < cuts[k + 1]]


def atom_x_monotonicity_violation(points: np.ndarray,
                                  tol: float = 1e-9) -> float:
    """Worst genuine backwards-x step of an oriented atom polyline.

    Vertical atoms (constant x) return 0. Nonvertical atoms must have
    dx >= -tol everywhere after left-to-right orientation; a real
    reversal is a Phase-1 bug, never repaired downstream.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    dx = np.diff(pts[:, 0])
    if abs(float(dx.sum())) <= VERTICAL_X_TOL and float(np.max(np.abs(dx))) \
            <= VERTICAL_X_TOL:
        return 0.0                    # genuinely vertical stroke
    worst = float(np.max(-dx)) if len(dx) else 0.0
    return max(0.0, worst - tol)
