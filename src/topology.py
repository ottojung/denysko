"""Phase 1: explicit boundary topology.

The glyph boundary is represented as ordered contours (the font's
flattened glyph outlines, normalized exactly like the canonical
rasterized boundary) and decomposed deterministically into maximal
x-monotone boundary paths. Topology is explicit data decided before any
polynomial fitting: a path is a route a graph y = f(x) could plausibly
follow, so x never reverses direction along it. Contours that turn back
in x are split into several paths; holes and counters produce their own
paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

GRID = 512
SIZE = 100.0


def _font_path() -> str:
    return os.path.join(
        matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"
    )


def _normalized_polygons(letter: str) -> list[np.ndarray]:
    """Flattened glyph outlines normalized like the canonical raster:
    bundled DejaVuSans at size 100, aspect preserved, filled-bbox
    lower-left mapped to (0, 0), max dimension 100, y-up."""
    tp = TextPath((0, 0), letter, size=100, prop=FontProperties(fname=_font_path()))
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


def glyph_boundary_cloud(letter: str) -> np.ndarray:
    """Canonical rasterized normalized boundary point cloud.

    Fill uses even-odd semantics across the glyph's rings (XOR of each
    ring's interior), so counters/holes contribute real boundary
    geometry instead of being silently filled by winding quirks.
    """
    tp = TextPath((0, 0), letter, size=100, prop=FontProperties(fname=_font_path()))
    polys = [np.asarray(p, dtype=float) for p in tp.to_polygons()]
    pts = np.vstack(polys)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    scale = SIZE / max(mx[0] - mn[0], mx[1] - mn[1])

    step = SIZE / GRID
    axis = (np.arange(GRID) + 0.5) * step
    gx, gy = np.meshgrid(axis, axis)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    # normalize grid through the same transform direction: grid is in
    # normalized space already, so rings are normalized first.
    rings = []
    for poly in polys:
        t = np.empty_like(poly)
        t[:, 0] = (poly[:, 0] - mn[0]) * scale
        t[:, 1] = (poly[:, 1] - mn[1]) * scale
        rings.append(t)

    inside = np.zeros(len(grid), dtype=bool)
    for ring in rings:
        rp = Path(ring)
        inside ^= rp.contains_points(grid)

    f = inside.reshape(GRID, GRID)
    fp = np.pad(f, 1, constant_values=False)
    interior_filled = (
        fp[1:-1, 1:-1]
        & fp[:-2, 1:-1]
        & fp[2:, 1:-1]
        & fp[1:-1, :-2]
        & fp[1:-1, 2:]
    )
    boundary = f & ~interior_filled
    iy, ix = np.nonzero(boundary)
    return np.column_stack([(ix + 0.5) * step, (iy + 0.5) * step])


@dataclass
class GlyphGeometry:
    letter: str
    contours: list[np.ndarray]
    points: np.ndarray
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def glyph_geometry(letter: str) -> GlyphGeometry:
    contours = _normalized_polygons(letter)
    points = glyph_boundary_cloud(letter)
    return GlyphGeometry(
        letter=letter,
        contours=contours,
        points=points,
        xmin=float(points[:, 0].min()),
        xmax=float(points[:, 0].max()),
        ymin=float(points[:, 1].min()),
        ymax=float(points[:, 1].max()),
    )


@dataclass
class BoundaryPath:
    """One maximal x-monotone boundary route (points sorted by x).

    source_edge_ids records which original contour edges (indices into
    the contour's edge cycle) this path represents, so the invariant
    "every non-degenerate contour edge belongs to exactly one path" is
    checkable. Vertical runs are represented as narrow monotone paths
    with VERTICAL_PATH_X_SPAN total x-width centered on the true edge.
    """

    points: np.ndarray
    contour_id: int
    source_edge_ids: tuple
    covered: np.ndarray | None = None


VERTICAL_PATH_X_SPAN = 0.5


def _resample(points: np.ndarray, target: int) -> np.ndarray:
    """Deterministic arc-length resampling of an open polyline."""
    seg = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return np.vstack([points[0], points[-1]])
    targets = np.linspace(0.0, total, target)
    out = np.empty((target, 2))
    j = 0
    for i, tval in enumerate(targets):
        while j < len(s) - 2 and s[j + 1] < tval:
            j += 1
        span = s[j + 1] - s[j]
        frac = 0.0 if span <= 0 else (tval - s[j]) / span
        out[i] = points[j] + frac * (points[j + 1] - points[j])
    return out


def _max_x_monotone_chains(loop: np.ndarray, eps: float = 1e-9):
    """Split a closed ordered loop into maximal x-monotone chains.

    Returns a list of (points, source_edge_ids). Edge classification is
    cyclic over all N edges; horizontal runs group same-direction edges,
    and maximal runs of exactly-vertical edges become narrow monotone
    paths instead of being discarded. Only zero-length edges (dx == dy
    == 0) are degenerate and dropped, and that rule is documented here.

    Chains moving in decreasing x are reversed so every chain is sorted
    by increasing x.
    """
    pts = loop[:-1] if len(loop) > 1 and np.allclose(loop[0], loop[-1]) else loop
    n = len(pts)
    if n < 2:
        return []
    nxt = (np.arange(n) + 1) % n
    dx = pts[nxt, 0] - pts[:, 0]
    dy = pts[nxt, 1] - pts[:, 1]
    direction = np.where(dx > eps, 1, np.where(dx < -eps, -1, 0))

    def finish(points, edge_ids):
        points = points.copy()
        if points[-1, 0] < points[0, 0]:
            points = points[::-1].copy()
            edge_ids = edge_ids[::-1]
        return (points, tuple(edge_ids))

    out = []
    begin = int(np.argmin(pts[:, 0]))
    i = 0
    while i < n:
        idx = (begin + i) % n
        d = direction[idx]
        i += 1
        if dx[idx] == 0.0 and abs(dy[idx]) <= eps:
            continue  # degenerate zero-length edge (documented rule)
        if d == 0:
            # maximal vertical run: consume consecutive vertical edges
            run = [idx]
            vpts = [pts[idx], pts[(idx + 1) % n]]
            while i < n:
                nxt_idx = (begin + i) % n
                if direction[nxt_idx] != 0 or not (
                    abs(dx[nxt_idx]) <= eps
                ):
                    break
                run.append(nxt_idx)
                vpts.append(pts[(nxt_idx + 1) % n])
                i += 1
            vp = np.asarray(vpts, dtype=float)
            # graph-compatible narrow sweep across the full vertical run
            e = VERTICAL_PATH_X_SPAN / 2.0
            x0 = float(np.mean(vp[:, 0]))
            narrow = np.asarray([
                [x0 - e, float(vp[:, 1].max())],
                [x0, float(vp[:, 1].mean())],
                [x0 + e, float(vp[:, 1].min())],
            ], dtype=float)
            out.append(finish(narrow, run))
            continue
        run = [idx]
        chain_pts = [pts[idx], pts[(idx + 1) % n]]
        while i < n:
            nxt_idx = (begin + i) % n
            if direction[nxt_idx] != d:
                break
            run.append(nxt_idx)
            chain_pts.append(pts[(nxt_idx + 1) % n])
            i += 1
        out.append(finish(np.asarray(chain_pts, dtype=float), run))
    return out


def extract_paths(
    contours: list[np.ndarray],
    *,
    min_x_span: float = 1.0,
    min_y_span: float = 2.0,
    min_points: int = 2,
    resample_cap: int = 60,
) -> list[BoundaryPath]:
    """Deterministically decompose ordered contours into maximal
    x-monotone paths.

    Near-vertical geometry is kept as a narrow-span steep path rather
    than rejected; only degenerate slivers that are narrow in both x and
    y are dropped. Font outlines of straight-sided glyphs can be as
    coarse as two vertices per side, so short chains are valid paths.
    """
    paths: list[BoundaryPath] = []
    for cid, contour in enumerate(contours):
        for chain, edge_ids in _max_x_monotone_chains(contour):
            if len(chain) < min_points:
                continue
            xspan = chain[-1, 0] - chain[0, 0]
            yspan = float(chain[:, 1].max() - chain[:, 1].min())
            if xspan < min_x_span and yspan < min_y_span:
                continue
            arc = float(
                np.hypot(*np.diff(chain, axis=0).T).sum()
            )
            nodes = int(min(resample_cap, max(2 * min_points, arc / 2.0)))
            nodes = max(nodes, min_points)
            paths.append(
                BoundaryPath(
                    points=_resample(chain, nodes),
                    contour_id=cid,
                    source_edge_ids=edge_ids,
                )
            )
    return paths


def contour_edge_count(contour: np.ndarray) -> int:
    pts = (
        contour[:-1]
        if len(contour) > 1 and np.allclose(contour[0], contour[-1])
        else contour
    )
    return len(pts)


def min_dists(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    da = np.full(len(a), np.inf)
    db = np.full(len(b), np.inf)
    b2 = (b * b).sum(axis=1)
    block = max(1, int(2_000_000 / max(1, len(b))))
    for i in range(0, len(a), block):
        ai = a[i : i + block]
        d2 = (ai * ai).sum(axis=1)[:, None] - 2.0 * (ai @ b.T) + b2[None, :]
        np.maximum(d2, 0.0, out=d2)
        np.minimum(da[i : i + block], np.sqrt(d2.min(axis=1)), out=da[i : i + block])
        np.minimum(db, np.sqrt(d2.min(axis=0)), out=db)
    return da, db


def assign_coverage(paths, cloud: np.ndarray, tau: float):
    """Mark which boundary-cloud samples each path represents."""
    masks = []
    for p in paths:
        d, _ = min_dists(cloud, p.points)
        masks.append(d <= tau)
    return masks


def dedupe_paths(paths, masks, jaccard: float = 0.85):
    """Drop later paths whose covered set nearly duplicates an earlier one."""
    keep_p, keep_m = [], []
    for p, m in zip(paths, masks):
        dup = False
        for km in keep_m:
            inter = np.logical_and(m, km).sum()
            union = np.logical_or(m, km).sum()
            if union and inter / union > jaccard:
                dup = True
                break
        if not dup:
            keep_p.append(p)
            keep_m.append(m)
    for p, m in zip(keep_p, keep_m):
        p.covered = m
    return keep_p, keep_m


# ---------------------------------------------------------------------------
# Corridors
# ---------------------------------------------------------------------------

TAU = 2.0
MIN_COVERAGE = 0.95
DEFAULT_MAX_CURVES = 12

MIN_CORRIDOR_WIDTH = 0.4
ESC_OFFSETS = (1.0, 2.0, 3.5, 5.5, 8.0)
ESCAPE_RATE = 2.5          # band-clearance growth per unit x (band exits)
BAND_EDGE_TOL = 1.5        # endpoint this close to a glyph x-edge is "at" it
FAR_ROWS = (3.0, 8.0)      # beyond-window distances for non-return rows
FAR_CLEARANCE = 3.0


@dataclass
class EscapeSpec:
    """Outward escape constraints for one end of a path.

    kind='band': the endpoint lies inside the glyph's x-span, so the
    tail must leave the vertical band: at offsets ESC_OFFSETS past the
    end, P must clear the nearer band edge with ESC_RATE growth
    (inequalities, not exact targets). kind='far': the endpoint sits at
    a glyph x-edge, so the tail immediately leaves the drawn region;
    only far-field rows forbid swinging back into the band later.
    """

    kind: str            # 'band' or 'far'
    side: str            # 'L' or 'R'
    sigma: int           # +1 upward exit, -1 downward exit
    x_end: float
    edge: float          # band edge being exited (ymax up / ymin down)
    y_end: float         # path endpoint y (ramp anchor)
    rows: list           # [(x, lo, hi)] absolute one-sided rows


@dataclass
class Corridor:
    """Allowed region for one path's polynomial.

    Interior: piecewise-linear [lower(x), upper(x)] around the path.
    Escapes: explicit one-sided rows (band exits or far-field
    non-return). Topology is fixed by the corridor itself.
    """

    path: BoundaryPath
    xa: float
    xb: float
    xs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    escapes: tuple       # (EscapeSpec left, EscapeSpec right)

    def lower_at(self, x):
        return np.interp(x, self.xs, self.lower)

    def upper_at(self, x):
        return np.interp(x, self.xs, self.upper)


def _sigma_band(y_end: float, geom: GlyphGeometry) -> int:
    """Band-exit direction toward the nearer band edge (ties go up)."""
    return 1 if (geom.ymax - y_end) <= (y_end - geom.ymin) else -1


def _make_escape(endpoint: np.ndarray, side: str, geom: GlyphGeometry) -> EscapeSpec:
    x_end = float(endpoint[0])
    y_end = float(endpoint[1])
    # Band escapes always exit toward the NEARER band edge and anchor the
    # ramp at the endpoint itself, so the required slope stays at
    # ESCAPE_RATE regardless of how deep inside the band the route sits.
    # (A mass-based direction here can pair an upward climb with the
    # downward edge distance, producing a 50-unit instant cliff.)
    at_left_edge = (x_end - geom.xmin) <= BAND_EDGE_TOL
    at_right_edge = (geom.xmax - x_end) <= BAND_EDGE_TOL
    if (side == "L" and at_left_edge) or (side == "R" and at_right_edge):
        # tail immediately leaves the drawn x-region; forbid returning
        sigma = _sigma_band(y_end, geom)
        edge = geom.ymax if sigma == 1 else geom.ymin
        sign = -1.0 if side == "L" else 1.0
        base = x_end + sign * 2.0
        rows = []
        for d in FAR_ROWS:
            x = base + sign * d
            lo = geom.ymax + FAR_CLEARANCE if sigma == 1 else -np.inf
            hi = geom.ymin - FAR_CLEARANCE if sigma == -1 else np.inf
            rows.append((x, lo, hi))
        return EscapeSpec("far", side, sigma, x_end, edge, y_end, rows)

    sigma = _sigma_band(y_end, geom)
    edge = geom.ymax if sigma == 1 else geom.ymin
    sign = -1.0 if side == "L" else 1.0
    # Ramp anchored at the path endpoint (continuous with the fitted
    # route), moving outward at ESCATE_RATE; endpoints deep inside the
    # band get a proportionally longer run so the slope stays gentle.
    edge_dist = abs(edge - y_end)
    run = max(ESC_OFFSETS[-1], edge_dist / ESCAPE_RATE + 1.0)
    # linear ramp from the endpoint down/up to the final clearance:
    # constant slope, no kink at the joint, no instant cliff.
    slope = (edge_dist + ESCAPE_RATE * run) / run
    offs = np.linspace(0.0, run, len(ESC_OFFSETS) + 1)
    rows = []
    for off in offs:
        x = x_end + sign * off
        level = y_end + sigma * slope * off
        lo = level if sigma == 1 else -np.inf
        hi = level if sigma == -1 else np.inf
        rows.append((x, lo, hi))
    return EscapeSpec("band", side, sigma, x_end, edge, y_end, rows)


def build_corridors(paths, geom: GlyphGeometry) -> list[Corridor]:
    """Construct the allowed region around every path.

    Interior width hugs surface tolerance but shrinks near competing
    geometry (half the distance to the nearest boundary sample NOT
    covered by this path, floored), so corridors never merge distinct
    strokes. Each end gets an EscapeSpec: a band exit with growing
    clearance when the endpoint is interior to the glyph's x-span, or
    far-field non-return rows when the endpoint already sits on a glyph
    x-edge (its tail leaves the drawn region immediately).
    """
    out = []
    for p in paths:
        own = p.covered if p.covered is not None else np.zeros(
            len(geom.points), bool
        )
        foreign = geom.points[~own]
        widths = []
        for pt in p.points:
            if len(foreign):
                d = float(np.hypot(
                    foreign[:, 0] - pt[0], foreign[:, 1] - pt[1]
                ).min())
            else:
                d = TAU
            widths.append(min(TAU, max(MIN_CORRIDOR_WIDTH, 0.5 * d)))
        widths = np.asarray(widths)
        xs = p.points[:, 0]
        esc_l = _make_escape(p.points[0], "L", geom)
        esc_r = _make_escape(p.points[-1], "R", geom)
        pad_l = ESC_OFFSETS[-1] if esc_l.kind == "band" else 2.0
        pad_r = ESC_OFFSETS[-1] if esc_r.kind == "band" else 2.0
        out.append(
            Corridor(
                path=p,
                xa=float(xs[0] - pad_l),
                xb=float(xs[-1] + pad_r),
                xs=xs,
                lower=p.points[:, 1] - widths,
                upper=p.points[:, 1] + widths,
                escapes=(esc_l, esc_r),
            )
        )
    return out


def select_paths(
    corridors: list[Corridor],
    *,
    coverage_target: float = MIN_COVERAGE,
    max_paths: int = DEFAULT_MAX_CURVES,
):
    """Deterministic greedy set cover over path coverage masks.

    Repeatedly take the corridor covering the most currently-uncovered
    boundary samples. Ties break by longer path, then by stable index.
    Returns (selected_corridors, covered_mask). Selection is decided
    entirely in Phase 1 - no polynomial coefficients are consulted.
    """
    n = len(corridors[0].path.covered)
    covered = np.zeros(n, dtype=bool)
    remaining = list(range(len(corridors)))
    selected = []
    while len(selected) < max_paths:
        best = None
        best_key = None
        for idx in remaining:
            mask = corridors[idx].path.covered
            new = int((mask & ~covered).sum())
            length = float(
                corridors[idx].path.points[-1, 0]
                - corridors[idx].path.points[0, 0]
            )
            key = (new, length, -idx)
            if new == 0:
                continue
            if best_key is None or key > best_key:
                best_key, best = key, idx
        if best is None:
            break
        covered |= corridors[best].path.covered
        selected.append(corridors[best])
        remaining.remove(best)
        if covered.mean() >= coverage_target:
            break
    return selected, covered
