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
    """Canonical rasterized normalized boundary point cloud."""
    tp = TextPath((0, 0), letter, size=100, prop=FontProperties(fname=_font_path()))
    polys = tp.to_polygons()
    pts = np.vstack(polys)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    scale = SIZE / max(mx[0] - mn[0], mx[1] - mn[1])

    verts = []
    codes = []
    for poly in polys:
        t = np.empty_like(poly)
        t[:, 0] = (poly[:, 0] - mn[0]) * scale
        t[:, 1] = (poly[:, 1] - mn[1]) * scale
        verts.append(t)
        codes.append([Path.MOVETO] + [Path.LINETO] * (len(t) - 1))
    path = Path(np.vstack(verts), np.concatenate(codes))

    step = SIZE / GRID
    axis = (np.arange(GRID) + 0.5) * step
    gx, gy = np.meshgrid(axis, axis)
    filled = path.contains_points(
        np.column_stack([gx.ravel(), gy.ravel()])
    ).reshape(GRID, GRID)

    f = np.pad(filled, 1, constant_values=False)
    interior_filled = (
        f[1:-1, 1:-1]
        & f[:-2, 1:-1]
        & f[2:, 1:-1]
        & f[1:-1, :-2]
        & f[1:-1, 2:]
    )
    boundary = filled & ~interior_filled
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
    """One maximal x-monotone boundary route (points sorted by x)."""

    points: np.ndarray
    contour_id: int
    covered: np.ndarray | None = None


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
    """Split a closed ordered loop into maximal x-monotone open chains.

    Edges are grouped by travel direction cyclically, so every outline
    edge — including the one crossing the rotation seam — belongs to
    exactly one chain. Chains moving in decreasing x are reversed so
    all paths come out sorted by increasing x.
    """
    pts = loop[:-1] if len(loop) > 1 and np.allclose(loop[0], loop[-1]) else loop
    n = len(pts)
    if n < 2:
        return []
    nxt = (np.arange(n) + 1) % n
    dx = pts[nxt, 0] - pts[:, 0]
    direction = np.where(dx > eps, 1, np.where(dx < -eps, -1, 0))

    chains: list[np.ndarray] = []
    # deterministic cycle start: first edge after the global x-min vertex
    begin = int(np.argmin(pts[:, 0]))
    i = 0
    while i < n:
        idx = (begin + i) % n
        d = direction[idx]
        i += 1
        if d == 0:
            continue
        run = [idx]
        while i < n:
            nxt_idx = (begin + i) % n
            if direction[nxt_idx] != d:
                break
            run.append(nxt_idx)
            i += 1
        last = (run[-1] + 1) % n
        chain = pts[run + [last]].copy()
        if chain[-1, 0] < chain[0, 0]:
            chain = chain[::-1].copy()
        if len(chain) >= 2:
            chains.append(chain)
    return chains


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
        for chain in _max_x_monotone_chains(contour):
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
                BoundaryPath(points=_resample(chain, nodes), contour_id=cid)
            )
    return paths


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
