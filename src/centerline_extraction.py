#!/usr/bin/env python3
"""
Centerline extraction using deterministic horizontal scanline midpoints.
"""

import numpy as np
from .path_processing import rasterize_path, decompose_into_h_monotonic_components


def extract_skeleton_from_path(path):
    """
    Extract centerlines by:
    1) Rasterizing the glyph to a binary mask
    2) Decomposing into horizontal-monotonic components (via width-profile waists)
    3) For each component and each column, choosing the widest inside-interval and
       taking its vertical midpoint as the centerline sample
    4) Returning the longest centerline (for backward compatibility)
    """
    vertices = path.vertices
    if len(vertices) < 6:
        return vertices

    # Rasterize the path to work with pixels
    mask, x_grid, y_grid = rasterize_path(path, resolution=400)
    if mask.sum() == 0:
        return vertices

    # Decompose into horizontal-monotonic components
    components = decompose_into_h_monotonic_components(mask)
    if not components:
        return _create_simple_stroke_approximation(path)

    all_centerlines = []
    for comp in components:
        cl = _component_midpoint_centerline(comp, x_grid, y_grid)
        if cl is None or len(cl) < 3:
            continue
        # Verify points are inside the letter body (tolerate small numeric error)
        if _verify_centerline_within_letter(cl, path):
            all_centerlines.append(cl)

    if not all_centerlines:
        return _create_simple_stroke_approximation(path)

    # Return the longest to keep current call sites stable
    return max(all_centerlines, key=len)


def _component_midpoint_centerline(component, x_grid, y_grid):
    """Build a deterministic centerline by taking midpoints of the widest interval per column."""
    cols = sorted(component.keys())
    if len(cols) < 2:
        return None

    pts = []
    for c in cols:
        intervals = component[c]
        if not intervals:
            continue
        # Pick widest interval to ensure stability
        lengths = [(end - start + 1, start, end) for (start, end) in intervals]
        _, s, e = max(lengths, key=lambda t: t[0])
        row = 0.5 * (s + e)
        h = y_grid.shape[0]
        row_idx = int(np.clip(row, 0, h - 1))
        x = x_grid[0, c]
        y = y_grid[row_idx, 0]
        pts.append([x, y])

    if len(pts) < 3:
        return None

    cl = np.array(pts)
    # Light smoothing and de-duplication
    if len(cl) >= 5:
        cl = _smooth_polyline(cl, window=5)
    cl = _dedupe_close_points(cl, tol=1e-3)
    return cl


def _verify_centerline_within_letter(centerline, path, tolerance=0.05):
    """True if most samples lie inside the glyph body."""
    if len(centerline) == 0:
        return False
    inside = sum(1 for p in centerline if path.contains_point(p))
    frac = inside / len(centerline)
    return frac >= (1.0 - tolerance)


def _smooth_polyline(pts, window=7):
    """Smooth a polyline using moving average."""
    if len(pts) < 3 or window < 3:
        return pts
    w = window if window % 2 == 1 else window + 1
    k = w // 2
    pad = np.vstack([pts[0:1].repeat(k, axis=0), pts, pts[-1:].repeat(k, axis=0)])
    sm = []
    for i in range(k, k + len(pts)):
        sm.append(pad[i - k : i + k + 1].mean(axis=0))
    return np.array(sm)


def _dedupe_close_points(pts, tol=1e-3):
    """Remove consecutive points that are too close."""
    if len(pts) <= 1:
        return pts
    out = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(p)
    return np.array(out)


def _create_simple_stroke_approximation(path):
    """Fallback method: Create a simple approximation when geometric detection fails."""
    vertices = path.vertices
    if len(vertices) < 6:
        return vertices
    # Use every 10th vertex to create a simplified representation
    step = max(1, len(vertices) // 10)
    simplified = vertices[::step]
    # Ensure we have at least a few points
    if len(simplified) < 3:
        simplified = vertices[[0, len(vertices) // 2, -1]]
    return simplified


def upsample_centerline(points, target_count):
    """Upsample centerline to target_count via arc-length interpolation."""
    if len(points) < 2:
        return points
    seg = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    d = np.insert(np.cumsum(seg), 0, 0.0)
    total = d[-1]
    if total == 0:
        return points
    targets = np.linspace(0, total, target_count)
    out = []
    j = 1
    for t in targets:
        while j < len(d) and d[j] < t:
            j += 1
        if j == 0:
            out.append(points[0])
        elif j >= len(d):
            out.append(points[-1])
        else:
            t0, t1 = d[j - 1], d[j]
            if t1 == t0:
                out.append(points[j - 1])
            else:
                a = (t - t0) / (t1 - t0)
                out.append(points[j - 1] + a * (points[j] - points[j - 1]))
    return np.array(out)
