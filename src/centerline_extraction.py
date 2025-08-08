#!/usr/bin/env python3
"""
Centerline extraction using horizontal-monotonic component decomposition.
"""

import numpy as np
from .path_processing import rasterize_path, decompose_into_h_monotonic_components


def extract_skeleton_from_path(path):
    """
    Extract centerlines using horizontal-monotonic component decomposition:
    1. Decompose letter into components where each x-position is covered once per component
    2. Generate random left-to-right walks through each component
    3. Average walks to get centerlines
    4. Return multiple continuous paths (one per component)
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

    # Generate centerlines for each component
    all_centerlines = []
    for component in components:
        centerline = _generate_component_centerline(component, x_grid, y_grid)
        if centerline is not None and len(centerline) >= 3:
            # Verify that centerline points are within the letter body
            if _verify_centerline_within_letter(centerline, path):
                all_centerlines.append(centerline)

    if not all_centerlines:
        return _create_simple_stroke_approximation(path)

    # For backward compatibility, return the longest centerline
    # TODO: Later modify interface to return all components as separate paths
    longest = max(all_centerlines, key=len)

    # Log information about components found
    print(f"Found {len(components)} components, {len(all_centerlines)} valid centerlines")
    if len(all_centerlines) > 1:
        lengths = [len(c) for c in all_centerlines]
        print(f"Centerline lengths: {lengths} (returning longest: {len(longest)})")

    return longest


def _verify_centerline_within_letter(centerline, path, tolerance=0.1):
    """
    Verify that centerline points are within the letter body (with small tolerance).

    Args:
        centerline: Array of (x,y) points
        path: Original letter path
        tolerance: Fraction of points that can be outside (for numerical errors)

    Returns:
        bool: True if centerline is mostly within the letter
    """
    if len(centerline) == 0:
        return False

    # Check how many points are inside the path
    inside_count = 0
    for point in centerline:
        if path.contains_point(point):
            inside_count += 1

    inside_fraction = inside_count / len(centerline)
    is_valid = inside_fraction >= (1.0 - tolerance)

    if not is_valid:
        print(f"Warning: Only {inside_fraction:.1%} of centerline points are within letter body")

    return is_valid


def _generate_component_centerline(component, x_grid, y_grid, num_walks=50):
    """
    Generate centerline for a single component using random walks.

    Args:
        component: Dict mapping column indices to list of y-intervals
        x_grid, y_grid: Coordinate grids from rasterization
        num_walks: Number of random walks to average

    Returns:
        np.array: Centerline points in original coordinates
    """
    if not component:
        return None

    # Get sorted column indices for left-to-right traversal
    columns = sorted(component.keys())
    if len(columns) < 2:
        return None

    # Generate multiple random walks
    walks = []
    for _ in range(num_walks):
        walk = _generate_random_walk(component, columns)
        if walk and len(walk) >= 3:
            walks.append(walk)

    if not walks:
        return None

    # Average the walks to get centerline
    # All walks should have same length (one point per column)
    walk_length = len(walks[0])
    averaged_walk = []

    for i in range(walk_length):
        # Average y-coordinates at this column position
        y_sum = sum(walk[i][1] for walk in walks)
        y_avg = y_sum / len(walks)
        col_idx = walks[0][i][0]  # Column index should be same for all walks
        averaged_walk.append((col_idx, y_avg))

    # Convert pixel coordinates back to original space
    centerline_points = []
    for col_idx, row_avg in averaged_walk:
        # Clamp row_avg to valid range
        h = y_grid.shape[0]
        row_idx = int(np.clip(row_avg, 0, h - 1))

        x = x_grid[0, col_idx]
        y = y_grid[row_idx, 0]
        centerline_points.append([x, y])

    centerline = np.array(centerline_points)

    # Additional verification: ensure all points are reasonable
    if len(centerline) > 5:
        # Remove any obvious outliers
        x_min, x_max = centerline[:, 0].min(), centerline[:, 0].max()
        y_min, y_max = centerline[:, 1].min(), centerline[:, 1].max()

        # Filter out points that are way outside reasonable bounds
        valid_mask = (
            (centerline[:, 0] >= x_min - (x_max - x_min) * 0.1)
            & (centerline[:, 0] <= x_max + (x_max - x_min) * 0.1)
            & (centerline[:, 1] >= y_min - (y_max - y_min) * 0.1)
            & (centerline[:, 1] <= y_max + (y_max - y_min) * 0.1)
        )

        if valid_mask.sum() >= 3:
            centerline = centerline[valid_mask]

    # Smooth the centerline gently
    if len(centerline) >= 5:
        centerline = _smooth_polyline(centerline, window=5)
    centerline = _dedupe_close_points(centerline, tol=1e-3)

    return centerline


def _generate_random_walk(component, columns):
    """
    Generate a single random left-to-right walk through the component.
    Keep walks STRICTLY within the intervals to ensure they stay in the letter body.

    Returns:
        List of (column_idx, row_pos) tuples
    """
    walk = []

    # Start from leftmost column - pick random y-position in first interval
    first_col = columns[0]
    first_intervals = component[first_col]

    # Pick random interval and random position within it
    interval_idx = np.random.randint(0, len(first_intervals))
    start_row, end_row = first_intervals[interval_idx]

    # Stay well within the interval bounds
    margin = max(1, (end_row - start_row) * 0.1)  # 10% margin
    safe_start = start_row + margin
    safe_end = end_row - margin

    if safe_start >= safe_end:
        current_y = (start_row + end_row) / 2.0
    else:
        current_y = np.random.uniform(safe_start, safe_end)

    walk.append((first_col, current_y))

    # Continue walk through remaining columns
    for col in columns[1:]:
        intervals = component[col]

        # Find best interval based on current y position
        best_interval = None
        best_distance = float("inf")

        for start_row, end_row in intervals:
            # Distance from current y to interval center
            interval_center = (start_row + end_row) / 2
            distance = abs(current_y - interval_center)

            if distance < best_distance:
                best_distance = distance
                best_interval = (start_row, end_row)

        if best_interval is None:
            # This shouldn't happen, but fallback to first interval
            best_interval = intervals[0]

        start_row, end_row = best_interval
        interval_size = end_row - start_row + 1

        # Conservative approach: small brownian motion within interval
        # Bias heavily toward staying near current y position
        if interval_size <= 3:
            # Very thin interval, just use center
            next_y = (start_row + end_row) / 2.0
        else:
            # Add small amount of controlled randomness
            target_y = current_y  # Start from current position
            noise_scale = min(interval_size * 0.15, 3.0)  # Limit noise
            noise = np.random.normal(0, noise_scale)
            next_y = target_y + noise

            # Enforce strict bounds with margins
            margin = max(0.5, interval_size * 0.05)
            safe_min = start_row + margin
            safe_max = end_row - margin

            if safe_min >= safe_max:
                next_y = (start_row + end_row) / 2.0
            else:
                next_y = np.clip(next_y, safe_min, safe_max)

        walk.append((col, next_y))
        current_y = next_y

    return walk


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
