#!/usr/bin/env python3
"""
Path processing utilities for rasterization and component analysis.
"""

import numpy as np


def rasterize_path(path, resolution=400):
    """
    Rasterize a Path to a binary mask with given resolution.
    Returns (mask, x_grid, y_grid).
    """
    vertices = path.vertices
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    
    # Avoid zero-size
    if max_x <= min_x:
        max_x = min_x + 1.0
    if max_y <= min_y:
        max_y = min_y + 1.0

    # Keep aspect by basing resolution on max dimension
    width = max_x - min_x
    height = max_y - min_y
    base = float(max(width, height))
    
    # Scale resolution to size (cap between 200 and 600)
    res = int(np.clip((resolution * base / max(base, 1e-6)), 200, 600))
    
    # Create grid
    xs = np.linspace(min_x, max_x, res)
    ys = np.linspace(min_y, max_y, res)
    x_grid, y_grid = np.meshgrid(xs, ys)
    pts = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)

    inside = path.contains_points(pts)
    mask = inside.reshape(res, res)
    return mask, x_grid, y_grid


def decompose_into_h_monotonic_components(mask):
    """
    Decompose binary mask into horizontal-monotonic components.
    For filled letters, we detect "waist" regions where the letter narrows,
    then split components at those points to get separate stroke components.
    """
    h, w = mask.shape
    if w == 0:
        return []

    # Step 1: Analyze the shape's width profile to detect narrowing
    width_profile = []
    column_intervals = []

    for c in range(w):
        intervals = []
        in_interval = False
        start = None

        for r in range(h):
            if mask[r, c] and not in_interval:
                start = r
                in_interval = True
            elif not mask[r, c] and in_interval:
                intervals.append((start, r - 1))
                in_interval = False

        if in_interval:
            intervals.append((start, h - 1))

        column_intervals.append(intervals)

        # Calculate total width (height) for this column
        total_width = sum(end - start + 1 for start, end in intervals)
        width_profile.append(total_width)

    if not any(width_profile):
        return []

    # Step 2: Detect waist points (local minima in width profile)
    waist_points = _find_waist_points(width_profile)

    print(f"Shape analysis: width range {min(width_profile)}-{max(width_profile)}, waists at columns: {waist_points}")

    # Step 3: If no clear waists, treat as single component
    if not waist_points:
        # Single component approach
        components = []
        first_col = next((c for c in range(w) if column_intervals[c]), None)
        if first_col is not None:
            component = {}
            for c in range(first_col, w):
                if column_intervals[c]:
                    component[c] = column_intervals[c]
            if len(component) >= 3:
                components.append(component)
        return components

    # Step 4: Split into components based on waist points
    components = []
    split_points = [0] + waist_points + [w]

    for i in range(len(split_points) - 1):
        start_col = split_points[i]
        end_col = split_points[i + 1]

        component = {}
        for c in range(start_col, end_col):
            if c < len(column_intervals) and column_intervals[c]:
                component[c] = column_intervals[c]

        # Only keep components that span reasonable width
        if len(component) >= max(3, (end_col - start_col) * 0.3):
            components.append(component)
            print(f"Component {len(components)}: columns {start_col}-{end_col}, width {len(component)}")

    return components


def _find_waist_points(width_profile, min_prominence=0.1):
    """
    Find waist points (local minima) in the width profile.
    These indicate where the letter narrows and might split into components.
    """
    if len(width_profile) < 5:
        return []

    # Smooth the profile to avoid noise
    smoothed = _smooth_1d(width_profile, window=5)

    max_width = max(smoothed)
    if max_width == 0:
        return []

    # Find local minima that are significant
    waists = []
    for i in range(2, len(smoothed) - 2):
        current = smoothed[i]
        if current == 0:  # Skip empty columns
            continue

        # Check if this is a local minimum
        if current < smoothed[i - 1] and current < smoothed[i + 1]:
            # Check prominence: how much narrower is it than nearby maxima?
            left_range = smoothed[max(0, i - 15) : i]
            right_range = smoothed[i + 1 : min(len(smoothed), i + 16)]

            left_max = max(left_range) if left_range else current
            right_max = max(right_range) if right_range else current
            local_max = max(left_max, right_max)

            if local_max > current:
                prominence = (local_max - current) / local_max
                print(f"  Potential waist at {i}: width={current}, local_max={local_max}, prominence={prominence:.2f}")
                if prominence >= min_prominence:
                    waists.append(i)

    # Remove waists that are too close together
    filtered_waists = []
    min_separation = len(smoothed) // 8  # At least 12.5% of width apart

    for waist in waists:
        if not filtered_waists or (waist - filtered_waists[-1]) >= min_separation:
            filtered_waists.append(waist)

    return filtered_waists


def _smooth_1d(data, window=5):
    """Simple 1D smoothing with moving average."""
    if len(data) < window:
        return data

    smoothed = []
    half_win = window // 2

    for i in range(len(data)):
        start = max(0, i - half_win)
        end = min(len(data), i + half_win + 1)
        smoothed.append(sum(data[start:end]) / (end - start))

    return smoothed


def interval_overlap(int1, int2):
    """Calculate fractional overlap between two y-intervals."""
    start1, end1 = int1
    start2, end2 = int2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_end < overlap_start:
        return 0.0

    overlap_len = overlap_end - overlap_start + 1
    total_len = min(end1 - start1 + 1, end2 - start2 + 1)

    return overlap_len / max(total_len, 1)
