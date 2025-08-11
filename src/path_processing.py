#!/usr/bin/env python3
"""
Path processing utilities for rasterization and component analysis.
"""

import numpy as np


def rasterize_path(path, resolution=400):
    """
    Rasterize a Path to a binary mask with given resolution.
    Properly handles holes in the path using matplotlib's PathPatch rendering.
    Returns (mask, x_grid, y_grid).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
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
    
    # Create coordinate grids
    xs = np.linspace(min_x, max_x, res)
    ys = np.linspace(min_y, max_y, res)
    x_grid, y_grid = np.meshgrid(xs, ys)
    
    # Use matplotlib's rendering to properly handle holes
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Create a PathPatch that will respect holes in the path
    patch = patches.PathPatch(path, facecolor='white', edgecolor='none')
    ax.add_patch(patch)
    
    # Set the exact bounds we want
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Render to a bitmap at the desired resolution
    fig.patch.set_facecolor('black')  # Background = False
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Get the rendered image
    buf = canvas.buffer_rgba()
    img = np.asarray(buf).copy()
    
    # Convert to binary mask (white areas = True, black areas = False)
    # Use the alpha channel or brightness to determine inside/outside
    mask = img[:, :, 0] > 128  # White pixels = inside the path
    
    # Resize to exactly the resolution we want using simple interpolation
    if mask.shape != (res, res):
        # Simple nearest neighbor resize
        old_h, old_w = mask.shape
        new_mask = np.zeros((res, res), dtype=bool)
        for i in range(res):
            for j in range(res):
                old_i = int(i * old_h / res)
                old_j = int(j * old_w / res)
                old_i = min(old_i, old_h - 1)
                old_j = min(old_j, old_w - 1)
                new_mask[i, j] = mask[old_i, old_j]
        mask = new_mask
    
    plt.close(fig)
    
    # Flip vertically to match coordinate system (matplotlib renders top-down, we want bottom-up)
    mask = np.flipud(mask)
    
    # Also flip the y_grid to match the flipped mask
    y_grid = np.flipud(y_grid)
    
    return mask, x_grid, y_grid


def column_intervals(mask):
    """Compute list of intervals [(start_row, end_row), ...] for each column in mask."""
    h, w = mask.shape
    all_cols = []
    for c in range(w):
        intervals = []
        in_interval = False
        start = 0
        for r in range(h):
            if mask[r, c] and not in_interval:
                start = r
                in_interval = True
            elif not mask[r, c] and in_interval:
                intervals.append((start, r - 1))
                in_interval = False
        if in_interval:
            intervals.append((start, h - 1))
        all_cols.append(intervals)
    return all_cols


def decompose_into_h_monotonic_components(mask):
    """
    Decompose binary mask into horizontal-monotonic components.
    For filled letters, detect local-width minima ("waists") and split.
    """
    h, w = mask.shape
    if w == 0:
        return []

    width_profile = []
    cols = column_intervals(mask)

    for c in range(w):
        total_width = sum(end - start + 1 for start, end in cols[c])
        width_profile.append(total_width)

    if not any(width_profile):
        return []

    waist_points = _find_waist_points(width_profile)

    if not waist_points:
        # Single component: include all non-empty columns
        components = []
        first_col = next((c for c in range(w) if cols[c]), None)
        if first_col is not None:
            comp = {}
            for c in range(first_col, w):
                if cols[c]:
                    comp[c] = cols[c]
            if len(comp) >= 3:
                components.append(comp)
        return components

    # Split into components based on waist points
    components = []
    split_points = [0] + waist_points + [w]

    for i in range(len(split_points) - 1):
        start_col = split_points[i]
        end_col = split_points[i + 1]
        comp = {}
        for c in range(start_col, end_col):
            if c < w and cols[c]:
                comp[c] = cols[c]
        if len(comp) >= max(3, (end_col - start_col) * 0.3):
            components.append(comp)

    return components


def _find_waist_points(width_profile, min_prominence=0.1):
    """
    Find waist points (local minima) in the width profile.
    """
    if len(width_profile) < 5:
        return []

    smoothed = _smooth_1d(width_profile, window=5)

    max_width = max(smoothed)
    if max_width == 0:
        return []

    waists = []
    for i in range(2, len(smoothed) - 2):
        current = smoothed[i]
        if current == 0:
            continue
        if current < smoothed[i - 1] and current < smoothed[i + 1]:
            left_range = smoothed[max(0, i - 15) : i]
            right_range = smoothed[i + 1 : min(len(smoothed), i + 16)]
            left_max = max(left_range) if left_range else current
            right_max = max(right_range) if right_range else current
            local_max = max(left_max, right_max)
            if local_max > current:
                prominence = (local_max - current) / local_max
                if prominence >= min_prominence:
                    waists.append(i)

    # Enforce minimum separation
    filtered = []
    min_sep = max(1, len(smoothed) // 8)
    for wpt in waists:
        if not filtered or (wpt - filtered[-1]) >= min_sep:
            filtered.append(wpt)
    return filtered


def _smooth_1d(data, window=5):
    """Simple 1D moving average smoothing."""
    if len(data) < window:
        return data
    smoothed = []
    half = window // 2
    for i in range(len(data)):
        s = max(0, i - half)
        e = min(len(data), i + half + 1)
        smoothed.append(sum(data[s:e]) / (e - s))
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
