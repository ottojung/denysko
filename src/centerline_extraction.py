#!/usr/bin/env python3
"""
Centerline extraction that traces all strokes within letter boundaries.
"""

import numpy as np
from .path_processing import rasterize_path


def extract_skeleton_from_path(path):
    """
    Extract multiple centerlines that trace all strokes of the letter.
    Returns list of numpy arrays, each representing a separate trace.
    """
    vertices = path.vertices
    if len(vertices) < 6:
        return [vertices]

    # Rasterize the path to work with pixels
    mask, x_grid, y_grid = rasterize_path(path, resolution=300)
    if mask.sum() == 0:
        return [vertices]

    # Generate multiple stroke traces
    all_traces = []
    
    # Strategy 1: Top-to-bottom traces
    top_traces = _extract_top_to_bottom_traces(mask, x_grid, y_grid, path)
    all_traces.extend(top_traces)
    
    # Strategy 2: Left-to-right traces  
    lr_traces = _extract_left_to_right_traces(mask, x_grid, y_grid, path)
    all_traces.extend(lr_traces)
    
    # Strategy 3: Diagonal traces
    diag_traces = _extract_diagonal_traces(mask, x_grid, y_grid, path)
    all_traces.extend(diag_traces)

    if not all_traces:
        return [_create_simple_stroke_approximation(path)]

    print(f"Generated {len(all_traces)} stroke traces")
    return all_traces


def _extract_top_to_bottom_traces(mask, x_grid, y_grid, original_path):
    """Extract vertical traces from top to bottom of the letter."""
    h, w = mask.shape
    traces = []
    
    # Sample columns across the width
    for col_step in range(0, w, max(1, w // 20)):  # ~20 vertical traces
        col = min(col_step, w - 1)
        
        # Find all vertical segments in this column
        segments = []
        in_segment = False
        start_row = 0
        
        for row in range(h):
            if mask[row, col] and not in_segment:
                start_row = row
                in_segment = True
            elif not mask[row, col] and in_segment:
                segments.append((start_row, row - 1))
                in_segment = False
        
        if in_segment:
            segments.append((start_row, h - 1))
        
        # Create traces for each segment
        for start_row, end_row in segments:
            if end_row - start_row >= 5:  # Minimum length
                trace_points = []
                for row in range(start_row, end_row + 1, 2):  # Every 2nd row for efficiency
                    x = x_grid[0, col]
                    y = y_grid[row, 0]
                    point = np.array([x, y])
                    if original_path.contains_point(point):
                        trace_points.append(point)
                
                if len(trace_points) >= 3:
                    traces.append(np.array(trace_points))
    
    return traces


def _extract_left_to_right_traces(mask, x_grid, y_grid, original_path):
    """Extract horizontal traces from left to right of the letter."""
    h, w = mask.shape
    traces = []
    
    # Sample rows across the height  
    for row_step in range(0, h, max(1, h // 15)):  # ~15 horizontal traces
        row = min(row_step, h - 1)
        
        # Find all horizontal segments in this row
        segments = []
        in_segment = False
        start_col = 0
        
        for col in range(w):
            if mask[row, col] and not in_segment:
                start_col = col
                in_segment = True
            elif not mask[row, col] and in_segment:
                segments.append((start_col, col - 1))
                in_segment = False
        
        if in_segment:
            segments.append((start_col, w - 1))
        
        # Create traces for each segment
        for start_col, end_col in segments:
            if end_col - start_col >= 5:  # Minimum length
                trace_points = []
                for col in range(start_col, end_col + 1, 2):  # Every 2nd col for efficiency
                    x = x_grid[0, col]
                    y = y_grid[row, 0]
                    point = np.array([x, y])
                    if original_path.contains_point(point):
                        trace_points.append(point)
                
                if len(trace_points) >= 3:
                    traces.append(np.array(trace_points))
    
    return traces


def _extract_diagonal_traces(mask, x_grid, y_grid, original_path):
    """Extract diagonal traces to capture slanted strokes."""
    h, w = mask.shape
    traces = []
    
    # Diagonal traces from top-left to bottom-right
    for start_offset in range(-h//2, w//2, max(1, min(h, w) // 10)):
        trace_points = []
        
        if start_offset >= 0:
            # Start from top edge
            start_row, start_col = 0, start_offset
        else:
            # Start from left edge
            start_row, start_col = -start_offset, 0
        
        row, col = start_row, start_col
        while row < h and col < w:
            if mask[row, col]:
                x = x_grid[0, col]
                y = y_grid[row, 0]
                point = np.array([x, y])
                if original_path.contains_point(point):
                    trace_points.append(point)
            row += 2
            col += 2
        
        if len(trace_points) >= 5:
            traces.append(np.array(trace_points))
    
    # Diagonal traces from top-right to bottom-left
    for start_offset in range(-h//2, w//2, max(1, min(h, w) // 10)):
        trace_points = []
        
        if start_offset >= 0:
            # Start from top edge
            start_row, start_col = 0, w - 1 - start_offset
        else:
            # Start from right edge  
            start_row, start_col = -start_offset, w - 1
        
        row, col = start_row, start_col
        while row < h and col >= 0:
            if mask[row, col]:
                x = x_grid[0, col]
                y = y_grid[row, 0]
                point = np.array([x, y])
                if original_path.contains_point(point):
                    trace_points.append(point)
            row += 2
            col -= 2
        
        if len(trace_points) >= 5:
            traces.append(np.array(trace_points))
    
    return traces


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
