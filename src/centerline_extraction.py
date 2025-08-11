#!/usr/bin/env python3
"""
Centerline extraction using horizontal monotonic decomposition and random walks.
"""

import numpy as np
from .path_processing import rasterize_path, decompose_into_h_monotonic_components


def extract_skeleton_from_path(path):
    """
    Extract centerlines by:
    1. Decomposing letter into horizontal monotonic components
    2. Generating random left-to-right walks through each component  
    3. Averaging walks to get centerline for each component
    
    Returns list of numpy arrays, each representing a centerline trace.
    """
    vertices = path.vertices
    if len(vertices) < 6:
        return [vertices]

    # Rasterize at high resolution for better component detection
    mask, x_grid, y_grid = rasterize_path(path, resolution=400)
    if mask.sum() == 0:
        return [vertices]

    print(f"Rasterized shape: {mask.shape}, filled pixels: {mask.sum()}")

    # Decompose into horizontal monotonic components
    raw_components = decompose_into_h_monotonic_components(mask)
    print(f"Found {len(raw_components)} horizontal monotonic components")
    
    if not raw_components:
        print("Warning: No components found, using fallback")
        return [_create_simple_stroke_approximation(path)]

    # Convert raw components to format needed for centerline extraction
    components = []
    for comp_dict in raw_components:
        # Find bounding box of this component
        cols = list(comp_dict.keys())
        if not cols:
            continue
        x_min_col, x_max_col = min(cols), max(cols)
        
        # Create submask for this component
        component_mask = np.zeros_like(mask)
        for col, intervals in comp_dict.items():
            for start_row, end_row in intervals:
                component_mask[start_row:end_row+1, col] = mask[start_row:end_row+1, col]
        
        component = {
            'mask': component_mask,
            'x_grid': x_grid,
            'y_grid': y_grid, 
            'x_min': x_grid[0, x_min_col],
            'x_max': x_grid[0, x_max_col]
        }
        components.append(component)

    # Extract centerline for each component
    centerlines = []
    for i, component in enumerate(components):
        print(f"Processing component {i+1}/{len(components)}")
        centerline = _extract_centerline_from_component(component, path)
        if centerline is not None and len(centerline) >= 3:
            centerlines.append(centerline)
            print(f"  Generated centerline with {len(centerline)} points")
    
    if not centerlines:
        print("Warning: No centerlines extracted, using fallback")
        return [_create_simple_stroke_approximation(path)]

    print(f"Generated {len(centerlines)} centerline traces")
    return centerlines


def _extract_centerline_from_component(component, original_path):
    """
    Extract centerline from a horizontal monotonic component using random walks.
    
    Args:
        component: Dict with 'mask', 'x_grid', 'y_grid', 'x_min', 'x_max'
        original_path: Original path for boundary validation
    
    Returns:
        numpy array representing the centerline trace
    """
    mask = component['mask']
    x_grid = component['x_grid'] 
    y_grid = component['y_grid']
    x_min = component['x_min']
    x_max = component['x_max']
    
    h, w = mask.shape
    
    # Generate multiple random left-to-right walks through the component
    num_walks = 10  # Number of random walks to average
    walks = []
    
    for walk_i in range(num_walks):
        walk = _generate_random_walk(mask, x_grid, y_grid, original_path)
        if len(walk) >= 3:
            walks.append(walk)
    
    if not walks:
        return None
    
    print(f"    Generated {len(walks)} valid random walks")
    
    # Average the walks to get the centerline
    centerline = _average_walks_to_centerline(walks, x_min, x_max)
    
    return centerline


def _generate_random_walk(mask, x_grid, y_grid, original_path):
    """
    Generate a single random left-to-right walk through the component.
    Walk is brownian but constrained to move left-to-right overall.
    """
    h, w = mask.shape
    
    # Find leftmost column with pixels
    left_col = None
    for col in range(w):
        if np.any(mask[:, col]):
            left_col = col
            break
    
    if left_col is None:
        return []
    
    # Start at a random valid pixel in the leftmost column
    left_pixels = np.where(mask[:, left_col])[0]
    if len(left_pixels) == 0:
        return []
    
    start_row = np.random.choice(left_pixels)
    
    # Random walk from left to right
    walk = []
    current_row = start_row
    
    for col in range(left_col, w):
        # Find valid pixels in current column
        valid_rows = np.where(mask[:, col])[0]
        if len(valid_rows) == 0:
            break
            
        # Brownian motion: prefer staying near current row, but allow drift
        if col > left_col:
            # Weight rows by distance from current position (closer = higher weight)
            distances = np.abs(valid_rows - current_row)
            max_dist = max(distances.max(), 1)
            weights = np.exp(-2 * distances / max_dist)  # Exponential falloff
            weights /= weights.sum()
            
            current_row = np.random.choice(valid_rows, p=weights)
        else:
            current_row = start_row
        
        # Convert to world coordinates
        x = x_grid[0, col]
        y = y_grid[current_row, 0]
        point = np.array([x, y])
        
        # Verify point is within original letter boundary
        if original_path.contains_point(point):
            walk.append(point)
    
    return np.array(walk) if walk else np.array([])


def _average_walks_to_centerline(walks, x_min, x_max):
    """
    Average multiple walks to produce a smooth centerline.
    Samples walks at regular x intervals and averages y coordinates.
    """
    if not walks:
        return None
    
    # Sample at regular x intervals
    num_samples = 50  # Number of points in final centerline
    x_samples = np.linspace(x_min, x_max, num_samples)
    
    centerline_points = []
    
    for x_target in x_samples:
        y_values = []
        
        # For each walk, find y-value at this x position
        for walk in walks:
            if len(walk) < 2:
                continue
                
            x_coords = walk[:, 0]
            y_coords = walk[:, 1]
            
            # Find y at x_target by linear interpolation
            if x_target >= x_coords.min() and x_target <= x_coords.max():
                y_interp = np.interp(x_target, x_coords, y_coords)
                y_values.append(y_interp)
        
        # Average the y values from all walks
        if y_values:
            avg_y = np.mean(y_values)
            centerline_points.append([x_target, avg_y])
    
    return np.array(centerline_points) if centerline_points else None


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
