#!/usr/bin/env python3
"""
Centerline extraction using random starting points and monotonic random walks.
"""

import numpy as np
from .path_processing import rasterize_path


def extract_skeleton_from_path(path):
    """
    Extract centerlines by:
    1. Choosing random points on the letter shape
    2. From each point, doing two monotonic random walks (left-to-right and right-to-left)
    3. Returning all walk paths
    
    Returns list of numpy arrays, each representing a walk path.
    """
    vertices = path.vertices
    if len(vertices) < 6:
        return [vertices]

    # Rasterize at high resolution to get valid interior points
    mask, x_grid, y_grid = rasterize_path(path, resolution=400)
    if mask.sum() == 0:
        return [vertices]

    print(f"Rasterized shape: {mask.shape}, filled pixels: {mask.sum()}")

    # Get all valid points inside the letter shape (in real coordinates)
    valid_points = _get_valid_interior_points(mask, x_grid, y_grid, path)
    if len(valid_points) == 0:
        print("Warning: No valid interior points found")
        return [_create_simple_stroke_approximation(path)]

    print(f"Found {len(valid_points)} valid interior points")

    # Choose random starting points
    num_starts = 15  # Number of random starting points
    if len(valid_points) < num_starts:
        start_points = valid_points
    else:
        start_indices = np.random.choice(len(valid_points), num_starts, replace=False)
        start_points = valid_points[start_indices]

    print(f"Using {len(start_points)} random starting points")

    # Generate monotonic walks from each starting point
    all_walks = []
    step_distance = 2.0  # Distance "C" between steps
    
    for i, start_point in enumerate(start_points):
        # Left-to-right walk
        lr_walk = _monotonic_random_walk(start_point, path, step_distance, direction='left-to-right')
        if len(lr_walk) >= 3:
            all_walks.append(lr_walk)
            
        # Right-to-left walk  
        rl_walk = _monotonic_random_walk(start_point, path, step_distance, direction='right-to-left')
        if len(rl_walk) >= 3:
            all_walks.append(rl_walk)

    if not all_walks:
        print("Warning: No valid walks generated")
        return [_create_simple_stroke_approximation(path)]

    print(f"Generated {len(all_walks)} walk paths")
    return all_walks


def _get_valid_interior_points(mask, x_grid, y_grid, original_path):
    """
    Get all valid points inside the letter shape as real coordinates.
    Sample from the rasterized mask and convert to world coordinates.
    """
    h, w = mask.shape
    valid_points = []
    
    # Sample every few pixels to get a good distribution of interior points
    sample_step = 3  # Sample every 3rd pixel
    
    for row in range(0, h, sample_step):
        for col in range(0, w, sample_step):
            if mask[row, col]:
                # Convert pixel coordinates to world coordinates
                x = x_grid[0, col]
                y = y_grid[row, 0]
                point = np.array([x, y])
                
                # Double-check that point is within original path
                if original_path.contains_point(point):
                    valid_points.append(point)
    
    return np.array(valid_points)


def _monotonic_random_walk(start_point, original_path, step_distance, direction='left-to-right'):
    """
    Generate a monotonic random walk from a starting point.
    
    Args:
        start_point: Starting coordinates [x, y]
        original_path: Original letter path for boundary checking
        step_distance: Distance between consecutive steps
        direction: 'left-to-right' or 'right-to-left'
    
    Returns:
        numpy array of walk coordinates
    """
    walk = [start_point.copy()]
    current_point = start_point.copy()
    max_steps = 200  # Prevent infinite walks
    
    x_direction = 1.0 if direction == 'left-to-right' else -1.0
    
    for step in range(max_steps):
        # Generate next point with monotonic constraint
        # x must increase/decrease monotonically
        # y can vary randomly within bounds
        
        # Random angle bias towards the monotonic direction
        if direction == 'left-to-right':
            # Bias angle towards rightward (0 to π/2 and 3π/2 to 2π)
            if np.random.random() < 0.7:  # 70% chance of rightward bias
                angle = np.random.uniform(-np.pi/3, np.pi/3)  # -60° to +60°
            else:
                angle = np.random.uniform(0, 2*np.pi)  # Any direction
        else:  # right-to-left
            # Bias angle towards leftward (π/2 to 3π/2)
            if np.random.random() < 0.7:  # 70% chance of leftward bias
                angle = np.random.uniform(2*np.pi/3, 4*np.pi/3)  # 120° to 240°
            else:
                angle = np.random.uniform(0, 2*np.pi)  # Any direction
        
        # Calculate next point
        dx = step_distance * np.cos(angle)
        dy = step_distance * np.sin(angle)
        
        # Ensure monotonic constraint
        if direction == 'left-to-right' and dx < 0:
            dx = abs(dx)  # Force rightward
        elif direction == 'right-to-left' and dx > 0:
            dx = -abs(dx)  # Force leftward
            
        next_point = current_point + np.array([dx, dy])
        
        # Check if next point is still within the letter boundary
        if not original_path.contains_point(next_point):
            break
            
        walk.append(next_point.copy())
        current_point = next_point
    
    return np.array(walk)


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
