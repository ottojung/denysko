#!/usr/bin/env python3
"""
Clean centerline extraction using random starting points and monotonic random walks.
"""

import numpy as np


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


def extract_skeleton_from_path(path, step_size=1.0, num_starting_points=25, num_walks=2):
    """
    Extract skeleton using random starting points and monotonic walks.
    
    Args:
        path: matplotlib Path object
        step_size: Distance between walk steps
        num_starting_points: Number of random interior starting points
        num_walks: Number of walks per starting point (default 2 for bidirectional)
    
    Returns:
        List of walk paths, each path is an array of (x, y) coordinates
    """
    vertices = path.vertices
    if len(vertices) < 6:
        return [vertices]

    # Get proper bounds with padding
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    padding = 0.05 * max(max_x - min_x, max_y - min_y)
    bounds = (min_x - padding, min_y - padding, max_x + padding, max_y + padding)

    # Create clean binary mask using contains_points (handles holes correctly)
    mask = _create_clean_mask(path, bounds, resolution=400)
    if mask.sum() == 0:
        return [vertices]

    print(f"Clean mask: {mask.shape}, filled pixels: {mask.sum()}")

    # Find random interior starting points
    start_points = _find_interior_starting_points(mask, bounds, num_points=15)
    if not start_points:
        return [vertices]

    print(f"Found {len(start_points)} interior starting points")

    # Generate bidirectional walks from each starting point
    all_walks = []
    step_distance = 1.0  # Step distance C
    max_steps = 50

    for start_point in start_points:
        # Right-biased walk
        right_walk = _generate_monotonic_walk(
            start_point, 'right', mask, bounds, step_distance, max_steps
        )
        if len(right_walk) > 1:
            all_walks.append(right_walk)
            
        # Left-biased walk  
        left_walk = _generate_monotonic_walk(
            start_point, 'left', mask, bounds, step_distance, max_steps
        )
        if len(left_walk) > 1:
            all_walks.append(left_walk)

    print(f"Generated {len(all_walks)} walks")
    
    # Convert each walk to numpy array for compatibility with downstream code
    return [np.array(walk) for walk in all_walks]


def _create_clean_mask(path, bounds, resolution=400):
    """Create binary mask using matplotlib's contains_points (proper hole handling)."""
    min_x, min_y, max_x, max_y = bounds
    
    # Create coordinate grid
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten to points array
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Use matplotlib's contains_points - handles holes with winding rules
    inside = path.contains_points(points)
    
    # Reshape back to 2D mask
    mask = inside.reshape((resolution, resolution))
    
    return mask


def _find_interior_starting_points(mask, bounds, num_points=15):
    """Find random interior points from the mask."""
    # Sample every 3rd pixel for performance
    y_indices, x_indices = np.where(mask[::3, ::3])
    
    if len(y_indices) == 0:
        return []
    
    # Scale back to full resolution
    y_indices = y_indices * 3
    x_indices = x_indices * 3
    
    # Convert to world coordinates
    min_x, min_y, max_x, max_y = bounds
    height, width = mask.shape
    
    world_x = min_x + (x_indices / (width - 1)) * (max_x - min_x)
    world_y = min_y + (y_indices / (height - 1)) * (max_y - min_y)
    
    # Random sample
    n_available = len(world_x)
    n_sample = min(num_points, n_available)
    indices = np.random.choice(n_available, n_sample, replace=False)
    
    return [(world_x[i], world_y[i]) for i in indices]


def _generate_monotonic_walk(start_point, direction, mask, bounds, step_size, max_steps):
    """Generate a monotonic random walk with proper boundary checking."""
    walk = [start_point]
    current = np.array(start_point)
    
    for _ in range(max_steps):
        # Generate biased random step
        angle = np.random.uniform(-np.pi, np.pi)
        
        # Apply directional bias (70% chance)
        if np.random.random() < 0.7:
            if direction == 'right':
                angle = np.random.uniform(-np.pi/4, np.pi/4)  # Rightward bias
            else:  # left
                angle = np.random.uniform(3*np.pi/4, 5*np.pi/4)  # Leftward bias
        
        # Calculate next point
        step = step_size * np.array([np.cos(angle), np.sin(angle)])
        next_point = current + step
        
        # Check if next point is valid (inside letter, outside holes)
        if not _is_point_valid(next_point, mask, bounds):
            break
        
        # Additional check: midpoint to prevent tunneling
        midpoint = (current + next_point) / 2
        if not _is_point_valid(midpoint, mask, bounds):
            break
        
        walk.append(tuple(next_point))
        current = next_point
    
    return walk


def _is_point_valid(point, mask, bounds):
    """Check if a point is inside the valid region (letter shape, excluding holes)."""
    x, y = point
    min_x, min_y, max_x, max_y = bounds
    
    # Check bounds
    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False
    
    # Convert to pixel coordinates
    height, width = mask.shape
    px = int((x - min_x) / (max_x - min_x) * (width - 1))
    py = int((y - min_y) / (max_y - min_y) * (height - 1))
    
    # Clamp to valid range
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    
    return mask[py, px]
