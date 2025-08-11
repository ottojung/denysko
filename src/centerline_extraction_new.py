#!/usr/bin/env python3
"""
New centerline extraction using integer point jumping.
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
    Extract skeleton using integer point jumping algorithm.
    
    Args:
        path: matplotlib Path object
        step_size: Not used (for compatibility) 
        num_starting_points: Number of random starting points
        num_walks: Number of walks per starting point (default 2 for bidirectional)
    
    Returns:
        List of walk paths, each path is an array of (x, y) coordinates
    """
    vertices = path.vertices
    if len(vertices) < 6:
        return [vertices]

    print("Starting integer point jumping algorithm...")

    # Step 1: Generate collection of integer points representing the letter
    letter_points = _generate_letter_points(path)
    if len(letter_points) < 10:
        print(f"Too few letter points ({len(letter_points)}), falling back to vertices")
        return [vertices]
        
    print(f"Generated {len(letter_points)} integer points for letter representation")

    # Step 2: Create spatial index for fast neighbor lookup
    point_index = _build_spatial_index(letter_points)
    
    # Step 3: Select random starting points
    num_points = min(num_starting_points, len(letter_points))
    start_indices = np.random.choice(len(letter_points), num_points, replace=False)
    start_points = [letter_points[i] for i in start_indices]
    print(f"Selected {len(start_points)} starting points")

    # Step 4: Generate walks from each starting point
    all_walks = []
    step_distance = 3  # Jump distance for neighbors
    max_steps = 100
    
    for start_point in start_points:
        # Right-to-left walk  
        right_walk = _generate_jumping_walk(
            start_point, 'right', letter_points, point_index, step_distance, max_steps
        )
        if len(right_walk) > 1:
            all_walks.append(right_walk)
            
        # Left-to-right walk  
        left_walk = _generate_jumping_walk(
            start_point, 'left', letter_points, point_index, step_distance, max_steps
        )
        if len(left_walk) > 1:
            all_walks.append(left_walk)

    print(f"Generated {len(all_walks)} walks")
    
    # Convert each walk to numpy array for compatibility with downstream code
    return [np.array(walk, dtype=float) for walk in all_walks]


def _generate_letter_points(path):
    """Generate collection of integer points representing the letter shape."""
    from .path_processing import rasterize_path
    
    # Get binary mask with proper hole handling
    mask, x_grid, y_grid = rasterize_path(path, resolution=200)
    
    # Find all filled pixel coordinates
    filled_rows, filled_cols = np.where(mask)
    
    # Convert pixel coordinates to world coordinates
    min_x, max_x = x_grid[0, 0], x_grid[0, -1]
    min_y, max_y = y_grid[0, 0], y_grid[-1, 0]
    height, width = mask.shape
    
    letter_points = []
    for row, col in zip(filled_rows, filled_cols):
        # Convert to world coordinates and round to integers
        world_x = min_x + (col / (width - 1)) * (max_x - min_x)
        world_y = min_y + (row / (height - 1)) * (max_y - min_y)
        letter_points.append((int(round(world_x)), int(round(world_y))))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_points = []
    for point in letter_points:
        if point not in seen:
            seen.add(point)
            unique_points.append(point)
    
    return unique_points


def _build_spatial_index(letter_points):
    """Build spatial index for fast neighbor lookup."""
    # Simple spatial index using dictionary mapping (x,y) -> index
    point_index = {}
    for i, point in enumerate(letter_points):
        point_index[point] = i
    return point_index


def _generate_jumping_walk(start_point, direction, letter_points, point_index, step_distance, max_steps):
    """Generate a walk by jumping to neighboring points."""
    walk = [start_point]
    current = start_point
    visited = {start_point}
    
    for _ in range(max_steps):
        # Find neighboring points within step_distance
        neighbors = _find_neighbors(current, letter_points, point_index, step_distance, visited)
        
        if not neighbors:
            break
            
        # Apply directional bias for monotonic behavior
        if direction == 'right':
            # Prefer neighbors to the right (higher x)
            neighbors = sorted(neighbors, key=lambda p: -p[0])  # Sort by x descending
        else:  # left
            # Prefer neighbors to the left (lower x) 
            neighbors = sorted(neighbors, key=lambda p: p[0])   # Sort by x ascending
            
        # Take top candidates with some randomness
        num_candidates = min(5, len(neighbors))
        candidates = neighbors[:num_candidates]
        next_point = candidates[np.random.randint(len(candidates))]
        
        walk.append(next_point)
        visited.add(next_point)
        current = next_point
    
    return walk


def _find_neighbors(current, letter_points, point_index, max_distance, visited):
    """Find neighboring letter points within max_distance."""
    x, y = current
    neighbors = []
    
    # Search in a square around current point
    for dx in range(-max_distance, max_distance + 1):
        for dy in range(-max_distance, max_distance + 1):
            if dx == 0 and dy == 0:
                continue
                
            neighbor = (x + dx, y + dy)
            
            # Check if this point exists in our letter representation
            if neighbor in point_index and neighbor not in visited:
                # Check distance
                distance = np.sqrt(dx*dx + dy*dy)
                if distance <= max_distance:
                    neighbors.append(neighbor)
    
    return neighbors
