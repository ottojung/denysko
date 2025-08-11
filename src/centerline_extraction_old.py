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
    for target in targets:
        while j < len(d) and d[j] < target:
            j += 1
        if j == len(d):
            out.append(points[-1])
        elif d[j] == target:
            out.append(points[j])
        else:
            prev_point = points[j - 1]
            next_point = points[j]
            t = (target - d[j - 1]) / (d[j] - d[j - 1])
            out.append(prev_point + t * (next_point - prev_point))
    return np.array(out)


class CenterlineExtractor:
    """Class for extracting centerlines using integer point jumping algorithm."""

    def _generate_letter_points(self, path, resolution=200):
        """Generate integer coordinate points from the letter path.
        
        Args:
            path: matplotlib Path object
            resolution: Resolution for rasterization
            
        Returns:
            List of (x, y) integer coordinate tuples
        """
        from .path_processing import rasterize_path
        
        # Rasterize the path to get a binary mask
        mask = rasterize_path(path, resolution=resolution)
        
        # Find all filled pixels and convert to integer coordinates
        y_coords, x_coords = np.where(mask)
        
        # Convert to list of integer coordinate tuples and remove duplicates
        letter_points = []
        seen = set()
        for y, x in zip(y_coords, x_coords):
            point = (int(x), int(y))
            if point not in seen:
                seen.add(point)
                letter_points.append(point)
        
        return letter_points

    def _build_spatial_index(self, letter_points):
        """Build spatial index for fast neighbor lookup."""
        # Simple spatial index using dictionary mapping (x,y) -> index
        point_index = {}
        for i, point in enumerate(letter_points):
            point_index[point] = i
        return point_index

    def _find_neighbors(self, point, spatial_index, visited, step_distance, mask):
        """Find valid unvisited neighbors within step_distance that are inside the letter boundary."""
        x, y = point
        neighbors = []
        
        # Search in a square region around the point
        for dx in range(-step_distance, step_distance + 1):
            for dy in range(-step_distance, step_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor = (x + dx, y + dy)
                
                # Check if neighbor exists in spatial index and hasn't been visited
                if neighbor in spatial_index and neighbor not in visited:
                    # Check distance constraint
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= step_distance:
                        # Check if neighbor is within mask bounds and inside letter boundary
                        nx, ny = neighbor
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                            if mask[ny, nx]:  # Point is inside the letter
                                neighbors.append(neighbor)
        
        return neighbors

    def _generate_jumping_walk(self, start_point, direction, letter_points, point_index, step_distance, max_steps, mask):
        """Generate a walk by jumping to neighboring points."""
        walk = [start_point]
        current = start_point
        visited = {start_point}
        
        for _ in range(max_steps):
            # Find neighboring points within step_distance
            neighbors = self._find_neighbors(current, point_index, visited, step_distance, mask)
            
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

    def extract_skeleton_from_path(self, path, num_walks=25, step_distance=3, max_steps=100):
        """Extract skeleton using integer point jumping algorithm.
        
        Args:
            path: matplotlib Path object representing the letter
            num_walks: Number of random walks to generate
            step_distance: Maximum distance for neighbor search
            max_steps: Maximum steps per walk
            
        Returns:
            List of numpy arrays, each representing a walk path
        """
        print("Starting integer point jumping algorithm...")
        
        # Generate integer points from the letter
        letter_points = self._generate_letter_points(path)
        print(f"Generated {len(letter_points)} integer points for letter representation")
        
        if len(letter_points) < 10:
            print("Warning: Very few points generated, may not produce good results")
            return []
        
        # Build spatial index for fast neighbor lookup
        point_index = self._build_spatial_index(letter_points)
        
        # Generate mask for boundary checking
        from .path_processing import rasterize_path
        mask = rasterize_path(path, resolution=200)
        
        # Generate random starting points
        num_starts = min(num_walks, len(letter_points) // 2)
        start_indices = np.random.choice(len(letter_points), size=num_starts, replace=False)
        start_points = [letter_points[i] for i in start_indices]
        print(f"Selected {len(start_points)} starting points")
        
        # Generate walks
        all_walks = []
        for start_point in start_points:
            # Generate right-to-left walk  
            right_walk = self._generate_jumping_walk(
                start_point, 'right', letter_points, point_index, 
                step_distance, max_steps, mask
            )
            
            # Generate left-to-right walk
            left_walk = self._generate_jumping_walk(
                start_point, 'left', letter_points, point_index, 
                step_distance, max_steps, mask
            )
            
            if len(right_walk) > 1:
                all_walks.append(np.array(right_walk, dtype=float))
            if len(left_walk) > 1:
                all_walks.append(np.array(left_walk, dtype=float))
        
        print(f"Generated {len(all_walks)} walks")
        return all_walks


# Standalone function for backwards compatibility
def extract_skeleton_from_path(path, num_walks=25, step_distance=3, max_steps=100):
    """Standalone function using CenterlineExtractor class."""
    extractor = CenterlineExtractor()
    return extractor.extract_skeleton_from_path(path, num_walks, step_distance, max_steps)


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


    def _build_spatial_index(self, letter_points):
        """Build spatial index for fast neighbor lookup."""
        # Simple spatial index using dictionary mapping (x,y) -> index
        point_index = {}
        for i, point in enumerate(letter_points):
            point_index[point] = i
        return point_index


    def _generate_jumping_walk(self, start_point, direction, letter_points, point_index, step_distance, max_steps, mask):
        """Generate a walk by jumping to neighboring points."""
        walk = [start_point]
        current = start_point
        visited = {start_point}
        
        for _ in range(max_steps):
            # Find neighboring points within step_distance
            neighbors = self._find_neighbors(current, point_index, visited, step_distance, mask)
            
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


    def _find_neighbors(self, point, spatial_index, visited, step_distance, mask):
        """Find valid unvisited neighbors within step_distance that are inside the letter boundary."""
        x, y = point
        neighbors = []
        
        # Search in a square region around the point
        for dx in range(-step_distance, step_distance + 1):
            for dy in range(-step_distance, step_distance + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                neighbor = (x + dx, y + dy)
                
                # Check if neighbor exists in spatial index and hasn't been visited
                if neighbor in spatial_index and neighbor not in visited:
                    # Check distance constraint
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance <= step_distance:
                        # Check if neighbor is within mask bounds and inside letter boundary
                        nx, ny = neighbor
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                            if mask[ny, nx]:  # Point is inside the letter
                                neighbors.append(neighbor)
        
        return neighbors