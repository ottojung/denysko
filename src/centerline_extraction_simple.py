#!/usr/bin/env python3
"""
Simplified centerline extraction that returns all points inside the letter shape.
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
    """Extract all points inside a letter's shape for polynomial fitting."""
    
    def __init__(self):
        pass
    
    def _generate_letter_points(self, path, resolution=400):
        """Generate all points inside the letter path."""
        from .path_processing import rasterize_path
        
        # Rasterize the path to get a binary mask and coordinate grids
        mask, x_grid, y_grid = rasterize_path(path, resolution=resolution)
        
        # Store coordinate transformation parameters including the actual grids
        coord_info = {
            'mask_shape': mask.shape,
            'x_grid': x_grid,
            'y_grid': y_grid
        }
        
        # Find all filled pixels and convert to letter coordinates
        y_coords, x_coords = np.where(mask)
        
        # Convert pixel coordinates to letter coordinates
        letter_points = []
        for py, px in zip(y_coords, x_coords):
            # Use the 2D coordinate grids for transformation
            letter_x = x_grid[py, px]
            letter_y = y_grid[py, px]
            
            # Invert y-coordinate to match the inverted y-axis in preview
            min_y = np.min(y_grid)
            max_y = np.max(y_grid)
            letter_y = min_y + max_y - letter_y  # Flip y within its range
            
            letter_points.append((letter_x, letter_y))
        
        return np.array(letter_points), coord_info

    def extract_skeleton_from_path(self, path, num_walks=25, step_distance=3, max_steps=100):
        """
        Extract all points inside the letter shape for genetic polynomial fitting.
        Parameters are kept for compatibility but ignored.
        Returns a single array containing all points inside the shape.
        """
        print("Extracting all points inside letter shape...")
        
        letter_points, coord_info = self._generate_letter_points(path)
        
        print(f"Generated {len(letter_points)} points inside letter shape")
        
        # Return as a list with a single trace containing all points
        return [letter_points]


def extract_skeleton_from_path(path, num_walks=25, step_distance=3, max_steps=100):
    """Module-level function for backward compatibility."""
    extractor = CenterlineExtractor()
    return extractor.extract_skeleton_from_path(path, num_walks, step_distance, max_steps)
