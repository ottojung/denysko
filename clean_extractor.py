#!/usr/bin/env python3
"""
Clean, principled centerline extraction from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import font_manager
from matplotlib.textpath import TextPath


class CenterlineExtractor:
    """Clean implementation of centerline extraction with proper hole handling."""
    
    def __init__(self):
        pass
    
    def extract_centerlines(self, text_path, step_size=1.0, num_starts=15, max_steps=50):
        """
        Extract centerlines using random walks.
        
        Args:
            text_path: matplotlib TextPath object
            step_size: Distance between walk steps
            num_starts: Number of random starting points
            max_steps: Maximum steps per walk
            
        Returns:
            List of walk paths, each as list of (x, y) tuples
        """
        # Step 1: Get proper bounds
        bounds = self._get_path_bounds(text_path)
        print(f"Path bounds: {bounds}")
        
        # Step 2: Create high-resolution binary mask with proper hole handling
        mask = self._create_binary_mask(text_path, bounds, resolution=400)
        print(f"Mask: {mask.shape}, filled pixels: {np.sum(mask)}")
        
        # Step 3: Find valid interior starting points
        start_points = self._find_interior_points(mask, bounds, num_starts)
        print(f"Found {len(start_points)} interior starting points")
        
        if not start_points:
            return []
        
        # Step 4: Generate bidirectional walks from each starting point
        all_walks = []
        for start_point in start_points:
            # Right walk
            right_walk = self._generate_walk(start_point, 'right', mask, bounds, step_size, max_steps)
            if len(right_walk) > 1:
                all_walks.append(right_walk)
            
            # Left walk
            left_walk = self._generate_walk(start_point, 'left', mask, bounds, step_size, max_steps)
            if len(left_walk) > 1:
                all_walks.append(left_walk)
        
        print(f"Generated {len(all_walks)} walks")
        return all_walks
    
    def _get_path_bounds(self, path):
        """Get the bounding box of the path."""
        vertices = path.vertices
        if len(vertices) == 0:
            return (0, 0, 1, 1)
        
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        
        # Add small padding
        padding = 0.1 * max(max_x - min_x, max_y - min_y)
        return (min_x - padding, min_y - padding, max_x + padding, max_y + padding)
    
    def _create_binary_mask(self, path, bounds, resolution=400):
        """
        Create a binary mask using matplotlib's contains_points with proper hole handling.
        This is the most reliable way to handle complex paths with holes.
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Create coordinate grid
        x = np.linspace(min_x, max_x, resolution)
        y = np.linspace(min_y, max_y, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Flatten to points array
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Use matplotlib's contains_points - this properly handles holes with winding rules
        inside = path.contains_points(points)
        
        # Reshape back to 2D mask
        mask = inside.reshape((resolution, resolution))
        
        return mask
    
    def _find_interior_points(self, mask, bounds, num_points):
        """Find random interior points from the mask."""
        # Get all interior pixels (sample every few pixels for performance)
        y_indices, x_indices = np.where(mask[::3, ::3])
        
        if len(y_indices) == 0:
            return []
        
        # Scale back up to full resolution
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
    
    def _generate_walk(self, start_point, direction, mask, bounds, step_size, max_steps):
        """Generate a monotonic random walk."""
        walk = [start_point]
        current = np.array(start_point)
        
        for _ in range(max_steps):
            # Generate biased random step
            angle = np.random.uniform(-np.pi, np.pi)
            
            # Apply directional bias (70% chance)
            if np.random.random() < 0.7:
                if direction == 'right':
                    angle = np.random.uniform(-np.pi/4, np.pi/4)
                else:  # left
                    angle = np.random.uniform(3*np.pi/4, 5*np.pi/4)
            
            # Calculate next point
            step = step_size * np.array([np.cos(angle), np.sin(angle)])
            next_point = current + step
            
            # Check if next point is valid
            if not self._is_point_inside(next_point, mask, bounds):
                break
            
            # Additional check: midpoint (prevents tunneling)
            midpoint = (current + next_point) / 2
            if not self._is_point_inside(midpoint, mask, bounds):
                break
            
            walk.append(tuple(next_point))
            current = next_point
        
        return walk
    
    def _is_point_inside(self, point, mask, bounds):
        """Check if a point is inside the mask."""
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


def test_clean_extraction():
    """Test the clean extraction implementation."""
    print("Testing clean centerline extraction...")
    
    # Create letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    # Extract centerlines
    extractor = CenterlineExtractor()
    walks = extractor.extract_centerlines(text_path)
    
    print(f"\\nExtraction results:")
    print(f"Total walks: {len(walks)}")
    
    total_points = 0
    for i, walk in enumerate(walks):
        print(f"Walk {i+1}: {len(walk)} points")
        total_points += len(walk)
    
    if walks:
        print(f"Average points per walk: {total_points / len(walks):.1f}")
    
    # Visualize
    plt.figure(figsize=(12, 8))
    
    # Plot walks
    colors = plt.cm.tab20(np.linspace(0, 1, len(walks)))
    for i, walk in enumerate(walks):
        if len(walk) > 1:
            walk_array = np.array(walk)
            plt.plot(walk_array[:, 0], walk_array[:, 1], 
                    color=colors[i], linewidth=2, alpha=0.8)
            plt.plot(walk_array[0, 0], walk_array[0, 1], 'o',
                    color=colors[i], markersize=6)
    
    # Plot letter outline
    vertices = text_path.vertices
    plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=1, alpha=0.3, label='Letter outline')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Clean Centerline Extraction for "A"\\n{len(walks)} walks, {total_points} points')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('clean_extraction_test.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'clean_extraction_test.png'")
    
    return walks


if __name__ == "__main__":
    np.random.seed(42)  # Reproducible results
    test_clean_extraction()
