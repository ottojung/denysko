#!/usr/bin/env python3
"""
Standalone test of the random walk extraction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import random

# Copy the rasterize_path function directly
def rasterize_path(path, bounds, resolution=400):
    """
    Convert a matplotlib Path to a rasterized binary mask using proper rendering.
    This handles holes correctly by using matplotlib's PathPatch rendering.
    """
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    
    # Create figure and axis with exact pixel dimensions
    dpi = 100
    fig_width = resolution / dpi
    fig_height = resolution / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Render the path as a filled patch
    patch = PathPatch(path, facecolor='white', edgecolor='none')
    ax.add_patch(patch)
    ax.set_facecolor('black')  # Background is black (False in mask)
    
    # Render to bitmap
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    
    # Convert to numpy array and extract binary mask
    # buf is RGBA, we want to detect white pixels (foreground)
    rgba_array = np.frombuffer(buf, dtype=np.uint8).reshape(resolution, resolution, 4)
    
    # White pixels (R=G=B=255) are foreground (True in mask)
    # We check the red channel (index 0) for white pixels
    mask = rgba_array[:, :, 0] > 127  # White pixels
    
    # Flip vertically because matplotlib has origin at bottom-left
    mask = np.flipud(mask)
    
    plt.close(fig)
    
    return mask

# Copy the extraction functions directly
def _get_valid_interior_points(mask, bounds, num_samples=15):
    """Get random interior points from the rasterized mask."""
    min_x, min_y, max_x, max_y = bounds
    
    # Find all filled pixels (every 3rd pixel for performance)
    filled_y, filled_x = np.where(mask[::3, ::3])
    
    if len(filled_x) == 0:
        return []
    
    # Scale back up (since we sampled every 3rd pixel)
    filled_x = filled_x * 3
    filled_y = filled_y * 3
    
    # Convert pixel coordinates to world coordinates
    height, width = mask.shape
    world_x = min_x + (filled_x / width) * (max_x - min_x)
    world_y = min_y + (filled_y / height) * (max_y - min_y)
    
    # Randomly sample points
    indices = np.random.choice(len(world_x), size=min(num_samples, len(world_x)), replace=False)
    
    return list(zip(world_x[indices], world_y[indices]))

def _is_point_inside_mask(point, mask, bounds):
    """Check if a point is inside the rasterized mask using nearest neighbor lookup."""
    x, y = point
    min_x, min_y, max_x, max_y = bounds
    height, width = mask.shape
    
    # Convert world coordinates to pixel coordinates
    px = int((x - min_x) / (max_x - min_x) * width)
    py = int((y - min_y) / (max_y - min_y) * height)
    
    # Clamp to valid bounds
    px = max(0, min(width - 1, px))
    py = max(0, min(height - 1, py))
    
    return mask[py, px]

def _monotonic_random_walk(start_point, direction, mask, bounds, max_steps=50, step_size=1.0):
    """Generate a monotonic random walk from a starting point."""
    path = [start_point]
    current_point = np.array(start_point)
    
    for _ in range(max_steps):
        # Generate random step with bias towards monotonic direction
        angle = np.random.uniform(-np.pi, np.pi)
        
        # Bias the angle towards the monotonic direction (70% chance)
        if np.random.random() < 0.7:
            if direction == 'right':
                angle = np.random.uniform(-np.pi/4, np.pi/4)  # Bias towards positive x
            else:  # direction == 'left'
                angle = np.random.uniform(3*np.pi/4, 5*np.pi/4)  # Bias towards negative x
        
        # Calculate step
        step = step_size * np.array([np.cos(angle), np.sin(angle)])
        next_point = current_point + step
        
        # Check midpoint to ensure we don't tunnel through thin sections
        midpoint = (current_point + next_point) / 2
        
        # Check if both midpoint and endpoint are inside the mask
        if not (_is_point_inside_mask(midpoint, mask, bounds) and 
                _is_point_inside_mask(next_point, mask, bounds)):
            break
        
        path.append(tuple(next_point))
        current_point = next_point
    
    return path

def extract_skeleton_from_path(path, step_size=1.0):
    """
    Extract random walk skeleton from a matplotlib Path.
    """
    # Get path bounds
    vertices = path.vertices
    if len(vertices) == 0:
        return []
    
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    bounds = (min_x, min_y, max_x, max_y)
    
    # Rasterize the path to create a mask
    mask = rasterize_path(path, bounds)
    
    # Get random interior starting points
    interior_points = _get_valid_interior_points(mask, bounds)
    
    if not interior_points:
        return []
    
    # Generate bidirectional walks from each starting point
    all_walks = []
    
    for start_point in interior_points:
        # Generate walk going right
        right_walk = _monotonic_random_walk(start_point, 'right', mask, bounds, step_size=step_size)
        if len(right_walk) > 1:
            all_walks.append(right_walk)
        
        # Generate walk going left
        left_walk = _monotonic_random_walk(start_point, 'left', mask, bounds, step_size=step_size)
        if len(left_walk) > 1:
            all_walks.append(left_walk)
    
    return all_walks

def test_extraction():
    """Test the random walk extraction directly."""
    print("Testing random walk extraction...")
    
    # Create path for letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    # Extract walks
    all_walks = extract_skeleton_from_path(text_path)
    
    print(f"Extracted {len(all_walks)} walks for letter 'A'")
    
    # Show statistics for each walk
    total_points = 0
    for i, walk_points in enumerate(all_walks):
        num_points = len(walk_points)
        total_points += num_points
        print(f"Walk {i+1}: {num_points} points")
    
    if all_walks:
        print(f"Total points across all walks: {total_points}")
        print(f"Average points per walk: {total_points / len(all_walks):.1f}")
    
    # Plot all walks
    plt.figure(figsize=(12, 8))
    
    # Plot each walk with a different color
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_walks)))
    
    for i, walk_points in enumerate(all_walks):
        if len(walk_points) > 1:
            walk_array = np.array(walk_points)
            plt.plot(walk_array[:, 0], walk_array[:, 1], 
                    color=colors[i], linewidth=1.5, alpha=0.8)
            
            # Mark start points
            plt.plot(walk_array[0, 0], walk_array[0, 1], 'o', 
                    color=colors[i], markersize=6, alpha=0.9)
    
    # Add letter outline for reference
    vertices = []
    codes = []
    for path in text_path.iter_segments():
        if len(path) == 2:  # Contains vertex and code
            vertex, code = path
            if len(vertex) == 2:
                vertices.append(vertex)
                codes.append(code)
    
    if vertices:
        vertices = np.array(vertices)
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2, alpha=0.4, label='Letter outline')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Random Walk Extraction for Letter "A"\\n{len(all_walks)} walks, {total_points} total points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_walks_standalone_A.png', dpi=150, bbox_inches='tight')
    print("Walk visualization saved as 'test_walks_standalone_A.png'")
    
    return all_walks

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    np.random.seed(42)
    test_extraction()
