#!/usr/bin/env python3
"""
Test different rasterization methods and examine the path structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath

def test_path_methods():
    """Test different methods of working with the path."""
    print("Testing different path methods...")
    
    # Create path for letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    # Examine the path structure
    print("\nPath structure:")
    print(f"Number of vertices: {len(text_path.vertices)}")
    print(f"Number of codes: {len(text_path.codes)}")
    print(f"Path codes: {set(text_path.codes)}")
    
    # Get bounds
    vertices = text_path.vertices
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    bounds = (min_x, min_y, max_x, max_y)
    print(f"Bounds: {bounds}")
    
    # Test the old contains_points method
    print("\nTesting path.contains_points method:")
    resolution = 400
    
    # Create grid of points
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Test contains_points
    contains = text_path.contains_points(points)
    mask_old = contains.reshape((resolution, resolution))
    
    print(f"Old method filled pixels: {np.sum(mask_old)}")
    
    # Test some specific points manually
    test_points = [
        (0, 0),  # Corner
        (33.79, 36.45),  # Center - should be False for A hole
        (33.79, 20),  # Lower center - should be True
        (20, 50),  # Left side - should be True
        (50, 50),  # Right side - should be True
    ]
    
    for x_test, y_test in test_points:
        single_contains = text_path.contains_point((x_test, y_test))
        print(f"  Point ({x_test:.2f}, {y_test:.2f}): {single_contains}")
    
    # Create visualization comparing methods
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original path outline
    ax1 = axes[0]
    vertices_list = []
    for path_data in text_path.iter_segments():
        if len(path_data) == 2:
            vertex, code = path_data
            if len(vertex) == 2:
                vertices_list.append(vertex)
    
    if vertices_list:
        vertices_array = np.array(vertices_list)
        ax1.plot(vertices_array[:, 0], vertices_array[:, 1], 'b-', linewidth=2)
        # Also try to show the filled path
        ax1.add_patch(plt.matplotlib.patches.PathPatch(text_path, facecolor='lightblue', alpha=0.5))
    
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_aspect('equal')
    ax1.set_title('Original Path')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: contains_points result
    ax2 = axes[1]
    ax2.imshow(mask_old, extent=[min_x, max_x, min_y, max_y], 
              origin='lower', cmap='gray')
    ax2.set_title(f'contains_points\\n({np.sum(mask_old)} filled pixels)')
    ax2.set_aspect('equal')
    
    # Plot 3: Test specific points overlay
    ax3 = axes[2]
    ax3.imshow(mask_old, extent=[min_x, max_x, min_y, max_y], 
              origin='lower', cmap='gray', alpha=0.7)
    
    # Overlay test points
    for i, (x_test, y_test) in enumerate(test_points):
        contains_val = text_path.contains_point((x_test, y_test))
        color = 'red' if contains_val else 'blue'
        ax3.plot(x_test, y_test, 'o', color=color, markersize=8, 
                label=f'Point {i+1}: {contains_val}')
    
    ax3.set_title('Test Points Overlay')
    ax3.set_aspect('equal')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('debug_path_methods_A.png', dpi=150, bbox_inches='tight')
    print("Path methods comparison saved as 'debug_path_methods_A.png'")
    
    return mask_old

if __name__ == "__main__":
    test_path_methods()
