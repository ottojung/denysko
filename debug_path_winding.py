#!/usr/bin/env python3
"""
Test path winding direction for holes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def test_path_winding():
    """Test different winding directions for paths with holes."""
    print("Testing path winding directions...")
    
    # Create outer rectangle (counter-clockwise)
    outer_ccw = np.array([
        [10, 10],   # bottom-left
        [60, 10],   # bottom-right  
        [60, 60],   # top-right
        [10, 60],   # top-left
        [10, 10]    # close
    ])
    
    # Create inner rectangle - CLOCKWISE (opposite winding for hole)
    inner_cw = np.array([
        [25, 25],   # hole bottom-left
        [25, 45],   # hole top-left  (going clockwise)
        [45, 45],   # hole top-right
        [45, 25],   # hole bottom-right
        [25, 25]    # close hole
    ])
    
    # Combine vertices
    all_verts = np.vstack([outer_ccw, inner_cw])
    
    # Create codes
    outer_codes = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    inner_codes = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    all_codes = np.concatenate([outer_codes, inner_codes])
    
    # Create path with proper winding
    path_proper_winding = Path(all_verts, all_codes)
    
    # Test center point (should be False - in hole)
    center_test = path_proper_winding.contains_point((35, 35))
    edge_test = path_proper_winding.contains_point((15, 35))  
    
    print(f"Path with proper winding (outer CCW, inner CW):")
    print(f"  Center (35, 35) in hole: {center_test}")
    print(f"  Edge (15, 35) in shape: {edge_test}")
    
    # Also test the contains_points method for a grid
    resolution = 100
    x = np.linspace(0, 70, resolution)
    y = np.linspace(0, 70, resolution) 
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    contains_grid = path_proper_winding.contains_points(points)
    mask = contains_grid.reshape((resolution, resolution))
    
    filled_pixels = np.sum(mask)
    total_pixels = resolution * resolution
    print(f"Grid test: {filled_pixels}/{total_pixels} pixels filled ({filled_pixels/total_pixels*100:.1f}%)")
    
    # Visualize the result
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Path visualization
    plt.subplot(1, 2, 1)
    
    # Draw outer path
    plt.plot(outer_ccw[:, 0], outer_ccw[:, 1], 'b-', linewidth=2, label='Outer (CCW)')
    plt.plot(inner_cw[:, 0], inner_cw[:, 1], 'r-', linewidth=2, label='Inner (CW)')
    
    # Mark test points
    plt.plot(35, 35, 'ro', markersize=8, label=f'Center: {center_test}')
    plt.plot(15, 35, 'go', markersize=8, label=f'Edge: {edge_test}')
    
    plt.xlim(0, 70)
    plt.ylim(0, 70)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Path with Proper Winding')
    plt.axis('equal')
    
    # Plot 2: Rasterized result
    plt.subplot(1, 2, 2)
    plt.imshow(mask, extent=[0, 70, 0, 70], origin='lower', cmap='gray')
    plt.title(f'Rasterized Result\\n{filled_pixels} filled pixels')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('debug_path_winding.png', dpi=150, bbox_inches='tight')
    print("Path winding test saved as 'debug_path_winding.png'")
    
    return path_proper_winding, mask

if __name__ == "__main__":
    test_path_winding()
