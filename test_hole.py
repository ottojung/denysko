#!/usr/bin/env python3
"""
Test hole detection in letter A.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from src.text_extractor import TextExtractor

def test_hole_detection():
    """Test if the path for letter A properly represents a hole."""
    extractor = TextExtractor()
    paths = extractor.text_to_paths("A", font_size=100)
    
    if not paths:
        print("No paths generated")
        return
        
    path = paths[0]
    print(f"Path codes: {path.codes}")
    print(f"Path vertices count: {len(path.vertices)}")
    
    # Check winding order by analyzing the codes
    moveto_count = np.sum(path.codes == 1) if path.codes is not None else 0  # MOVETO
    print(f"Number of MOVETO commands: {moveto_count}")
    
    # Test specific points
    # Center of A hole should be outside
    center_x, center_y = np.mean(path.vertices, axis=0)
    hole_center = (center_x, center_y - 10)  # Move down into potential hole area
    
    print(f"Letter center: ({center_x:.2f}, {center_y:.2f})")
    print(f"Testing hole center: ({hole_center[0]:.2f}, {hole_center[1]:.2f})")
    
    # Test with contains_point
    is_inside = path.contains_point(hole_center)
    print(f"Hole center is inside path: {is_inside}")
    
    # Test with contains_points
    test_points = np.array([hole_center])
    is_inside_array = path.contains_points(test_points)
    print(f"Hole center is inside path (array): {is_inside_array[0]}")
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Draw letter outline
    patch = PathPatch(path, facecolor='lightgray', edgecolor='black', alpha=0.3, linewidth=1)
    ax.add_patch(patch)
    
    # Mark the test point
    ax.plot(hole_center[0], hole_center[1], 'ro', markersize=8, label='Test point (hole center)')
    
    # Set axis bounds
    vertices = path.vertices
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    padding = 0.1 * max(max_x - min_x, max_y - min_y)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    ax.set_title('Letter A - Hole Detection Test')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.set_aspect('equal')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('hole_test_A.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Hole test visualization saved as 'hole_test_A.png'")

if __name__ == "__main__":
    test_hole_detection()
