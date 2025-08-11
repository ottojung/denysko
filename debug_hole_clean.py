#!/usr/bin/env python3
"""
Debug the clean extractor to verify hole detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from clean_extractor import CenterlineExtractor


def debug_hole_detection():
    """Debug to verify the hole in letter A is properly detected."""
    print("Debugging hole detection...")
    
    # Create letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    extractor = CenterlineExtractor()
    bounds = extractor._get_path_bounds(text_path)
    mask = extractor._create_binary_mask(text_path, bounds, resolution=400)
    
    print(f"Bounds: {bounds}")
    print(f"Mask shape: {mask.shape}")
    print(f"Filled pixels: {np.sum(mask)} / {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
    
    # Test specific points
    min_x, min_y, max_x, max_y = bounds
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    test_points = [
        (center_x, center_y, "Center (should be in hole)"),
        (center_x, center_y - 20, "Lower center (should be solid)"),
        (center_x - 20, center_y + 20, "Left upper (should be solid)"),
        (center_x + 20, center_y + 20, "Right upper (should be solid)"),
        (min_x, min_y, "Bottom-left corner (should be empty)"),
        (max_x, max_y, "Top-right corner (should be empty)"),
    ]
    
    print("\\nTesting specific points:")
    for x, y, desc in test_points:
        inside_path = text_path.contains_point((x, y))
        inside_mask = extractor._is_point_inside((x, y), mask, bounds)
        status = "✓" if inside_path == inside_mask else "❌"
        print(f"  {desc}: path={inside_path}, mask={inside_mask} {status}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original letter path
    ax1 = axes[0]
    vertices = text_path.vertices
    ax1.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=2, label='Path outline')
    
    # Fill the path to show its shape
    from matplotlib.patches import PathPatch
    patch = PathPatch(text_path, facecolor='lightblue', alpha=0.5, edgecolor='blue')
    ax1.add_patch(patch)
    
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)
    ax1.set_aspect('equal')
    ax1.set_title('Original Letter Path')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Binary mask
    ax2 = axes[1]
    ax2.imshow(mask, extent=[min_x, max_x, min_y, max_y], origin='lower', 
              cmap='gray', interpolation='nearest')
    ax2.set_title(f'Binary Mask\\n{np.sum(mask)} filled pixels')
    ax2.set_aspect('equal')
    
    # Plot 3: Test points overlay
    ax3 = axes[2]
    ax3.imshow(mask, extent=[min_x, max_x, min_y, max_y], origin='lower', 
              cmap='gray', alpha=0.7, interpolation='nearest')
    
    # Overlay test points
    for x, y, desc in test_points:
        inside_path = text_path.contains_point((x, y))
        color = 'red' if inside_path else 'blue'
        marker = 'o' if inside_path else 'x'
        ax3.plot(x, y, marker=marker, color=color, markersize=8, markeredgewidth=2)
    
    ax3.set_title('Test Points Overlay\\n(red=inside, blue=outside)')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('debug_hole_detection.png', dpi=150, bbox_inches='tight')
    print("Debug visualization saved as 'debug_hole_detection.png'")
    
    # Count hole vs solid ratio
    hole_ratio = 1 - (np.sum(mask) / mask.size)
    print(f"\\nHole detection: {hole_ratio*100:.1f}% of area is holes/empty")
    
    return mask


if __name__ == "__main__":
    debug_hole_detection()
