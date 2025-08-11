#!/usr/bin/env python3
"""
Debug script to visualize boundary checking issues.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
import matplotlib

# Use non-interactive backend if no display available
import os
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

from src.text_extractor import TextExtractor
from src.path_processing import rasterize_path


def debug_boundary_checking(text="A"):
    """Debug boundary checking by visualizing the mask and walk paths together."""
    print(f"Debugging boundary checking for letter '{text}'...")
    
    extractor = TextExtractor()
    paths = extractor.text_to_paths(text, font_size=100)
    
    if not paths:
        print("No paths generated")
        return
        
    path = paths[0]  # First character
    
    # Get the same rasterization as the extractor
    mask, x_grid, y_grid = rasterize_path(path, resolution=400)
    print(f"Mask shape: {mask.shape}, filled pixels: {mask.sum()}")
    
    # Get the clean mask from our implementation
    from src.centerline_extraction import _create_clean_mask
    vertices = path.vertices
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    bounds = (min_x, max_x, min_y, max_y)
    clean_mask = _create_clean_mask(path, bounds)
    print(f"Clean mask: {clean_mask.shape}, filled pixels: {clean_mask.sum()}")
    
    # Get walk paths from extractor
    walks = extractor.extract_skeleton_from_path(path)
    print(f"Generated {len(walks)} walks")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left plot: Show the rasterized mask
    ax1.imshow(mask, cmap='gray', extent=[x_grid[0,0], x_grid[0,-1], y_grid[-1,0], y_grid[0,0]])
    ax1.set_title(f"Rasterized mask for '{text}'")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    
    # Right plot: Show walks on top of letter outline
    # Draw letter outline
    patch = PathPatch(path, facecolor='lightgray', edgecolor='black', alpha=0.3, linewidth=1)
    ax2.add_patch(patch)
    
    # Set axis bounds from path vertices
    vertices = path.vertices
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    padding = 0.05 * max(max_x - min_x, max_y - min_y)
    ax2.set_xlim(min_x - padding, max_x + padding)
    ax2.set_ylim(min_y - padding, max_y + padding)
    
    # Draw all walks
    if walks:
        segments = [walk for walk in walks if len(walk) >= 2]
        if segments:
            cmap = plt.cm.get_cmap("tab20")
            colors = [cmap(i % 20) for i in range(len(segments))]
            lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.8)
            ax2.add_collection(lc)
    
    ax2.set_title(f"Walk paths for '{text}'")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"debug_boundary_{text}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug visualization saved as 'debug_boundary_{text}.png'")
    
    # Additional check: verify all walk points are within the clean mask
    total_violations = 0
    for i, walk in enumerate(walks):
        violations = 0
        for point in walk:
            # Check each point against the clean mask using the same method as our implementation
            x, y = point
            
            # Get bounds from clean mask
            vertices = path.vertices
            min_x, min_y = np.min(vertices, axis=0)
            max_x, max_y = np.max(vertices, axis=0)
            h, w = clean_mask.shape
            
            if min_x <= x <= max_x and min_y <= y <= max_y:
                col = int((x - min_x) / (max_x - min_x) * (w - 1))
                row = int((y - min_y) / (max_y - min_y) * (h - 1))
                col = max(0, min(w - 1, col))
                row = max(0, min(h - 1, row))
                
                if not clean_mask[row, col]:
                    violations += 1
        
        if violations > 0:
            print(f"Walk {i}: {violations} boundary violations out of {len(walk)} points ({100*violations/len(walk):.1f}%)")
            total_violations += violations
    
    if total_violations == 0:
        print("✅ No boundary violations detected!")
    else:
        print(f"❌ Total boundary violations: {total_violations}")


if __name__ == "__main__":
    debug_boundary_checking("A")
