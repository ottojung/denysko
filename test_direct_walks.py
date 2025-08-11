#!/usr/bin/env python3
"""
Direct test of the centerline extraction.
"""

import sys
sys.path.append('/media/mybtrfs/home-submodule/my-link-files/root/home/user1/.local/share/miyka/root/repositories/gcvx2dldwd1sbp40/wd/home/my/project/main-repo/src')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from centerline_extraction import extract_skeleton_from_path

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
    
    print(f"Total points across all walks: {total_points}")
    print(f"Average points per walk: {total_points / len(all_walks):.1f}")
    
    # Plot all walks
    plt.figure(figsize=(10, 8))
    
    # Plot each walk with a different color
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_walks)))
    
    for i, walk_points in enumerate(all_walks):
        if len(walk_points) > 1:
            walk_array = np.array(walk_points)
            plt.plot(walk_array[:, 0], walk_array[:, 1], 
                    color=colors[i], linewidth=1.5, alpha=0.8)
            
            # Mark start and end points
            plt.plot(walk_array[0, 0], walk_array[0, 1], 'o', 
                    color=colors[i], markersize=4, alpha=0.8)
            plt.plot(walk_array[-1, 0], walk_array[-1, 1], 's', 
                    color=colors[i], markersize=4, alpha=0.8)
    
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
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=0.5, alpha=0.3, label='Letter outline')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Random Walk Extraction for Letter "A"\\n{len(all_walks)} walks, {total_points} total points')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_walks_direct_A.png', dpi=150, bbox_inches='tight')
    print("Walk visualization saved as 'test_walks_direct_A.png'")

if __name__ == "__main__":
    test_extraction()
