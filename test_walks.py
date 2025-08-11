#!/usr/bin/env python3
"""
Quick test of the random walk extraction.
"""

import sys
import os
sys.path.append('/media/mybtrfs/home-submodule/my-link-files/root/home/user1/.local/share/miyka/root/repositories/gcvx2dldwd1sbp40/wd/home/my/project/main-repo/src')

import matplotlib.pyplot as plt
import numpy as np
from text_extractor import TextExtractor

def test_extraction():
    """Test the random walk extraction."""
    print("Testing random walk extraction...")
    
    # Extract paths for letter A
    extractor = TextExtractor(font_size=100)
    all_walks = extractor.extract_text("A")
    
    print(f"Extracted {len(all_walks)} walks for letter 'A'")
    
    # Show statistics for each walk
    for i, walk_points in enumerate(all_walks):
        print(f"Walk {i+1}: {len(walk_points)} points")
    
    # Plot all walks
    plt.figure(figsize=(10, 8))
    
    # Plot each walk with a different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_walks)))
    
    for i, walk_points in enumerate(all_walks):
        if len(walk_points) > 1:
            walk_array = np.array(walk_points)
            plt.plot(walk_array[:, 0], walk_array[:, 1], 
                    color=colors[i], linewidth=2, alpha=0.7,
                    label=f'Walk {i+1} ({len(walk_points)} pts)')
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Random Walk Extraction for Letter "A" ({len(all_walks)} walks)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Only show legend if we have few walks
    if len(all_walks) <= 10:
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_walks_A.png', dpi=150, bbox_inches='tight')
    print("Walk visualization saved as 'test_walks_A.png'")

if __name__ == "__main__":
    test_extraction()
