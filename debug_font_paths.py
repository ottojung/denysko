#!/usr/bin/env python3
"""
Examine the font path codes and try different fonts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from matplotlib.path import Path

def examine_path_codes():
    """Examine the path codes to understand the structure."""
    print("Examining path codes and structure...")
    
    # Create path for letter A
    font_prop = font_manager.FontProperties(size=100)
    text_path = TextPath((0, 0), 'A', prop=font_prop)
    
    print(f"Font family: {font_prop.get_family()}")
    print(f"Font style: {font_prop.get_style()}")
    print(f"Font weight: {font_prop.get_weight()}")
    
    # Examine vertices and codes in detail
    vertices = text_path.vertices
    codes = text_path.codes
    
    print(f"\nPath details:")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Codes shape: {codes.shape}")
    
    # Path codes meaning:
    # 1 = MOVETO, 2 = LINETO, 3 = CURVE3, 4 = CURVE4, 79 = CLOSEPOLY
    code_names = {1: 'MOVETO', 2: 'LINETO', 3: 'CURVE3', 4: 'CURVE4', 79: 'CLOSEPOLY'}
    
    print(f"\nPath segments:")
    for i, (vertex, code) in enumerate(zip(vertices, codes)):
        code_name = code_names.get(code, f'UNKNOWN({code})')
        print(f"  {i:2d}: {vertex} - {code_name}")
    
    # Count MOVETO commands - should be 2 for letter A (outer + inner hole)
    moveto_count = np.sum(codes == 1)
    print(f"\nMOVETO commands: {moveto_count}")
    if moveto_count == 1:
        print("⚠️  Only 1 MOVETO found - this means no hole/inner path!")
    elif moveto_count == 2:
        print("✓ 2 MOVETO commands - should have outer shape + inner hole")
    else:
        print(f"⚠️  Unexpected number of MOVETO commands: {moveto_count}")
    
    # Try a different font that definitely should have a hole
    print("\n" + "="*50)
    print("Testing with different font (serif)...")
    
    try:
        # Try to get a serif font that's more likely to have holes
        serif_fonts = ['Times', 'Times New Roman', 'serif']
        for font_name in serif_fonts:
            try:
                serif_prop = font_manager.FontProperties(family=font_name, size=100)
                serif_path = TextPath((0, 0), 'A', prop=serif_prop)
                
                serif_codes = serif_path.codes
                serif_moveto_count = np.sum(serif_codes == 1)
                
                print(f"Font '{font_name}': {serif_moveto_count} MOVETO commands")
                
                if serif_moveto_count > 1:
                    print(f"✓ Found font with {serif_moveto_count} MOVETO commands!")
                    
                    # Test the center point with this font
                    serif_vertices = serif_path.vertices
                    serif_min_x, serif_min_y = serif_vertices.min(axis=0)
                    serif_max_x, serif_max_y = serif_vertices.max(axis=0)
                    center_x = (serif_min_x + serif_max_x) / 2
                    center_y = (serif_min_y + serif_max_y) / 2
                    
                    center_contains = serif_path.contains_point((center_x, center_y))
                    print(f"Center point ({center_x:.1f}, {center_y:.1f}): {center_contains}")
                    
                    if not center_contains:
                        print("✓ Center is NOT contained - hole detected!")
                        return serif_path, (serif_min_x, serif_min_y, serif_max_x, serif_max_y)
                    
                break
            except Exception as e:
                print(f"Failed to load font '{font_name}': {e}")
                continue
    except Exception as e:
        print(f"Error testing fonts: {e}")
    
    # Also try explicitly creating a path with a hole
    print("\n" + "="*50)
    print("Creating explicit path with hole...")
    
    # Create outer rectangle
    outer_verts = np.array([
        [10, 10],   # bottom-left
        [60, 10],   # bottom-right
        [60, 60],   # top-right
        [10, 60],   # top-left
        [10, 10]    # close
    ])
    
    # Create inner rectangle (hole)
    inner_verts = np.array([
        [25, 25],   # hole bottom-left
        [45, 25],   # hole bottom-right
        [45, 45],   # hole top-right
        [25, 45],   # hole top-left
        [25, 25]    # close hole
    ])
    
    # Combine vertices
    all_verts = np.vstack([outer_verts, inner_verts])
    
    # Create codes
    outer_codes = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    inner_codes = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    all_codes = np.concatenate([outer_codes, inner_codes])
    
    # Create path with hole
    path_with_hole = Path(all_verts, all_codes)
    
    # Test center point
    center_test = path_with_hole.contains_point((35, 35))  # Should be False (in hole)
    edge_test = path_with_hole.contains_point((15, 35))    # Should be True (in shape)
    
    print(f"Manual path with hole:")
    print(f"  Center (35, 35) in hole: {center_test}")
    print(f"  Edge (15, 35) in shape: {edge_test}")
    
    return text_path, None

if __name__ == "__main__":
    examine_path_codes()
