#!/usr/bin/env python3
"""
Quick test of the new geometric stroke extraction approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_extractor import TextExtractor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def test_geometric_extraction():
    """Test the new geometric approach on letter 'A'"""
    
    print("=== TESTING GEOMETRIC STROKE EXTRACTION ===")
    
    extractor = TextExtractor()
    
    # Test on letter 'A' 
    text = "A"
    print(f"\nTesting letter: '{text}'")
    
    # Get paths
    paths = extractor.text_to_paths(text, font_size=100)
    
    if not paths:
        print("ERROR: No paths generated!")
        return
    
    path = paths[0]  # First character
    print(f"Original path has {len(path.vertices)} vertices")
    
    # Test the new geometric extraction
    try:
        skeleton_points = extractor.extract_skeleton_from_path(path)
        print(f"Geometric extraction produced {len(skeleton_points)} points")
        
        # Show first few points
        if len(skeleton_points) > 0:
            print(f"First 5 points: {skeleton_points[:5]}")
            
            # Check if points form reasonable strokes
            if len(skeleton_points) > 20:
                print("SUCCESS: Generated sufficient points for letter structure")
            else:
                print("WARNING: Very few points generated")
        else:
            print("ERROR: No skeleton points generated!")
            
    except Exception as e:
        print(f"ERROR in geometric extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_geometric_extraction()
