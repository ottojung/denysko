#!/usr/bin/env python3
"""
Quick test of the new structural stroke extraction approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from text_extractor import TextExtractor
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_LIBS = True
except ImportError as e:
    print(f"Import error: {e}")
    HAS_LIBS = False

def test_structural_extraction():
    """Test the new structural approach on letter 'A'"""
    
    if not HAS_LIBS:
        print("Required libraries not available, cannot run test")
        return
    
    print("=== TESTING STRUCTURAL STROKE EXTRACTION ===")
    
    extractor = TextExtractor()
    
    # Test on letter 'A' 
    text = "A"
    print(f"\nTesting letter: '{text}'")
    
    try:
        # Get paths
        paths = extractor.text_to_paths(text, font_size=100)
        
        if not paths:
            print("ERROR: No paths generated!")
            return
        
        path = paths[0]  # First character
        print(f"Original path has {len(path.vertices)} vertices")
        
        # Test the key point detection
        vertices = path.vertices
        key_points = extractor.find_letter_key_points(vertices)
        print(f"Key points found: {len(key_points)}")
        
        for i, point in enumerate(key_points):
            print(f"  Point {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
        
        # Test stroke creation
        if len(key_points) >= 3:
            strokes = extractor.create_structural_strokes(key_points)
            print(f"Structural strokes created: {len(strokes)}")
            
            for i, stroke in enumerate(strokes):
                start = stroke['start']
                end = stroke['end']
                stroke_type = stroke['type']
                length = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
                print(f"  Stroke {i+1} ({stroke_type}): ({start[0]:.1f},{start[1]:.1f}) -> ({end[0]:.1f},{end[1]:.1f}) [length: {length:.1f}]")
        
        # Test full skeleton extraction
        skeleton_points = extractor.extract_skeleton_from_path(path)
        print(f"Final skeleton: {len(skeleton_points)} points")
        
        if len(skeleton_points) > 0:
            print("SUCCESS: Structural extraction completed!")
            
            # Show first few points as sample
            print("Sample points:", skeleton_points[:3].tolist() if len(skeleton_points) >= 3 else skeleton_points.tolist())
        else:
            print("ERROR: No skeleton points generated!")
            
    except Exception as e:
        print(f"ERROR in structural extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_structural_extraction()
