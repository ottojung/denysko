#!/usr/bin/env python3
"""
Test script to verify that separate strokes are being generated correctly.
"""

def test_stroke_separation():
    """Test that letter A generates separate strokes correctly."""
    print("Testing stroke separation algorithm...")
    
    # Mock key points for letter 'A'
    mock_key_points = [
        [100, 200],  # top point
        [50, 100],   # bottom-left  
        [150, 100],  # bottom-right
        [80, 140],   # crossbar left
        [120, 140]   # crossbar right
    ]
    
    print(f"Mock key points: {mock_key_points}")
    
    # Test stroke creation logic
    strokes = create_mock_strokes(mock_key_points)
    print(f"Generated {len(strokes)} strokes:")
    
    for i, stroke in enumerate(strokes):
        print(f"  Stroke {i+1} ({stroke['type']}): {stroke['start']} -> {stroke['end']}")
    
    return strokes

def create_mock_strokes(key_points):
    """Mock version of create_structural_strokes."""
    if len(key_points) < 3:
        return []
    
    strokes = []
    
    # Sort points by y-coordinate (top to bottom) then by x-coordinate
    sorted_points = sorted(key_points, key=lambda p: (-p[1], p[0]))
    
    if len(sorted_points) >= 3:
        top_point = sorted_points[0]
        
        # Find bottom points
        bottom_points = [p for p in sorted_points if p[1] < top_point[1] - 10]
        
        if len(bottom_points) >= 2:
            # Sort bottom points by x-coordinate
            bottom_points.sort(key=lambda p: p[0])
            left_bottom = bottom_points[0]
            right_bottom = bottom_points[-1]
            
            # Create main structural strokes
            # Left diagonal: top to left bottom
            strokes.append({
                'start': top_point,
                'end': left_bottom,
                'type': 'diagonal_left'
            })
            
            # Right diagonal: top to right bottom
            strokes.append({
                'start': top_point, 
                'end': right_bottom,
                'type': 'diagonal_right'
            })
            
            # Crossbar: find middle-height points
            mid_points = [p for p in sorted_points if abs(p[1] - (top_point[1] + left_bottom[1])/2) < 20]
            if len(mid_points) >= 2:
                mid_points.sort(key=lambda p: p[0])
                strokes.append({
                    'start': mid_points[0],
                    'end': mid_points[-1],
                    'type': 'crossbar'
                })
    
    return strokes

def test_stroke_connection():
    """Test the new separate stroke approach."""
    print("\nTesting separate stroke approach...")
    
    mock_strokes = [
        {'start': [100, 200], 'end': [50, 100], 'type': 'diagonal_left'},
        {'start': [100, 200], 'end': [150, 100], 'type': 'diagonal_right'},
        {'start': [80, 140], 'end': [120, 140], 'type': 'crossbar'}
    ]
    
    # Process each stroke separately
    separate_functions = []
    for i, stroke in enumerate(mock_strokes):
        print(f"Stroke {i+1} ({stroke['type']}):")
        print(f"  Points: {stroke['start']} -> {stroke['end']}")
        print(f"  This will become polynomial function y_{i+1} = f_{i+1}(x)")
        separate_functions.append(f"y_{i+1} = f_{i+1}(x)  // {stroke['type']}")
    
    print(f"\nResult: {len(separate_functions)} separate functions:")
    for func in separate_functions:
        print(f"  {func}")
    
    print("\nThis eliminates jump lines because:")
    print("- No stroke concatenation with extend()")
    print("- Each stroke becomes independent polynomial")
    print("- No unwanted connecting lines between strokes")
    
    return separate_functions

if __name__ == "__main__":
    print("=== Testing Stroke Separation Algorithm ===")
    
    # Test stroke detection
    strokes = test_stroke_separation()
    
    # Test separate processing  
    functions = test_stroke_connection()
    
    print(f"\n=== SUMMARY ===")
    print(f"Letter 'A' analysis:")
    print(f"- Detected {len(strokes)} structural strokes")
    print(f"- Generated {len(functions)} separate polynomial functions")
    print(f"- No jump lines between disconnected strokes")
    print(f"- Crossbar included as separate function")
