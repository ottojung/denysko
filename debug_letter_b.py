#!/usr/bin/env python3
"""Debug complexity analysis for letter B."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_extractor import TextExtractor

def debug_letter_b_complexity():
    """Debug the complexity analysis for letter B."""
    print("=== Debugging Letter B Complexity Analysis ===")
    
    extractor = TextExtractor()
    
    # Extract letter B points
    paths = extractor.text_to_paths("B", 100)
    if paths:
        contours = extractor.extract_contour_points(paths[0], 500)
        if contours:
            data_points = contours[0]
            print(f"Letter B has {len(data_points)} points")
            
            # Analyze complexity manually
            x_coords = [p[0] for p in data_points]
            y_coords = [p[1] for p in data_points]
            
            x_span = max(x_coords) - min(x_coords)
            y_span = max(y_coords) - min(y_coords)
            
            print(f"X span: {x_span:.1f}")
            print(f"Y span: {y_span:.1f}")
            
            # Check middle region analysis
            x_min = min(x_coords)
            middle_x = x_min + 0.5 * x_span
            middle_width = 0.3 * x_span
            
            middle_points = [(x, y) for x, y in data_points 
                            if middle_x - middle_width/2 <= x <= middle_x + middle_width/2]
            
            print(f"Middle region points: {len(middle_points)}")
            
            if len(middle_points) > 20:
                middle_y_vals = [y for x, y in middle_points]
                middle_y_variation = max(middle_y_vals) - min(middle_y_vals)
                complexity_ratio = middle_y_variation / y_span
                
                print(f"Middle Y variation: {middle_y_variation:.1f}")
                print(f"Complexity ratio: {complexity_ratio:.3f}")
                
                if complexity_ratio > 0.6:
                    print("Classification: Complex letter (should allow 5 polynomials)")
                elif complexity_ratio > 0.3:
                    print("Classification: Moderate complexity (should allow 3 polynomials)")
                else:
                    print("Classification: Simple letter (should allow 2 polynomials)")
            else:
                print("Not enough middle points for analysis")

if __name__ == "__main__":
    debug_letter_b_complexity()
