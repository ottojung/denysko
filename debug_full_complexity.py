#!/usr/bin/env python3
"""Debug complexity analysis using full point sets before sampling."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_extractor import TextExtractor

def analyze_full_letter_complexity(letter):
    """Analyze complexity using the full point set before sampling."""
    print(f"=== Full Complexity Analysis for Letter {letter} ===")
    
    extractor = TextExtractor()
    
    # Extract letter paths and full point set
    paths = extractor.text_to_paths(letter, 100)
    if paths:
        contours = extractor.extract_contour_points(paths[0], 10000)  # Get many more points
        if contours:
            data_points = contours[0]
            print(f"Letter {letter} has {len(data_points)} full points")
            
            # Extract coordinates
            x_coords = [p[0] for p in data_points]
            y_coords = [p[1] for p in data_points]
            
            x_span = max(x_coords) - min(x_coords)
            y_span = max(y_coords) - min(y_coords)
            
            point_density = len(data_points) / (x_span * y_span) if (x_span * y_span) > 0 else 0
            compactness_ratio = y_span / x_span
            
            print("Full metrics:")
            print(f"  Points: {len(data_points)}")
            print(f"  X span: {x_span:.1f}, Y span: {y_span:.1f}")
            print(f"  Density: {point_density:.1f}")
            print(f"  Compactness: {compactness_ratio:.2f}")
            
            # Classification
            if point_density > 800 and compactness_ratio > 1.4:
                classification = "Complex (5 polynomials)"
            elif point_density > 600 or len(data_points) > 25000:
                classification = "Medium (3 polynomials)"
            else:
                classification = "Simple (2 polynomials)"
                
            print(f"  Classification: {classification}")
            return len(data_points), point_density, compactness_ratio

if __name__ == "__main__":
    for letter in ['A', 'B', 'C']:
        analyze_full_letter_complexity(letter)
        print()
