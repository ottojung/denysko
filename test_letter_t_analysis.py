#!/usr/bin/env python3
"""
Test for letter "T" polynomial fitting coverage.

Letter T has a unique structure:
- Horizontal bar at the top
- Vertical stem in the middle
- This creates a cross/T shape that might be challenging for y = f(x) polynomials
"""

import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_extractor import TextExtractor
from src.polynomial_fitter_genetic import PolynomialFitter
from src.genetic_polynomial_fitter import GeneticPolynomialFitter, Polynomial


def analyze_letter_t():
    """Analyze the structure of letter T to understand fitting challenges."""
    print("=== Letter T Structure Analysis ===")
    
    # Generate letter T points
    extractor = TextExtractor()
    paths = extractor.text_to_paths("T", font_size=100)
    
    all_points = []
    for path in paths:
        contours = extractor.extract_contour_points(path, num_points=1000)
        for contour in contours:
            if hasattr(contour, "__len__") and len(contour) > 1:
                all_points.extend(contour)
    
    if not all_points:
        print("ERROR: No points extracted for letter T")
        return None
    
    letter_t_points = np.array(all_points)
    print(f"Extracted {len(letter_t_points)} points for letter T")
    
    # Analyze the structure
    x_coords = letter_t_points[:, 0]
    y_coords = letter_t_points[:, 1]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_span = x_max - x_min
    y_span = y_max - y_min
    
    print(f"Bounding box: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    print(f"Dimensions: width={x_span:.2f}, height={y_span:.2f}")
    
    # Analyze y-variation for different x-regions
    print("\n=== Y-variation analysis (key challenge for y=f(x)) ===")
    
    # Divide x-range into regions and check y-variation
    num_regions = 10
    for i in range(num_regions):
        x_start = x_min + (i / num_regions) * x_span
        x_end = x_min + ((i + 1) / num_regions) * x_span
        x_mid = (x_start + x_end) / 2
        
        # Find points in this x-region
        mask = (x_coords >= x_start) & (x_coords <= x_end)
        region_y = y_coords[mask]
        
        if len(region_y) > 0:
            y_var = np.max(region_y) - np.min(region_y)
            y_range = f"[{np.min(region_y):.1f}, {np.max(region_y):.1f}]"
            print(f"  x≈{x_mid:.1f}: {len(region_y):3d} points, y-variation={y_var:.1f}, y-range={y_range}")
            
            # Flag problematic regions (high y-variation = not a function)
            if y_var > y_span * 0.3:  # More than 30% of total height
                print(f"    ⚠️  HIGH Y-VARIATION: This x-region is problematic for y=f(x)")
    
    # Analyze specific structural features of T
    print("\n=== Letter T structural features ===")
    
    # Top horizontal bar: should have points across most of x-range at high y
    top_y_threshold = y_min + 0.8 * y_span  # Top 20% of y-range
    top_points = letter_t_points[y_coords >= top_y_threshold]
    
    if len(top_points) > 0:
        top_x_span = np.max(top_points[:, 0]) - np.min(top_points[:, 0])
        print(f"Top bar: {len(top_points)} points, x-span={top_x_span:.1f} ({top_x_span/x_span:.1%} of total width)")
    
    # Vertical stem: should be in middle x, spanning most of y
    middle_x = x_min + 0.5 * x_span
    stem_width = 0.2 * x_span  # Allow 20% width for stem
    stem_mask = (x_coords >= middle_x - stem_width/2) & (x_coords <= middle_x + stem_width/2)
    stem_points = letter_t_points[stem_mask]
    
    if len(stem_points) > 0:
        stem_y_span = np.max(stem_points[:, 1]) - np.min(stem_points[:, 1])
        print(f"Vertical stem: {len(stem_points)} points, y-span={stem_y_span:.1f} ({stem_y_span/y_span:.1%} of total height)")
    
    return letter_t_points


def test_letter_t_fitting():
    """Test polynomial fitting on letter T with different parameters."""
    letter_t_points = analyze_letter_t()
    
    if letter_t_points is None:
        return
    
    print("\n=== Testing different fitting approaches ===")
    
    # Test 1: Current approach (2 polynomials, degree 5)
    print("\n1. Current approach (2 polynomials, max degree 5):")
    fitter1 = GeneticPolynomialFitter(
        population_size=50,
        generations=50,
        max_polynomials=2,
        max_degree=5,
        mutation_rate=0.05,
        tournament_size=5
    )
    
    # Sample points for faster testing
    if len(letter_t_points) > 200:
        indices = np.linspace(0, len(letter_t_points)-1, 200, dtype=int)
        test_points = letter_t_points[indices]
    else:
        test_points = letter_t_points
    
    result1 = fitter1.fit(test_points)
    print(f"  Result: {len(result1)} polynomials, degrees: {[p.degree for p in result1]}")
    
    # Test coverage
    covered_points = 0
    for x, y in test_points:
        min_error = float('inf')
        for poly in result1:
            pred = poly.evaluate(x)
            if pred is not None:
                error = abs(pred - y)
                min_error = min(min_error, error)
        
        if min_error <= 10.0:  # Same tolerance as letter A
            covered_points += 1
    
    coverage = covered_points / len(test_points)
    print(f"  Coverage: {coverage:.1%} ({covered_points}/{len(test_points)} points)")
    
    # Test 2: More polynomials for complex T shape
    print("\n2. More polynomials approach (4 polynomials, max degree 4):")
    fitter2 = GeneticPolynomialFitter(
        population_size=50,
        generations=50,
        max_polynomials=4,  # More polynomials to handle T complexity
        max_degree=4,
        mutation_rate=0.05,
        tournament_size=5
    )
    
    result2 = fitter2.fit(test_points)
    print(f"  Result: {len(result2)} polynomials, degrees: {[p.degree for p in result2]}")
    
    # Test coverage
    covered_points = 0
    for x, y in test_points:
        min_error = float('inf')
        for poly in result2:
            pred = poly.evaluate(x)
            if pred is not None:
                error = abs(pred - y)
                min_error = min(min_error, error)
        
        if min_error <= 10.0:
            covered_points += 1
    
    coverage = covered_points / len(test_points)
    print(f"  Coverage: {coverage:.1%} ({covered_points}/{len(test_points)} points)")
    
    # Show sample predictions for debugging
    print(f"\n=== Sample predictions (first 5 points) ===")
    for i, (x, y) in enumerate(test_points[:5]):
        print(f"Point {i}: x={x:.2f}, y_actual={y:.2f}")
        for j, poly in enumerate(result2):
            pred = poly.evaluate(x)
            if pred is not None:
                error = abs(pred - y)
                print(f"  Poly {j}: pred={pred:.2f}, error={error:.2f}")


if __name__ == "__main__":
    test_letter_t_fitting()
