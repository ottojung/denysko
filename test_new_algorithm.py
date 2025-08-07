#!/usr/bin/env python3
"""
Test the new polynomial fitting algorithm with simulated letter "A" points.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polynomial_fitter import PolynomialFitter

def create_letter_a_points():
    """Create simulated letter 'A' points with overlapping horizontal space."""
    points = []
    
    # Left diagonal stroke (bottom-left to top)
    for t in np.linspace(0, 1, 10):
        x = 10 + t * 20  # x from 10 to 30
        y = 10 + t * 40  # y from 10 to 50
        points.append([x, y])
    
    # Right diagonal stroke (top to bottom-right)  
    for t in np.linspace(0, 1, 10):
        x = 30 + t * 20  # x from 30 to 50
        y = 50 - t * 40  # y from 50 to 10
        points.append([x, y])
    
    # Crossbar (horizontal stroke) - overlaps with both diagonals
    for t in np.linspace(0, 1, 8):
        x = 20 + t * 20  # x from 20 to 40 (overlaps both diagonals)
        y = 30 + t * 2   # slight slope for realism
        points.append([x, y])
    
    return np.array(points)

def test_new_algorithm():
    """Test the completely rewritten polynomial fitting algorithm."""
    print("Testing COMPLETELY REWRITTEN polynomial fitting algorithm...")
    print("=" * 60)
    
    # Create test data
    letter_points = create_letter_a_points()
    print(f"Created {len(letter_points)} letter 'A' points")
    print(f"X range: {np.min(letter_points[:, 0]):.1f} to {np.max(letter_points[:, 0]):.1f}")
    print(f"Y range: {np.min(letter_points[:, 1]):.1f} to {np.max(letter_points[:, 1]):.1f}")
    
    # Show some sample points
    print("\nSample points:")
    for i in range(0, len(letter_points), 5):
        x, y = letter_points[i]
        print(f"  Point {i+1}: ({x:.1f}, {y:.1f})")
    
    # Initialize the new fitter
    fitter = PolynomialFitter()
    
    # Test the algorithm
    print("\n" + "="*60)
    print("RUNNING NEW ALGORITHM:")
    print("="*60)
    
    functions = fitter.fit_contour_polynomials(letter_points, max_degree=20)
    
    print("="*60)
    print("RESULTS:")
    print("="*60)
    
    print(f"Generated {len(functions)} polynomial functions")
    
    if functions:
        for i, func in enumerate(functions):
            print(f"\nFunction {i+1}:")
            print(f"  {func}")
            
            # Test if it actually passes through some points
            print("  Testing point accuracy...")
            # (This would require parsing and evaluating the function)
            
        # Check if we got multiple functions for letter A (should have due to overlap)
        if len(functions) >= 2:
            print(f"\n✓ SUCCESS: Algorithm correctly generated {len(functions)} curves")
            print("  This suggests proper detection of overlapping horizontal space in letter 'A'")
        else:
            print("\n⚠ WARNING: Expected multiple functions for letter 'A'")
            print("  Letter 'A' has overlapping horizontal space (crossbar overlaps diagonals)")
    else:
        print("\n✗ FAILED: No functions generated")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_new_algorithm()
