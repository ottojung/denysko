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
    """Test the new polynomial fitting algorithm."""
    print("Testing new polynomial fitting algorithm...")
    
    # Create test data
    letter_points = create_letter_a_points()
    print(f"Created {len(letter_points)} letter 'A' points")
    print(f"X range: {np.min(letter_points[:, 0]):.1f} to {np.max(letter_points[:, 0]):.1f}")
    print(f"Y range: {np.min(letter_points[:, 1]):.1f} to {np.max(letter_points[:, 1]):.1f}")
    
    # Initialize fitter
    fitter = PolynomialFitter(max_degree=15)
    
    # Test the new algorithm
    print("\n" + "="*50)
    functions = fitter.fit_contour_polynomials(letter_points, max_degree=15)
    print("="*50)
    
    print("\nResults:")
    print(f"Generated {len(functions)} polynomial functions")
    
    if functions:
        for i, func in enumerate(functions):
            print(f"Function {i+1}: {func}")
            
        # Check if we got multiple functions for letter A (should have due to overlap)
        if len(functions) >= 2:
            print("\n✓ SUCCESS: Algorithm correctly detected overlapping strokes")
        else:
            print("\n⚠ WARNING: Expected multiple functions for letter 'A' due to overlapping horizontal space")
    else:
        print("\n✗ FAILED: No functions generated")

if __name__ == "__main__":
    test_new_algorithm()
