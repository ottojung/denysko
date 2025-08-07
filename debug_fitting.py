#!/usr/bin/env python3
"""
Debug the polynomial fitting to see why curves don't pass through points.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polynomial_fitter import PolynomialFitter

def test_simple_points():
    """Test with very simple points to debug the fitting."""
    print("DEBUGGING POLYNOMIAL FITTING")
    print("="*50)
    
    # Simple test case: 4 points that should be easy to fit exactly
    test_points = np.array([
        [1.0, 2.0],    # (1, 2)
        [2.0, 5.0],    # (2, 5)  
        [3.0, 10.0],   # (3, 10)
        [4.0, 17.0]    # (4, 17)
    ])
    
    print(f"Test points:")
    for i, (x, y) in enumerate(test_points):
        print(f"  Point {i+1}: ({x}, {y})")
    
    # Test the fitter
    fitter = PolynomialFitter(max_degree=10)
    
    print(f"\nTesting polynomial fitting...")
    
    # Test the internal method directly
    func_str = fitter._fit_exact_polynomial(test_points, max_degree=10)
    
    print(f"\nGenerated function: {func_str}")
    
    if func_str:
        # Manual verification - parse and test the polynomial
        print(f"\nManual verification:")
        
        # Try to extract coefficients and test manually
        try:
            # Sort points
            x_data, y_data = test_points[:, 0], test_points[:, 1]
            sort_idx = np.argsort(x_data)
            x_sorted, y_sorted = x_data[sort_idx], y_data[sort_idx]
            
            print(f"Sorted points: {list(zip(x_sorted, y_sorted))}")
            
            # Fit polynomial directly
            degree = len(x_sorted) - 1
            coeffs = np.polyfit(x_sorted, y_sorted, degree)
            
            print(f"Coefficients: {coeffs}")
            print(f"Degree: {degree}")
            
            # Test the polynomial
            poly = np.poly1d(coeffs)
            
            print(f"\nTesting polynomial at original points:")
            for x, y in zip(x_sorted, y_sorted):
                predicted = poly(x)
                error = abs(predicted - y)
                print(f"  x={x}: expected={y}, predicted={predicted:.8f}, error={error:.8f}")
                
        except Exception as e:
            print(f"Error in manual verification: {e}")
    else:
        print("No function generated!")

def test_letter_a_overlap():
    """Test the overlap detection with letter A points."""
    print("\n" + "="*50)
    print("TESTING OVERLAP DETECTION")
    print("="*50)
    
    # Create letter A points with clear overlap
    points = []
    
    # Left diagonal: x=10 to 30, y=10 to 50
    for i in range(5):
        x = 10 + i * 5  # 10, 15, 20, 25, 30
        y = 10 + i * 10 # 10, 20, 30, 40, 50
        points.append([x, y])
    
    # Right diagonal: x=30 to 50, y=50 to 10  
    for i in range(5):
        x = 30 + i * 5  # 30, 35, 40, 45, 50
        y = 50 - i * 10 # 50, 40, 30, 20, 10
        points.append([x, y])
        
    # Crossbar: x=20 to 40, y around 30 (OVERLAP!)
    for i in range(4):
        x = 20 + i * 7  # 20, 27, 34, 41
        y = 30 + i * 1  # 30, 31, 32, 33
        points.append([x, y])
    
    letter_a = np.array(points)
    
    print(f"Letter A points ({len(letter_a)}):")
    for i, (x, y) in enumerate(letter_a):
        print(f"  Point {i+1}: ({x:.1f}, {y:.1f})")
    
    # Test overlap detection
    fitter = PolynomialFitter()
    curves = fitter._detect_overlapping_strokes(letter_a)
    
    print(f"\nOverlap detection result:")
    print(f"Found {len(curves)} curves")
    
    for i, curve in enumerate(curves):
        print(f"\nCurve {i+1}: {len(curve)} points")
        for j, (x, y) in enumerate(curve):
            print(f"    Point {j+1}: ({x:.1f}, {y:.1f})")

if __name__ == "__main__":
    test_simple_points()
    test_letter_a_overlap()
