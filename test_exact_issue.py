#!/usr/bin/env python3
"""
Test to verify the exact issue with polynomial fitting.
This will show if the problem is in the fitting or in the function strings.
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polynomial_fitter import PolynomialFitter

def manual_polynomial_test():
    """Test polynomial fitting manually to see what's wrong."""
    print("MANUAL POLYNOMIAL TEST")
    print("="*40)
    
    # Very simple test case: 3 points for a quadratic
    points = np.array([
        [1.0, 4.0],   # x=1, y=4
        [2.0, 7.0],   # x=2, y=7  
        [3.0, 12.0]   # x=3, y=12
    ])
    
    print("Test points:")
    for i, (x, y) in enumerate(points):
        print(f"  ({x}, {y})")
    
    print("\nStep 1: Manual numpy polyfit")
    x_data = points[:, 0]
    y_data = points[:, 1]
    
    degree = len(points) - 1  # degree 2 for 3 points
    coeffs = np.polyfit(x_data, y_data, degree)
    print(f"Degree: {degree}")
    print(f"Coefficients: {coeffs}")
    
    # Test the polynomial
    poly = np.poly1d(coeffs)
    print(f"Polynomial object: {poly}")
    
    print("\nVerification:")
    for x, y in zip(x_data, y_data):
        predicted = poly(x)
        error = abs(predicted - y)
        print(f"  x={x}: expected={y}, predicted={predicted:.10f}, error={error:.2e}")
    
    print("\nStep 2: Manual function string creation")
    # Create function string manually
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        
        if power == 0:
            terms.append(f"{c}")
        elif power == 1:
            terms.append(f"{c}*x")
        else:
            terms.append(f"{c}*x^{power}")
    
    func_str = "y = " + " + ".join(terms).replace("+ -", "- ")
    print(f"Function string: {func_str}")
    
    print("\nStep 3: Test with PolynomialFitter class")
    fitter = PolynomialFitter()
    result = fitter._fit_exact_polynomial(points, max_degree=10)
    print(f"Fitter result: {result}")
    
    return func_str

def test_evaluation():
    """Test if we can evaluate the generated function."""
    print("\n" + "="*40)
    print("FUNCTION EVALUATION TEST")
    print("="*40)
    
    func_str = manual_polynomial_test()
    
    if func_str:
        print(f"\nTrying to evaluate: {func_str}")
        
        # Parse and evaluate manually (simple case)
        try:
            # For the simple case, let's manually check if the string is correct
            print("\nManual check of function string:")
            
            # Test points
            test_x = [1.0, 2.0, 3.0]
            expected_y = [4.0, 7.0, 12.0]
            
            print(f"If we substitute x values into: {func_str}")
            print("We should get the original y values:")
            
            for x, expected in zip(test_x, expected_y):
                print(f"  When x={x}, should get y={expected}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_evaluation()
