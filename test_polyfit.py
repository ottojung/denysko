#!/usr/bin/env python3
"""
Simple test to verify polynomial fitting works correctly.
"""

import numpy as np

def test_numpy_polyfit():
    """Test that numpy's polyfit works as expected."""
    print("TESTING NUMPY POLYFIT")
    print("="*30)
    
    # Simple test points
    x = np.array([1, 2, 3, 4])
    y = np.array([2, 5, 10, 17])
    
    print(f"Test points:")
    for xi, yi in zip(x, y):
        print(f"  ({xi}, {yi})")
    
    # Fit exact polynomial (degree = n-1)
    degree = len(x) - 1  # degree 3 for 4 points
    print(f"\nFitting polynomial of degree {degree}")
    
    coeffs = np.polyfit(x, y, degree)
    print(f"Coefficients: {coeffs}")
    
    # Test the polynomial
    poly = np.poly1d(coeffs)
    print(f"\nTesting polynomial:")
    
    for xi, yi in zip(x, y):
        predicted = poly(xi)
        error = abs(predicted - yi)
        print(f"  x={xi}: expected={yi}, predicted={predicted:.10f}, error={error:.2e}")
        
    # Convert to string manually
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-12:
            continue
            
        if power == 0:
            terms.append(f"{c:.6g}")
        elif power == 1:
            terms.append(f"{c:.6g}*x")
        else:
            terms.append(f"{c:.6g}*x^{power}")
    
    if terms:
        func_str = "y = " + terms[0]
        for term in terms[1:]:
            if term.startswith('-'):
                func_str += " - " + term[1:]
            else:
                func_str += " + " + term
        
        print(f"\nFunction string: {func_str}")

if __name__ == "__main__":
    test_numpy_polyfit()
