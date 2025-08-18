#!/usr/bin/env python3
"""
Test script for the simple polynomial fitter.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_polynomial_fitter import SimplePolynomialFitter
import numpy as np

def test_simple_fitter():
    """Test the simple polynomial fitter with basic data."""
    
    # Create some test data - a simple curve
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + 0.5 * np.sin(x) + np.random.normal(0, 0.1, len(x))
    data_points = list(zip(x, y))
    
    print("=== Testing Simple Polynomial Fitter ===")
    print(f"Generated {len(data_points)} test data points")
    
    # Create fitter and fit
    fitter = SimplePolynomialFitter(max_iterations=500)
    polynomials = fitter.fit(data_points)
    
    # Print results
    print(f"\nGenerated {len(polynomials)} polynomials:")
    for i, poly in enumerate(polynomials):
        print(f"  Poly {i+1}: {poly}")

if __name__ == "__main__":
    test_simple_fitter()
