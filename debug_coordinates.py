#!/usr/bin/env python3
"""
Debug script to test coordinate normalization and polynomial evaluation.
"""

import numpy as np
from src.genetic_polynomial_fitter import Polynomial

def test_coordinate_normalization():
    """Test that coordinate normalization works correctly."""
    print("Testing coordinate normalization...")
    
    # Create a simple polynomial: y = 50 + 10*x_normalized
    # For x in domain [10, 60], this should give y values around 40-60
    coeffs = [50.0, 10.0]  # constant=50, linear=10
    x_min, x_max = 10.0, 60.0
    poly = Polynomial(coeffs, x_min, x_max)
    
    print(f"Polynomial: {coeffs[0]:.1f} + {coeffs[1]:.1f}*x_normalized")
    print(f"Domain: [{x_min}, {x_max}]")
    
    # Test at key points
    test_x_values = [10.0, 35.0, 60.0]  # left, middle, right
    
    for x in test_x_values:
        # Calculate expected normalized value
        x_normalized = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        expected_y = coeffs[0] + coeffs[1] * x_normalized
        
        # Get actual prediction
        actual_y = poly.evaluate(x)
        
        print(f"  x={x:.1f} -> x_norm={x_normalized:.2f} -> expected_y={expected_y:.2f}, actual_y={actual_y:.2f}")

def test_with_letter_data():
    """Test with actual letter A data range."""
    print("\nTesting with letter A data range...")
    
    # Simulate letter A data range: x~[11, 58], y~[8, 64]
    x_min, x_max = 11.0, 58.0
    y_min, y_max = 8.0, 64.0
    y_mean = (y_min + y_max) / 2  # ~36
    
    # Create polynomial: constant close to y_mean, small linear term
    coeffs = [y_mean, 5.0]  # y = 36 + 5*x_normalized
    poly = Polynomial(coeffs, x_min, x_max)
    
    print(f"Polynomial: {coeffs[0]:.1f} + {coeffs[1]:.1f}*x_normalized")
    print(f"Domain: [{x_min}, {x_max}]")
    print(f"Expected y range: [{y_mean - abs(coeffs[1]):.1f}, {y_mean + abs(coeffs[1]):.1f}]")
    
    # Test at several x points within the letter range
    test_x_values = [11.0, 30.0, 45.0, 58.0]
    
    for x in test_x_values:
        actual_y = poly.evaluate(x)
        print(f"  x={x:.1f} -> y={actual_y:.2f}")

if __name__ == "__main__":
    test_coordinate_normalization()
    test_with_letter_data()
