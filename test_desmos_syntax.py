#!/usr/bin/env python3
"""Test the new Desmos domain syntax."""

from src.genetic_polynomial_fitter import Polynomial

def test_new_syntax():
    """Test that polynomials generate correct Desmos syntax."""
    print("=== Testing New Desmos Domain Syntax ===")
    
    # Create test polynomial
    coefficients = [0.634068, 18.079992]  # y = 0.634068 + 18.079992*x
    fit_points = [(16.043, 290.0), (44.928, 830.0)]  # Matching your example
    
    poly = Polynomial(coefficients, fit_points, degree=1)
    result = str(poly)
    
    print("Generated polynomial:")
    print(result)
    print()
    
    # Check syntax components
    has_backslash_left = '\\left\\{' in result
    has_backslash_right = '\\right\\}' in result
    has_le = '\\le' in result
    has_proper_format = result.startswith('y=')
    
    print("Syntax verification:")
    print(f"  Has \\left\\{{: {has_backslash_left}")
    print(f"  Has \\right\\}}: {has_backslash_right}")
    print(f"  Has \\le: {has_le}")
    print(f"  Starts with y=: {has_proper_format}")
    
    expected_pattern = "y=0.634068+18.079992*x\\ \\left\\{"
    matches_expected = result.startswith("y=0.634068+18.079992*x\\ \\left\\{")
    print(f"  Matches expected pattern: {matches_expected}")

if __name__ == "__main__":
    test_new_syntax()
