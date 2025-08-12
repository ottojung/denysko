#!/usr/bin/env python3
"""
Simple test script to verify PyGAD implementation works.
"""

from src.genetic_polynomial_fitter import GeneticPolynomialFitter
import numpy as np

# Simple test data - quadratic function
data_points = []
for x in np.linspace(0, 5, 20):
    y = x**2 + 2*x + 1  # y = x^2 + 2x + 1
    data_points.append((x, y))

print(f"Testing with {len(data_points)} points from quadratic function y = x^2 + 2x + 1")

# Create simple fitter
fitter = GeneticPolynomialFitter(
    population_size=20, 
    generations=20, 
    max_polynomials=1, 
    max_degree=3
)

try:
    result = fitter.fit(data_points)
    print(f'SUCCESS: Got {len(result)} polynomials')
    for i, poly in enumerate(result):
        print(f'  Polynomial {i}: degree {poly.degree}')
        print(f'  Function: {str(poly)}')
    
    # Test coverage
    covered = 0
    for x, y_actual in data_points:
        for poly in result:
            pred = poly.evaluate(x)
            error = abs(pred - y_actual)
            if error <= 0.1:  # Very tight tolerance for simple data
                covered += 1
                break
    
    coverage = covered / len(data_points)
    print(f'Coverage: {coverage:.1%} ({covered}/{len(data_points)} points)')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
