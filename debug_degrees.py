#!/usr/bin/env python3
"""
Diagnostic script to investigate polynomial degree limitations.
"""

from src.genetic_polynomial_fitter import GeneticPolynomialFitter
import numpy as np

# Create test data - more complex than quadratic to require higher degrees
data_points = []
for x in np.linspace(0, 5, 30):
    y = x**4 - 2*x**3 + x**2 + 3*x + 1  # 4th degree polynomial
    data_points.append((x, y))

print(f"Testing with {len(data_points)} points from 4th degree polynomial")
print("y = x^4 - 2x^3 + x^2 + 3x + 1")

# Create fitter with higher max_degree
fitter = GeneticPolynomialFitter(
    population_size=50, 
    generations=30, 
    max_polynomials=1, 
    max_degree=10  # Allow up to degree 10
)

print(f"Max degree allowed: {fitter.max_degree}")

# Override the fitness function to debug point selection
original_fit = fitter._fit_polynomial_to_points

def debug_fit_polynomial_to_points(point_indices, data_points):
    print(f"\nDEBUG: point_indices = {point_indices}")
    unique_indices = list(set(point_indices))
    print(f"DEBUG: unique_indices = {unique_indices} (count: {len(unique_indices)})")
    
    result = original_fit(point_indices, data_points)
    print(f"DEBUG: resulting degree = {result.degree}")
    return result

fitter._fit_polynomial_to_points = debug_fit_polynomial_to_points

try:
    result = fitter.fit(data_points)
    print(f'\nFINAL RESULT:')
    for i, poly in enumerate(result):
        print(f'  Polynomial {i}: degree {poly.degree}')
        print(f'  Function: {str(poly)[:150]}...')
        
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
