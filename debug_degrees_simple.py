#!/usr/bin/env python3
"""
Simple diagnostic to test polynomial degree generation.
"""

from src.genetic_polynomial_fitter import GeneticPolynomialFitter
import numpy as np

# Test data that requires higher degree
data_points = []
for x in np.linspace(0, 3, 20):
    y = x**3 + 2*x**2 - x + 1  # 3rd degree polynomial
    data_points.append((x, y))

print(f"Testing with 3rd degree polynomial data: y = x^3 + 2x^2 - x + 1")

# Test the point fitting directly
fitter = GeneticPolynomialFitter(max_degree=8)

print("\nTesting different point selections:")

# Test with different numbers of unique points
test_cases = [
    [0, 1, 2],           # 3 unique points -> should give degree 2
    [0, 1, 2, 3],        # 4 unique points -> should give degree 3  
    [0, 1, 2, 3, 4],     # 5 unique points -> should give degree 4
    [0, 1, 2, 3, 4, 5],  # 6 unique points -> should give degree 5
    [0, 0, 1, 2, 3],     # 4 unique points (one duplicate) -> should give degree 3
    [5, 5, 5, 5, 5],     # 1 unique point -> should be handled specially
]

for i, point_indices in enumerate(test_cases):
    print(f"\nTest {i+1}: point_indices = {point_indices}")
    unique_count = len(set(point_indices))
    print(f"  Unique points: {unique_count}")
    
    try:
        poly = fitter._fit_polynomial_to_points(point_indices, data_points)
        print(f"  Result degree: {poly.degree}")
        print(f"  Expected degree: {max(1, min(unique_count - 1, 8))}")
        
        if unique_count >= 2:
            expected_degree = min(unique_count - 1, 8)
            if poly.degree != expected_degree:
                print(f"  ⚠️  MISMATCH: got {poly.degree}, expected {expected_degree}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nTesting if PyGAD generates diverse point selections:")
fitter2 = GeneticPolynomialFitter(population_size=10, generations=5, max_degree=8)
initial_pop = fitter2._create_strategic_initial_population(data_points)

print(f"Sample from initial population (first 5 individuals):")
for i in range(min(5, len(initial_pop))):
    solution = initial_pop[i]
    point_lists = fitter2._decode_solution(solution) 
    
    for j, point_list in enumerate(point_lists):
        unique_count = len(set(point_list))
        print(f"  Individual {i}, Poly {j}: {point_list} -> {unique_count} unique points")
