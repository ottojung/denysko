#!/usr/bin/env python3
"""Test with data that definitely needs high-degree polynomials."""

import numpy as np
from src.genetic_polynomial_fitter import GeneticPolynomialFitter

def test_high_degree_needed():
    # Create data that definitely needs high degrees - a degree 5 polynomial
    np.random.seed(42)
    x = np.linspace(-2, 2, 30)
    # Degree 5 polynomial: x^5 - 5x^3 + 4x (has multiple peaks and valleys)
    y = x**5 - 5*x**3 + 4*x + np.random.normal(0, 0.02, len(x))  # Very little noise
    data_points = list(zip(x, y))
    
    print(f"Generated {len(data_points)} test points from degree 5 polynomial: x^5 - 5x^3 + 4x")
    print(f"Data range: x from {x.min():.2f} to {x.max():.2f}, y from {y.min():.2f} to {y.max():.2f}")
    
    # Test with different degrees manually first
    fitter = GeneticPolynomialFitter()
    fitter.data_points = data_points
    
    # Create test solutions with different degrees for ONE polynomial
    # We'll use num_genes = max_polynomials * max_degree = 2 * 6 = 12
    
    print("\n=== Manual fitness comparison ===")
    results = []
    
    for target_degree in [2, 3, 4, 5]:
        # Create solution: first polynomial uses target_degree+1 points (for target_degree)
        # second polynomial uses just 2 points (degree 1, minimal)
        solution = []
        
        # First polynomial: select evenly spaced points for target degree
        indices = np.linspace(0, len(data_points)-1, target_degree+1, dtype=int)
        poly1_points = indices.tolist()
        # Pad to max_degree (6) by repeating last point
        while len(poly1_points) < 6:
            poly1_points.append(poly1_points[-1])
        solution.extend(poly1_points)
        
        # Second polynomial: minimal (2 points for degree 1)
        poly2_points = [0, len(data_points)-1] + [0] * 4  # Pad with repeats
        solution.extend(poly2_points)
        
        fitness = fitter._fitness_function(None, solution, 0)
        point_lists = fitter._decode_solution(solution)
        unique_counts = [len(set(pl)) for pl in point_lists]
        
        print(f"Target degree {target_degree}: unique_points={unique_counts}, fitness={fitness:.1f}")
        results.append((target_degree, fitness, unique_counts))
    
    # Now run actual GA
    print("\n=== Running GA with fixed mutation rate ===")
    fitter_ga = GeneticPolynomialFitter(
        population_size=20,
        generations=50,
        max_polynomials=1,  # Just one polynomial to focus on degree
        max_degree=6,
        mutation_rate=0.1,  # Low mutation rate
        tournament_size=3
    )
    
    result = fitter_ga.fit(data_points)
    
    return result, results

if __name__ == "__main__":
    test_high_degree_needed()
