#!/usr/bin/env python3
"""Quick test to verify fitness of different degrees on the SAME data as evolution debug."""

import numpy as np
from src.genetic_polynomial_fitter import GeneticPolynomialFitter

def test_fitness_on_evolution_data():
    # Use EXACT same data generation as debug_evolution.py
    np.random.seed(42)  # Same seed!
    x = np.linspace(-5, 5, 50)
    y = x**2 + 0.1*x**3 + np.random.normal(0, 0.1, len(x))  # Slight cubic component
    data_points = list(zip(x, y))
    
    print(f"Generated {len(data_points)} test points (same as evolution)")
    
    fitter = GeneticPolynomialFitter()
    fitter.data_points = data_points  # Set for fitness function
    
    # Create some test solutions with different degrees
    # Test solution 1: Try to create degree 5,5 (like Individual 1 from generation 1)
    solution_degree_5_5 = []
    # For polynomial 1: select 6 different points to get degree 5
    poly1_points = [0, 10, 20, 30, 40, 49]  # 6 unique points
    solution_degree_5_5.extend(poly1_points)
    # For polynomial 2: select 6 different points to get degree 5  
    poly2_points = [5, 15, 25, 35, 45, 48]  # 6 unique points
    solution_degree_5_5.extend(poly2_points)
    
    # Test solution 2: Try to create degree 4,3 (like Individual 0 from generation 1)
    solution_degree_4_3 = []
    # For polynomial 1: select 5 different points to get degree 4
    poly1_points = [0, 12, 24, 36, 48, 48]  # 5 unique points (last one repeated)
    solution_degree_4_3.extend(poly1_points)
    # For polynomial 2: select 4 different points to get degree 3
    poly2_points = [6, 18, 30, 42, 42, 42]  # 4 unique points (repeated)
    solution_degree_4_3.extend(poly2_points)
    
    # Test both
    fitness_5_5 = fitter._fitness_function(None, solution_degree_5_5, 0)
    fitness_4_3 = fitter._fitness_function(None, solution_degree_4_3, 0)
    
    print(f"\nDegrees [5, 5]: fitness = {fitness_5_5:.1f}")
    print(f"Degrees [4, 3]: fitness = {fitness_4_3:.1f}")
    
    # Analyze the solutions
    for name, solution in [("5,5", solution_degree_5_5), ("4,3", solution_degree_4_3)]:
        point_lists = fitter._decode_solution(solution)
        print(f"\nSolution {name}:")
        for i, point_list in enumerate(point_lists):
            unique_points = len(set(point_list))
            print(f"  Poly {i}: {unique_points} unique points from indices {point_list}")

if __name__ == "__main__":
    test_fitness_on_evolution_data()
