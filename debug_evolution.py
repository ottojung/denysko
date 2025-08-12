#!/usr/bin/env python3
"""Debug script to track evolution dynamics and see where high-degree solutions are lost."""

import numpy as np
from src.genetic_polynomial_fitter import GeneticPolynomialFitter

def debug_evolution():
    # Create some test data - simple parabola that needs degree 2+
    np.random.seed(42)
    x = np.linspace(-5, 5, 50)
    y = x**2 + 0.1*x**3 + np.random.normal(0, 0.1, len(x))  # Slight cubic component
    data_points = list(zip(x, y))
    
    print(f"Generated {len(data_points)} test points")
    
    # Create fitter with very small population and few generations for debugging
    fitter = GeneticPolynomialFitter(
        population_size=8,     # Small for debugging
        generations=5,         # Just a few generations
        max_polynomials=2,
        max_degree=6,
        mutation_rate=0.1,     # Much lower mutation rate
        tournament_size=3
    )
    
    # Override the callback to track evolution
    original_callback = fitter._on_generation_callback
    
    def debug_callback(ga_instance):
        generation = ga_instance.generations_completed
        print(f"\n--- Generation {generation} ---")
        
        # Get all solutions and their details
        for i, solution in enumerate(ga_instance.population):
            point_lists = fitter._decode_solution(solution)
            degrees = []
            unique_counts = []
            
            for point_list in point_lists:
                unique_points = len(set(point_list))
                unique_counts.append(unique_points)
                if unique_points > 1:
                    degrees.append(unique_points - 1)
                else:
                    degrees.append(0)
            
            fitness = fitter._fitness_function(None, solution, i)
            print(f"  Individual {i}: unique_counts={unique_counts}, degrees={degrees}, fitness={fitness:.1f}")
        
        # Call original callback if it exists
        if original_callback:
            original_callback(ga_instance)
    
    fitter._on_generation_callback = debug_callback
    
    print("\n=== Running evolution with debug tracking ===")
    result = fitter.fit(data_points)
    
    return result

if __name__ == "__main__":
    debug_evolution()
