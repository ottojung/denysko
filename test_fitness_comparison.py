#!/usr/bin/env python3
"""
Test if lower-degree polynomials are getting better fitness scores than higher-degree ones.
"""

from src.genetic_polynomial_fitter import GeneticPolynomialFitter, Individual
import numpy as np

# Complex test data that requires higher degrees
data_points = []
for x in np.linspace(0, 4, 25):
    # Complex function that needs high degree to fit well
    y = 0.1*x**5 - 0.5*x**4 + x**3 + 2*x**2 - x + 3
    data_points.append((x, y))

print("Testing fitness comparison between different degree polynomials")
print("Data: y = 0.1*x^5 - 0.5*x^4 + x^3 + 2*x^2 - x + 3")

fitter = GeneticPolynomialFitter(max_degree=8)

# Create polynomials with different degrees by selecting different numbers of points
test_cases = [
    ("Low degree (2-3)", [0, 5, 10]),           # 3 points -> degree 2
    ("Medium degree (3-4)", [0, 5, 10, 15]),    # 4 points -> degree 3  
    ("High degree (4-5)", [0, 5, 10, 15, 20]),  # 5 points -> degree 4
    ("Higher degree (5-6)", [0, 5, 10, 15, 20, 24]), # 6 points -> degree 5
]

print("\nComparing fitness scores:")

for name, point_indices in test_cases:
    # Create individual with this point selection for both polynomials
    point_lists = [point_indices, point_indices]  # Same for both polynomials
    
    # Fit polynomials
    polynomials = []
    for point_list in point_lists:
        poly = fitter._fit_polynomial_to_points(point_list, data_points)
        polynomials.append(poly)
    
    # Create individual and evaluate fitness
    individual = Individual(point_lists, polynomials)
    fitness = fitter._evaluate_fitness(individual, data_points)
    
    # Calculate actual accuracy for comparison
    total_distance = 0.0
    max_error = 0.0
    for x, y in data_points:
        min_distance = float('inf')
        for poly in polynomials:
            pred = poly.evaluate(x)
            distance = abs(pred - y)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
        max_error = max(max_error, min_distance)
    
    avg_distance = total_distance / len(data_points)
    
    print(f"\n{name}:")
    print(f"  Point indices: {point_indices}")
    print(f"  Degrees: {[p.degree for p in polynomials]}")
    print(f"  Fitness score: {fitness:.2f}")
    print(f"  Average distance: {avg_distance:.4f}")
    print(f"  Max error: {max_error:.4f}")

print("\nIf lower-degree polynomials have higher fitness, that's the problem!")
print("The algorithm converges to simple solutions instead of exploring complex ones.")
