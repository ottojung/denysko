#!/usr/bin/env python3
"""
Targeted test script for letter A coverage with more aggressive parameters.
"""

from src.text_extractor import TextExtractor
from src.genetic_polynomial_fitter import GeneticPolynomialFitter
import numpy as np

print("=== Letter A High-Performance Test ===")

# Generate letter A points
extractor = TextExtractor()
paths = extractor.text_to_paths("A", font_size=100)

all_points = []
for path in paths:
    contours = extractor.extract_contour_points(path, num_points=1000)
    for contour in contours:
        if hasattr(contour, "__len__") and len(contour) > 1:
            all_points.extend(contour)

letter_a_points = np.array(all_points)
print(f"Extracted {len(letter_a_points)} points for letter A")

# Sample points for training and testing
if len(letter_a_points) > 500:
    train_indices = np.random.choice(len(letter_a_points), 500, replace=False)
    train_points = letter_a_points[train_indices]
else:
    train_points = letter_a_points

if len(letter_a_points) > 200:
    test_indices = np.linspace(0, len(letter_a_points)-1, 200, dtype=int)
    test_points = letter_a_points[test_indices]
else:
    test_points = letter_a_points

print(f"Training on {len(train_points)} points, testing on {len(test_points)} points")

# Create ultra-high-performance fitter
fitter = GeneticPolynomialFitter(
    population_size=500,     # Even larger population
    generations=800,        # More generations
    max_polynomials=2,      # Keep 2 polynomials for letter A
    max_degree=8,           # Higher max degree - allow up to degree 8
    mutation_rate=0.4,      # Higher mutation for more exploration
    tournament_size=8       # Larger tournament for stronger selection
)

print(f"\nRunning ultra-high-performance GA:")
print(f"  Population: {fitter.population_size}")
print(f"  Generations: {fitter.max_generations}")
print(f"  Max degree: {fitter.max_degree}")
print(f"  Mutation rate: {fitter.mutation_rate}")

# Fit polynomials
result = fitter.fit(train_points)

print(f"\nFinal polynomials:")
for i, poly in enumerate(result):
    print(f"  Polynomial {i}: degree {poly.degree}")
    print(f"    {str(poly)[:100]}...")

# Test coverage on independent test set
covered_points = 0
tolerance = 5.0
max_error = 0.0
errors_above_threshold = 0

for x, y in test_points:
    best_error = float('inf')
    
    for poly in result:
        try:
            pred = poly.evaluate(x)
            error = abs(pred - y)
            best_error = min(best_error, error)
        except Exception:
            continue
    
    if best_error <= tolerance:
        covered_points += 1
    
    if best_error != float('inf'):
        max_error = max(max_error, best_error)
        if best_error > tolerance:
            errors_above_threshold += 1

coverage_ratio = covered_points / len(test_points)

print(f"\nCoverage Analysis:")
print(f"  Covered points: {covered_points}/{len(test_points)}")
print(f"  Coverage ratio: {coverage_ratio:.1%}")
print(f"  Max error: {max_error:.2f}")
print(f"  Points with errors > {tolerance}: {errors_above_threshold}")

# Success criteria
if coverage_ratio >= 0.99:
    print(f"\n🎉 SUCCESS: Achieved {coverage_ratio:.1%} coverage (≥99% target)")
else:
    print(f"\n❌ NOT YET: {coverage_ratio:.1%} coverage (need ≥99%)")
    print("Strategies for improvement:")
    print("  - Increase max_degree further")
    print("  - Use more training points")
    print("  - Adjust fitness function weights")
    print("  - Try different polynomial combinations")
