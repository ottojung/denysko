#!/usr/bin/env python3
"""
Point-selection genetic algorithm for polynomial fitting.

The genotype is a list of lists of points from the input data.
Each list has exactly max_degree points (initially all the same).
Evolution changes which points are selected.
Polynomials are fit exactly to the unique points in each list.
Fitness is coverage of all input points by all polynomials.
"""

import numpy as np
import random
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Polynomial:
    """Represents a polynomial y = f(x) fitted to specific points."""
    
    coefficients: List[float]  # [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
    fit_points: List[Tuple[float, float]]  # Points this polynomial was fitted to
    degree: int

    def evaluate(self, x):
        """Evaluate polynomial at x."""
        result = 0.0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x ** i)
        return result

    def __str__(self):
        """Convert to Desmos-compatible string."""
        terms = []
        for i, coeff in enumerate(self.coefficients):
            if abs(coeff) < 1e-10:  # Skip near-zero coefficients
                continue

            if i == 0:
                terms.append(f"{coeff:.6f}")
            elif i == 1:
                if coeff == 1.0:
                    terms.append("x")
                elif coeff == -1.0:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff:.6f}*x")
            else:
                if coeff == 1.0:
                    terms.append(f"x^{i}")
                elif coeff == -1.0:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff:.6f}*x^{i}")

        if not terms:
            return "y = 0"

        result = " + ".join(terms).replace(" + -", " - ")
        return f"y = {result}"


@dataclass
class Individual:
    """Individual with point-selection genotype."""
    
    point_lists: List[List[int]]  # List of lists of indices into data points
    polynomials: List[Polynomial]  # Fitted polynomials
    fitness: float = 0.0

    def num_polynomials(self):
        return len(self.point_lists)


class GeneticPolynomialFitter:
    """Point-selection genetic algorithm for polynomial fitting."""

    def __init__(
        self,
        population_size=100,
        generations=200,
        tournament_size=5,
        crossover_rate=0.8,
        mutation_rate=0.3,
        max_polynomials=2,
        max_degree=6,
        fitness_weights=None,
    ):
        self.population_size = population_size
        self.max_generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_polynomials = max_polynomials
        self.max_degree = max_degree
        self.fitness_weights = fitness_weights or {"coverage": 1.0}
        self.debug_count = 0

    def _fit_polynomial_to_points(self, point_indices, data_points):
        """Fit a polynomial exactly to the unique points specified by indices."""
        # Get unique points
        unique_indices = list(set(point_indices))
        if len(unique_indices) < 2:
            # Need at least 2 points, duplicate if necessary
            unique_indices = point_indices[:2] if len(point_indices) >= 2 else [point_indices[0], point_indices[0]]
        
        # Extract points
        fit_points = [data_points[i] for i in unique_indices]
        x_vals = [p[0] for p in fit_points]
        y_vals = [p[1] for p in fit_points]
        
        # Determine degree based on number of unique points
        degree = len(unique_indices) - 1
        degree = max(1, min(degree, self.max_degree))  # Clamp between 1 and max_degree
        
        try:
            # Fit polynomial using numpy polyfit
            coefficients = np.polyfit(x_vals, y_vals, degree)
            # Convert to ascending order (a0 + a1*x + a2*x^2 + ...)
            coefficients = coefficients[::-1].tolist()
            
            return Polynomial(coefficients, fit_points, degree)
            
        except Exception as e:
            print(f"Warning: Polynomial fitting failed: {e}")
            # Fallback: constant polynomial at mean y value
            mean_y = np.mean(y_vals)
            return Polynomial([mean_y], fit_points, 0)

    def _create_random_individual(self, data_points):
        """Create a random individual by selecting random points for each polynomial."""
        point_lists = []
        
        for _ in range(self.max_polynomials):
            # Start with the same point repeated max_degree times
            random_point_idx = random.randint(0, len(data_points) - 1)
            point_list = [random_point_idx] * self.max_degree
            point_lists.append(point_list)
        
        # Fit polynomials to selected points
        polynomials = []
        for point_list in point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            polynomials.append(poly)
        
        return Individual(point_lists, polynomials)

    def _evaluate_fitness(self, individual, data_points):
        """Evaluate fitness based on coverage of all data points."""
        covered_points = 0
        tolerance = 5.0  # Same tight tolerance as test
        
        for x, y in data_points:
            best_pred = None
            min_error = float('inf')
            
            # Find best prediction among all polynomials
            for poly in individual.polynomials:
                try:
                    pred = poly.evaluate(x)
                    error = abs(pred - y)
                    if error < min_error:
                        min_error = error
                        best_pred = pred
                except Exception:
                    continue
            
            # Check if point is covered within tolerance
            if best_pred is not None and min_error <= tolerance:
                covered_points += 1
        
        # Fitness is coverage percentage
        coverage_ratio = covered_points / len(data_points)
        fitness = coverage_ratio * 100  # Scale to 0-100
        
        # Bonus for very high coverage
        if coverage_ratio >= 0.99:
            fitness += 50
        elif coverage_ratio >= 0.95:
            fitness += 25
        elif coverage_ratio >= 0.90:
            fitness += 10
        
        return fitness

    def _tournament_selection(self, population, tournament_size=None):
        """Select an individual using tournament selection."""
        if tournament_size is None:
            tournament_size = self.tournament_size
        tournament_size = min(tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2, data_points):
        """Create offspring by crossing over point selections."""
        offspring_point_lists = []
        
        for i in range(len(parent1.point_lists)):
            # Mix point indices from both parents
            if random.random() < 0.5:
                # Take from parent1 with some mixing
                point_list = parent1.point_lists[i].copy()
                # Mix in some points from parent2
                for j in range(len(point_list)):
                    if random.random() < 0.3:  # 30% chance to take from parent2
                        if i < len(parent2.point_lists):
                            point_list[j] = parent2.point_lists[i][j]
            else:
                # Take from parent2 with some mixing
                point_list = parent2.point_lists[i].copy()
                # Mix in some points from parent1
                for j in range(len(point_list)):
                    if random.random() < 0.3:  # 30% chance to take from parent1
                        point_list[j] = parent1.point_lists[i][j]
            
            offspring_point_lists.append(point_list)
        
        # Fit polynomials to new point selections
        polynomials = []
        for point_list in offspring_point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            polynomials.append(poly)
        
        return Individual(offspring_point_lists, polynomials)

    def _mutate(self, individual, data_points):
        """Mutate an individual by changing some point selections."""
        for point_list in individual.point_lists:
            for i in range(len(point_list)):
                if random.random() < self.mutation_rate:
                    # Replace with a random point from the dataset
                    point_list[i] = random.randint(0, len(data_points) - 1)
        
        # Refit polynomials after mutation
        individual.polynomials = []
        for point_list in individual.point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            individual.polynomials.append(poly)
        
        return individual

    def fit(self, data_points):
        """Fit polynomials to data points using point-selection genetic algorithm."""
        self.debug_count = 0
        data_points = [(float(p[0]), float(p[1])) for p in data_points]
        
        print(f"Starting genetic algorithm with {len(data_points)} data points")
        print(f"Parameters: pop={self.population_size}, gen={self.max_generations}, "
              f"poly={self.max_polynomials}, degree={self.max_degree}")

        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual(data_points)
            individual.fitness = self._evaluate_fitness(individual, data_points)
            population.append(individual)

        # Track best fitness per generation
        best_fitness_history = []
        stagnation_count = 0
        tolerance = 1e-6

        # Evolution loop
        for generation in range(self.max_generations):
            # Sort population by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            current_best = population[0].fitness
            best_fitness_history.append(current_best)

            # Check for improvement
            if generation > 0 and abs(current_best - best_fitness_history[generation-1]) < tolerance:
                stagnation_count += 1
            else:
                stagnation_count = 0

            # Progress report every 50 generations
            if generation % 50 == 0 or generation == self.max_generations - 1:
                coverage_pct = current_best / 100.0 if current_best <= 100 else (current_best - 50) / 100.0
                print(f"Generation {generation}: Best fitness = {current_best:.1f} "
                      f"(~{coverage_pct:.1%} coverage), Stagnation = {stagnation_count}")

            # Create next generation
            new_population = []
            
            # Elite preservation - keep best 10%
            elite_size = max(1, self.population_size // 10)
            for i in range(elite_size):
                new_population.append(population[i])

            # Generate offspring to fill population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = self._crossover(parent1, parent2, data_points)
                else:
                    offspring = parent1  # Clone parent1

                # Mutation
                offspring = self._mutate(offspring, data_points)

                # Evaluate fitness
                offspring.fitness = self._evaluate_fitness(offspring, data_points)
                new_population.append(offspring)

            population = new_population

        # Return best individual's polynomials
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best_individual = population[0]
        
        final_coverage = best_individual.fitness / 100.0 if best_individual.fitness <= 100 else (best_individual.fitness - 50) / 100.0
        print("Final results:")
        print(f"  Best fitness: {best_individual.fitness:.1f}")
        print(f"  Estimated coverage: {final_coverage:.1%}")
        print(f"  Number of polynomials: {len(best_individual.polynomials)}")
        
        for i, poly in enumerate(best_individual.polynomials):
            unique_points = len(set(best_individual.point_lists[i]))
            print(f"  Polynomial {i}: degree {poly.degree}, fitted to {unique_points} unique points")
        
        return best_individual.polynomials
