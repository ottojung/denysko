#!/usr/bin/env python3
"""
Point-selection genetic algorithm for polynomial fitting using PyGAD.

The genotype is a list of lists of point indices from the input data.
Each list has exactly max_degree indices (initially all the same).
Evolution changes which points are selected for fitting.
Polynomials are fit exactly to the unique points in each list.
Fitness is coverage of all input points by all polynomials.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import warnings
import pygad

warnings.simplefilter("ignore", np.exceptions.RankWarning)


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
            result += coeff * (x**i)
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
    """Point-selection genetic algorithm for polynomial fitting using PyGAD."""

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

        # Store data points for use in fitness function
        self.data_points = None

        # Gene space: each gene is an index into data_points
        # Total genes = max_polynomials * max_degree
        self.num_genes = max_polynomials * max_degree

    def _decode_solution(self, solution):
        """Convert PyGAD solution (flat array) to our point_lists structure."""
        point_lists = []
        genes_per_poly = self.max_degree

        # Ensure solution is flat array
        flat_solution = np.array(solution).flatten()

        for i in range(self.max_polynomials):
            start_idx = i * genes_per_poly
            end_idx = start_idx + genes_per_poly
            point_list = flat_solution[start_idx:end_idx].tolist()
            point_lists.append([int(idx) for idx in point_list])  # Ensure integers

        return point_lists

    def _fitness_function(self, ga_instance, solution, solution_idx):
        """Fitness function for PyGAD - returns fitness score."""
        if self.data_points is None:
            return 0.0

        # Decode solution to point lists
        point_lists = self._decode_solution(solution)

        # Create Individual object with fitted polynomials
        polynomials = []
        for point_list in point_lists:
            poly = self._fit_polynomial_to_points(point_list, self.data_points)
            polynomials.append(poly)

        individual = Individual(point_lists, polynomials)

        # Calculate fitness using our existing logic
        return self._evaluate_fitness(individual, self.data_points)

    def fit(self, data_points):
        """Fit polynomials to data points using PyGAD genetic algorithm."""
        data_points = [(float(p[0]), float(p[1])) for p in data_points]
        self.data_points = data_points  # Store for fitness function

        print(
            f"Starting PyGAD point-selection genetic algorithm with {len(data_points)} data points"
        )
        print(
            f"Parameters: pop={self.population_size}, gen={self.max_generations}, "
            f"poly={self.max_polynomials}, degree={self.max_degree}"
        )

        # Define gene space - each gene is an index into data_points
        gene_space = list(range(len(data_points)))

        # Create strategic initial population
        initial_population = self._create_strategic_initial_population(data_points)

        # Create PyGAD instance with enhanced parameters
        ga_instance = pygad.GA(
            num_generations=self.max_generations,
            num_parents_mating=max(4, self.population_size // 3),  # More parents
            fitness_func=self._fitness_function,
            initial_population=initial_population,  # Use strategic initialization
            sol_per_pop=self.population_size,
            num_genes=self.num_genes,
            gene_space=gene_space,
            gene_type=int,
            parent_selection_type="tournament",
            K_tournament=self.tournament_size,
            keep_parents=max(1, self.population_size // 8),  # More elite preservation
            crossover_type="two_points",  # Better crossover
            mutation_type="random",  # Back to simple random mutation
            mutation_percent_genes=max(
                5, int(self.mutation_rate * 20)
            ),  # More reasonable mutation rate
            random_seed=None,
            suppress_warnings=True,
            on_generation=self._on_generation_callback,
            mutation_by_replacement=True,
            allow_duplicate_genes=True,  # Allow same point to be used multiple times
        )

        # Run the genetic algorithm
        ga_instance.run()

        # Get best solution
        best_solution, best_fitness, _ = ga_instance.best_solution()

        # Decode best solution to get final polynomials
        best_point_lists = self._decode_solution(best_solution)
        best_polynomials = []
        for point_list in best_point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            best_polynomials.append(poly)

        # Print final results
        final_coverage = (
            best_fitness / 100.0 if best_fitness <= 100 else (best_fitness - 50) / 100.0
        )
        print("Final results:")
        print(f"  Best fitness: {best_fitness:.1f}")
        print(f"  Estimated coverage: {final_coverage:.1%}")
        print(f"  Number of polynomials: {len(best_polynomials)}")

        for i, poly in enumerate(best_polynomials):
            unique_points = len(set(best_point_lists[i]))
            print(
                f"  Polynomial {i}: degree {poly.degree}, fitted to {unique_points} unique points"
            )

        return best_polynomials

    def _create_strategic_initial_population(self, data_points):
        """Create initial population with strategic point selections."""
        population = []
        x_coords = np.array([x for x, y in data_points])
        y_coords = np.array([y for x, y in data_points])

        # Strategy 1: Extremal points
        x_min_idx = np.argmin(x_coords)
        x_max_idx = np.argmax(x_coords)
        y_min_idx = np.argmin(y_coords)
        y_max_idx = np.argmax(y_coords)
        extremal_points = [x_min_idx, x_max_idx, y_min_idx, y_max_idx]

        # Strategy 2: Quantile points
        x_sorted_indices = np.argsort(x_coords)
        quantile_points = []
        for q in [0.1, 0.3, 0.5, 0.7, 0.9]:
            idx = x_sorted_indices[int(q * len(x_sorted_indices))]
            quantile_points.append(idx)

        # Strategy 3: High-variation points (potential corners/features)
        variation_scores = []
        for i in range(len(data_points)):
            # Calculate local variation
            distances = []
            for j in range(len(data_points)):
                if i != j:
                    dx = x_coords[i] - x_coords[j]
                    dy = y_coords[i] - y_coords[j]
                    distances.append((dx * dx + dy * dy, j))

            distances.sort()
            nearby_indices = [idx for _, idx in distances[: min(10, len(distances))]]

            if len(nearby_indices) > 1:
                nearby_y = [y_coords[idx] for idx in nearby_indices]
                variation = np.std(nearby_y)
            else:
                variation = 0
            variation_scores.append(variation)

        high_variation_indices = np.argsort(variation_scores)[-20:].tolist()

        # Create diverse initial solutions
        for i in range(self.population_size):
            solution = []

            for poly_idx in range(self.max_polynomials):
                if i % 4 == 0:
                    # Strategy: Use extremal points
                    point_candidates = extremal_points + quantile_points
                elif i % 4 == 1:
                    # Strategy: Use high-variation points
                    point_candidates = high_variation_indices
                elif i % 4 == 2:
                    # Strategy: Use quantile distribution
                    point_candidates = quantile_points + extremal_points
                else:
                    # Strategy: Random selection
                    point_candidates = list(range(len(data_points)))

                # Select points for this polynomial with diversity
                poly_points = []

                for gene_idx in range(self.max_degree):
                    if point_candidates:
                        candidate = np.random.choice(point_candidates)
                    else:
                        candidate = np.random.randint(0, len(data_points))
                    poly_points.append(candidate)

                solution.extend(poly_points)

            population.append(solution)

        return np.array(population, dtype=int)

    def _custom_mutation(self, offspring, ga_instance):
        """Custom mutation function that targets uncovered points."""
        if self.data_points is None:
            return offspring

        mutation_rate = max(0.1, self.mutation_rate)

        # Find uncovered points for this individual
        point_lists = self._decode_solution(offspring)
        polynomials = []
        for point_list in point_lists:
            poly = self._fit_polynomial_to_points(point_list, self.data_points)
            polynomials.append(poly)

        uncovered_points = []
        tolerance = 5.0

        for i, (x, y) in enumerate(self.data_points):
            best_error = float("inf")
            for poly in polynomials:
                try:
                    pred = poly.evaluate(x)
                    error = abs(pred - y)
                    best_error = min(best_error, error)
                except Exception:
                    continue

            if best_error > tolerance:
                uncovered_points.append(i)

        # Apply mutations
        for gene_idx in range(len(offspring)):
            if np.random.random() < mutation_rate:
                # 60% chance to use uncovered points, 40% random
                if uncovered_points and np.random.random() < 0.6:
                    offspring[gene_idx] = np.random.choice(uncovered_points)
                else:
                    offspring[gene_idx] = np.random.randint(0, len(self.data_points))

        # Force diversity within each polynomial
        genes_per_poly = self.max_degree
        for poly_idx in range(self.max_polynomials):
            start_idx = poly_idx * genes_per_poly
            end_idx = start_idx + genes_per_poly
            poly_genes = offspring[start_idx:end_idx]

            # If all genes are the same, diversify
            if len(set(poly_genes)) == 1:
                # Keep one, change others
                for i in range(1, len(poly_genes)):
                    if uncovered_points:
                        offspring[start_idx + i] = np.random.choice(uncovered_points)
                    else:
                        offspring[start_idx + i] = np.random.randint(
                            0, len(self.data_points)
                        )

        return offspring

    def _on_generation_callback(self, ga_instance):
        """Callback function called after each generation."""
        generation = ga_instance.generations_completed
        if generation % 50 == 0 or generation == self.max_generations:
            best_fitness = ga_instance.best_solution()[1]
            coverage_pct = (
                best_fitness / 100.0
                if best_fitness <= 100
                else (best_fitness - 50) / 100.0
            )
            print(
                f"Generation {generation}: Best fitness = {best_fitness:.1f} "
                f"(~{coverage_pct:.1%} coverage)"
            )

    # Keep all the existing helper methods unchanged
    def _fit_polynomial_to_points(self, point_indices, data_points):
        """Fit a polynomial exactly to the unique points specified by indices."""
        unique_indices = list(set(point_indices))
        if len(unique_indices) < 2:
            if len(point_indices) >= 1:
                unique_indices = [point_indices[0], point_indices[0]]
            else:
                unique_indices = [0, 0]

        fit_points = [data_points[i] for i in unique_indices]
        x_vals = [p[0] for p in fit_points]
        y_vals = [p[1] for p in fit_points]

        degree = len(unique_indices) - 1
        degree = max(1, min(degree, self.max_degree))

        try:
            coefficients = np.polyfit(x_vals, y_vals, degree)
            coefficients = coefficients[::-1].tolist()
            return Polynomial(coefficients, fit_points, degree)
        except Exception:
            mean_y = np.mean(y_vals) if y_vals else 0.0
            return Polynomial([mean_y], fit_points, 0)

    def _evaluate_fitness(self, individual, data_points):
        """Fitness function with complexity penalties: minimize distance AND complexity."""
        if len(data_points) == 0:
            return 0.0

        total_distance = 0.0

        # For each data point, find the minimum distance to any polynomial
        for x, y in data_points:
            min_distance = float("inf")

            # Check distance to each polynomial
            for poly in individual.polynomials:
                try:
                    pred = poly.evaluate(x)
                    distance = abs(pred - y)
                    min_distance = min(min_distance, distance)
                except Exception:
                    continue

            # Add the minimum distance to total
            if min_distance != float("inf"):
                total_distance += min_distance
            else:
                # If no polynomial could evaluate, add a large penalty
                total_distance += 1000.0

        # Calculate base accuracy fitness
        average_distance = total_distance / len(data_points)
        accuracy_fitness = 1000.0 / (1.0 + average_distance)
        
        # Calculate complexity penalties
        complexity_penalty = 0.0
        
        # Penalty 1: Too many polynomials (strongly prefer 2 polynomials)
        num_polynomials = len([p for p in individual.polynomials if p.degree > 0])
        if num_polynomials > 2:  # Prefer exactly 2 polynomials
            complexity_penalty += (num_polynomials - 2) * 50  # 50 point penalty per extra polynomial (increased)
        
        # Penalty 2: High degrees (prefer simpler polynomials)
        total_degree = sum(p.degree for p in individual.polynomials)
        if total_degree > 8:  # Allow slightly higher total degree (two degree-4 polynomials)
            complexity_penalty += (total_degree - 8) * 8   # 8 point penalty per extra degree
        
        # Penalty 3: Very high individual degrees (avoid overfitting)
        for poly in individual.polynomials:
            if poly.degree > 4:  # Prefer individual degrees ≤ 4
                complexity_penalty += (poly.degree - 4) * 15  # 15 point penalty per degree above 4 (increased)
        
        # Final fitness = accuracy - complexity penalty
        fitness = accuracy_fitness - complexity_penalty
        
        # Ensure fitness is never negative
        fitness = max(fitness, 1.0)

        return fitness
