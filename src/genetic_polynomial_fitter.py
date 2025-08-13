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
        """Convert to Desmos-compatible string with domain restrictions."""
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
            polynomial_expr = "0"
        else:
            polynomial_expr = " + ".join(terms).replace(" + -", " - ")

        # Calculate domain restrictions based on fit_points x-coordinates
        if self.fit_points:
            x_coords = [point[0] for point in self.fit_points]
            x_min = min(x_coords)
            x_max = max(x_coords)
            
            # Add small buffer to ensure smooth connections at endpoints
            x_range = x_max - x_min
            buffer = max(x_range * 0.05, 1.0)  # 5% buffer or at least 1 unit
            x_min -= buffer
            x_max += buffer
            
            return f"y = {polynomial_expr} {{{x_min:.2f} ≤ x ≤ {x_max:.2f}}}"
        else:
            return f"y = {polynomial_expr}"


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
            mutation_type=self._custom_mutation,  # Use two-tier custom mutation
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
        """Two-tier mutation function: small nudges (common) + big changes (rare)."""
        if self.data_points is None:
            return offspring

        mutation_rate = max(0.1, self.mutation_rate)

        # Pre-compute spatial relationships for efficient neighbor finding
        if not hasattr(self, '_neighbor_cache'):
            self._build_neighbor_cache()

        # Convert offspring to proper numpy array and work with individual solutions
        offspring_array = np.array(offspring)
        
        for solution_idx in range(offspring_array.shape[0]):  # For each solution in population
            solution = offspring_array[solution_idx]
            
            # Find uncovered points for this individual solution
            point_lists = self._decode_solution(solution)
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

            # Apply TWO-TIER MUTATIONS to this solution
            for gene_idx in range(len(solution)):
                if np.random.random() < mutation_rate:
                    current_point_idx = int(solution[gene_idx])  # Extract the point index
                    
                    # Decide mutation type: 80% small nudges, 20% big changes
                    if np.random.random() < 0.8:
                        # SMALL NUDGE: Replace with nearby neighbor
                        neighbors = self._get_neighbors(current_point_idx, radius=5)
                        if neighbors:
                            # Prefer uncovered neighbors if available
                            uncovered_neighbors = [n for n in neighbors if n in uncovered_points]
                            if uncovered_neighbors:
                                solution[gene_idx] = np.random.choice(uncovered_neighbors)
                            else:
                                solution[gene_idx] = np.random.choice(neighbors)
                        else:
                            # No neighbors found, fall back to random selection from uncovered points
                            if uncovered_points:
                                solution[gene_idx] = np.random.choice(uncovered_points)
                            else:
                                solution[gene_idx] = np.random.randint(0, len(self.data_points))
                    else:
                        # BIG CHANGE: Replace with completely random point
                        if uncovered_points and np.random.random() < 0.7:
                            # 70% chance to pick from uncovered points
                            solution[gene_idx] = np.random.choice(uncovered_points)
                        else:
                            # 30% chance for completely random point
                            solution[gene_idx] = np.random.randint(0, len(self.data_points))

            # Force diversity within each polynomial for this solution
            genes_per_poly = self.max_degree
            for poly_idx in range(self.max_polynomials):
                start_idx = poly_idx * genes_per_poly
                end_idx = start_idx + genes_per_poly
                poly_genes = solution[start_idx:end_idx]

                # Convert to list of integers for set operations
                poly_genes_list = [int(g) for g in poly_genes]
                
                # If all genes are the same, diversify
                if len(set(poly_genes_list)) == 1:
                    # Keep one, change others
                    for i in range(1, len(poly_genes)):
                        gene_pos = start_idx + i
                        if uncovered_points:
                            solution[gene_pos] = np.random.choice(uncovered_points)
                        else:
                            solution[gene_pos] = np.random.randint(0, len(self.data_points))

        return offspring_array

        return offspring

    def _build_neighbor_cache(self):
        """Build cache of spatial neighbors for efficient mutation."""
        if self.data_points is None:
            return
            
        print("Building spatial neighbor cache for improved mutations...")
        self._neighbor_cache = {}
        
        for i, (x1, y1) in enumerate(self.data_points):
            neighbors = []
            for j, (x2, y2) in enumerate(self.data_points):
                if i != j:
                    # Calculate Euclidean distance
                    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    if dist <= 10.0:  # Neighbor if within distance threshold
                        neighbors.append((j, dist))
            
            # Sort neighbors by distance and keep closest ones
            neighbors.sort(key=lambda x: x[1])
            self._neighbor_cache[i] = [idx for idx, dist in neighbors[:20]]  # Keep 20 closest neighbors
        
        print(f"Built neighbor cache for {len(self.data_points)} points")

    def _get_neighbors(self, point_idx, radius=5):
        """Get spatial neighbors of a point within given radius."""
        if not hasattr(self, '_neighbor_cache') or point_idx not in self._neighbor_cache:
            return []
        
        # Return subset of cached neighbors within radius
        all_neighbors = self._neighbor_cache[point_idx]
        if radius >= 20:  # If radius is large, return all cached neighbors
            return all_neighbors
        
        # For smaller radius, return first 'radius' neighbors (they're sorted by distance)
        return all_neighbors[:min(radius, len(all_neighbors))]

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
        
        # Calculate polynomial metrics for complexity analysis
        num_polynomials = len([p for p in individual.polynomials if p.degree > 0])
        total_degree = sum(p.degree for p in individual.polynomials)
        
        # Calculate complexity penalties - ADAPTIVE based on letter complexity
        complexity_penalty = 0.0
        
        # Calculate data complexity: measure y-variation across x-regions
        x_coords = [p[0] for p in data_points]
        y_coords = [p[1] for p in data_points]
        if len(x_coords) > 0:
            x_min, x_max = min(x_coords), max(x_coords)
            x_span = x_max - x_min
            
            # Check y-variation in middle region (where complex letters have problems)
            if x_span > 0:
                middle_x = x_min + 0.5 * x_span
                middle_width = 0.3 * x_span
                middle_points = [(x, y) for x, y in data_points 
                               if middle_x - middle_width/2 <= x <= middle_x + middle_width/2]
                
                if len(middle_points) > 10:  # Need sufficient points for analysis
                    middle_y = [y for x, y in middle_points]
                    y_variation = max(middle_y) - min(middle_y) if middle_y else 0
                    y_span = max(y_coords) - min(y_coords) if y_coords else 1
                    
                    # High y-variation indicates complex letter (like T, H, E, F)
                    complexity_ratio = y_variation / y_span if y_span > 0 else 0
                    is_complex_letter = complexity_ratio > 0.6  # More than 60% y-variation
                else:
                    is_complex_letter = False
            else:
                is_complex_letter = False
        else:
            is_complex_letter = False
        
        # ADAPTIVE penalties based on letter complexity
        if is_complex_letter:
            # Complex letters (T, H, E, F): Allow more polynomials, less penalty
            if num_polynomials > 4:  # Allow up to 4 polynomials for complex letters
                complexity_penalty += (num_polynomials - 4) * 30
            if total_degree > 12:  # Allow higher total degree
                complexity_penalty += (total_degree - 12) * 5
        else:
            # Simple letters (A, O, C, S): Prefer fewer polynomials, stronger penalty
            if num_polynomials > 2:  # Prefer exactly 2 polynomials for simple letters
                complexity_penalty += (num_polynomials - 2) * 50
            if total_degree > 8:  # Prefer lower total degree
                complexity_penalty += (total_degree - 8) * 8
        
        # Universal penalty: Very high individual degrees (avoid overfitting)
        for poly in individual.polynomials:
            if poly.degree > 4:  # Prefer individual degrees ≤ 4
                complexity_penalty += (poly.degree - 4) * 15
        
        # Final fitness = accuracy - complexity penalty
        fitness = accuracy_fitness - complexity_penalty
        
        # Ensure fitness is never negative
        fitness = max(fitness, 1.0)

        return fitness
