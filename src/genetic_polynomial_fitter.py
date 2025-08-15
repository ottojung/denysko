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
            # Join with + and - without spaces to match Desmos format
            polynomial_expr = "+".join(terms).replace("+-", "-")

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

            return f"y={polynomial_expr}\\ \\left\\{{{x_min:.3f}\\le x\\le{x_max:.3f}\\right\\}}"
        else:
            return f"y={polynomial_expr}"


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
        mutation_rate=0.4,
        max_polynomials=6,  # Always generate 6 polynomials
        max_degree=6,  # Always use degree 6
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
        """
        Fit polynomials to data points using PyGAD genetic algorithm.

        Args:
            data_points: sampled data points for genetic algorithm
            full_trace_complexity: complexity calculated from full trace (optional)
        """

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
            mutation_type=self._custom_mutation,
            mutation_percent_genes=round(self.mutation_rate * 100),
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
        # FILTER OUT REDUNDANT POLYNOMIALS - Remove polynomials that don't add significant coverage
        filtered_polynomials = self._filter_redundant_polynomials(
            best_polynomials, data_points
        )

        # Update count after filtering
        print("Final results:")
        print(f"  Best fitness: {best_fitness:.1f}")
        print(f"  Estimated coverage: {final_coverage:.1%}")
        print(f"  Number of polynomials (before filtering): {len(best_polynomials)}")
        print(f"  Number of polynomials (after filtering): {len(filtered_polynomials)}")

        for i, poly in enumerate(filtered_polynomials):
            # Find the original point list for this polynomial
            poly_index = best_polynomials.index(poly) if poly in best_polynomials else i
            if poly_index < len(best_point_lists):
                unique_points = len(set(best_point_lists[poly_index]))
            else:
                unique_points = poly.degree + 1
            print(
                f"  Polynomial {i}: degree {poly.degree}, fitted to {unique_points} unique points"
            )

        return filtered_polynomials

    def _filter_redundant_polynomials(self, polynomials, data_points):
        """Filter polynomials using the same principled coverage loss approach."""
        if len(polynomials) <= 1:
            return polynomials

        print(
            f"  Filtering debug: evaluating {len(polynomials)} polynomials using coverage loss analysis"
        )

        # Use the same coverage loss analysis as the fitness function
        necessary_polynomials = self._analyze_polynomial_necessity(
            polynomials, data_points
        )

        print(
            f"  Coverage loss analysis: {len(necessary_polynomials)} out of {len(polynomials)} polynomials are necessary"
        )
        for i, poly in enumerate(necessary_polynomials):
            print(f"    Polynomial {i + 1}: degree {poly.degree}")

        print(f"  Final selection: {len(necessary_polynomials)} polynomials")
        return necessary_polynomials

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

        # Pre-compute spatial relationships for efficient neighbor finding
        if not hasattr(self, "_neighbor_cache"):
            self._build_neighbor_cache()

        # Convert offspring to proper numpy array and work with individual solutions
        offspring_array = np.array(offspring)

        for solution_idx in range(
            offspring_array.shape[0]
        ):  # For each solution in population
            solution = offspring_array[solution_idx]

            # Apply TWO-TIER MUTATIONS to this solution
            for gene_idx in range(len(solution)):
                if np.random.random() < self.mutation_rate:
                    current_point_idx = int(
                        solution[gene_idx]
                    )  # Extract the point index

                    # Decide mutation type: small nudges vs big changes
                    if np.random.random() < 0.95:
                        # SMALL NUDGE: Replace with nearby neighbor
                        neighbors = self._get_neighbors(current_point_idx, radius=5)
                        if neighbors:
                            solution[gene_idx] = np.random.choice(neighbors)
                        else:
                            solution[gene_idx] = np.random.randint(
                                0, len(self.data_points)
                            )
                    else:

                        solution[gene_idx] = np.random.randint(
                            0, len(self.data_points)
                        )

        return offspring_array

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
            self._neighbor_cache[i] = [
                idx for idx, dist in neighbors[:20]
            ]  # Keep 20 closest neighbors

        print(f"Built neighbor cache for {len(self.data_points)} points")

    def _get_neighbors(self, point_idx, radius=5):
        """Get spatial neighbors of a point within given radius."""
        if (
            not hasattr(self, "_neighbor_cache")
            or point_idx not in self._neighbor_cache
        ):
            return []

        all_neighbors = self._neighbor_cache[point_idx]
        return all_neighbors[: min(radius, len(all_neighbors))]

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
        """Fitness function that balances accuracy with polynomial complexity to find optimal count."""
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

            total_distance += min_distance

        # Calculate base accuracy fitness
        average_distance = total_distance / len(data_points)
        accuracy_fitness = 1 / (1 + average_distance)

        # Count effective polynomials using coverage loss analysis
        effective_polynomials = self._analyze_polynomial_necessity(
            individual.polynomials, data_points
        )
        num_effective = len(effective_polynomials)

        complexity_penalty = 1 + (len(individual.polynomials) - num_effective)

        fitness = (
            accuracy_fitness / complexity_penalty
        )

        return 1000 + fitness

    def _analyze_polynomial_necessity(self, polynomials, data_points):
        """
        Determine which polynomials are necessary by measuring coverage loss.
        A polynomial is considered necessary if removing it causes a significant coverage loss.
        """
        if len(polynomials) <= 1:
            return polynomials

        # Calculate baseline coverage with all polynomials
        baseline_coverage = self._calculate_coverage(polynomials, data_points)

        necessary_polynomials = []

        # Sort polynomials by their individual coverage contribution (descending)
        poly_coverage_scores = []
        for i, test_poly in enumerate(polynomials):
            if test_poly.degree == 0:  # Skip constant polynomials
                poly_coverage_scores.append((i, test_poly, 0))
                continue
                
            # Calculate coverage without this polynomial
            remaining_polys = [p for j, p in enumerate(polynomials) if j != i]
            if not remaining_polys:
                poly_coverage_scores.append((i, test_poly, baseline_coverage))
                continue
                
            reduced_coverage = self._calculate_coverage(remaining_polys, data_points)
            coverage_contribution = baseline_coverage - reduced_coverage
            poly_coverage_scores.append((i, test_poly, coverage_contribution))

        # Sort by coverage contribution (descending)
        poly_coverage_scores.sort(key=lambda x: x[2], reverse=True)

        # Always keep the top 2 polynomials (minimum required)
        for i in range(min(2, len(poly_coverage_scores))):
            _, poly, _ = poly_coverage_scores[i]
            necessary_polynomials.append(poly)

        # For remaining polynomials, use the coverage loss threshold
        for i in range(2, len(poly_coverage_scores)):
            _, test_poly, coverage_contribution = poly_coverage_scores[i]
            
            if baseline_coverage > 0:
                coverage_loss_ratio = coverage_contribution / baseline_coverage
            else:
                coverage_loss_ratio = 0

            # If removing this polynomial causes >4% coverage loss, it's necessary
            if coverage_loss_ratio > 0.04:
                necessary_polynomials.append(test_poly)

        return necessary_polynomials

    def _calculate_coverage(self, polynomials, data_points):
        """Calculate how many data points are covered by the given polynomials."""
        if not polynomials:
            return 0

        covered_count = 0
        coverage_threshold = (
            15.0  # Distance threshold for considering a point "covered"
        )

        for x, y in data_points:
            min_distance = float("inf")

            for poly in polynomials:
                try:
                    pred = poly.evaluate(x)
                    distance = abs(pred - y)
                    min_distance = min(min_distance, distance)
                except Exception:
                    continue

            if min_distance < coverage_threshold:
                covered_count += 1

        return covered_count
