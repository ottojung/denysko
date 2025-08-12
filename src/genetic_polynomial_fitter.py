#!/usr/bin/env python3
"""
Point-selection genetic algorithm for polynomial fitting.

The genotype is a list of lists of point indices from the input data.
Each list has exactly max_degree indices (initially all the same).
Evolution changes which points are selected for fitting.
Polynomials are fit exactly to the unique points in each list.
Fitness is coverage of all input points by all polynomials.
"""

import numpy as np
import random
from typing import List, Tuple
from dataclasses import dataclass
import warnings

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

    def _create_random_individual(self, data_points):
        """Create a random individual with strategic point selection."""
        point_lists = []

        # Extract data characteristics for strategic selection
        x_coords = np.array([x for x, y in data_points])
        y_coords = np.array([y for x, y in data_points])
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        for _ in range(self.max_polynomials):
            point_list = []

            # Strategic initialization: use different selection strategies
            strategy = random.randint(0, 3)

            if strategy == 0:
                # Random diverse selection (original approach)
                for _ in range(self.max_degree):
                    random_point_idx = random.randint(0, len(data_points) - 1)
                    point_list.append(random_point_idx)

            elif strategy == 1:
                # Extremal points strategy - select from edges and extremes
                extremal_candidates = []

                # Find points at x extremes
                x_min_idx = np.argmin(x_coords)
                x_max_idx = np.argmax(x_coords)
                y_min_idx = np.argmin(y_coords)
                y_max_idx = np.argmax(y_coords)

                extremal_candidates.extend([x_min_idx, x_max_idx, y_min_idx, y_max_idx])

                # Add some intermediate points
                x_sorted_indices = np.argsort(x_coords)
                n_points = len(data_points)
                for frac in [0.25, 0.5, 0.75]:
                    idx = x_sorted_indices[int(frac * n_points)]
                    extremal_candidates.append(idx)

                # Select from candidates
                available_candidates = list(set(extremal_candidates))
                for _ in range(self.max_degree):
                    if available_candidates:
                        chosen_idx = random.choice(available_candidates)
                        point_list.append(chosen_idx)
                        # Allow reuse with some probability
                        if random.random() < 0.5:
                            available_candidates.remove(chosen_idx)
                    else:
                        # Fallback to random
                        point_list.append(random.randint(0, len(data_points) - 1))

            elif strategy == 2:
                # Clustered selection - pick points from different regions
                x_sorted_indices = np.argsort(x_coords)
                n_points = len(data_points)

                # Divide x-range into regions
                regions = []
                region_size = n_points // self.max_degree
                for i in range(self.max_degree):
                    start_idx = i * region_size
                    end_idx = min((i + 1) * region_size, n_points - 1)
                    if start_idx < n_points:
                        regions.append(x_sorted_indices[start_idx : end_idx + 1])

                # Select one point from each region
                for region in regions:
                    if len(region) > 0:
                        chosen_idx = random.choice(region)
                        point_list.append(chosen_idx)

                # Fill remaining slots if needed
                while len(point_list) < self.max_degree:
                    point_list.append(random.randint(0, len(data_points) - 1))

            else:  # strategy == 3
                # High-curvature selection - prefer points with high local variation
                curvature_scores = []
                for i, (x, y) in enumerate(data_points):
                    # Calculate local variation score
                    distances = [(abs(x - x2) + abs(y - y2)) for x2, y2 in data_points]
                    distances.sort()

                    # Score based on local neighborhood variation
                    nearby_points = []
                    for j, dist in enumerate(distances[: min(5, len(distances))]):
                        if dist < (x_max - x_min) * 0.1:  # Nearby points
                            nearby_points.append(j)

                    if len(nearby_points) > 1:
                        nearby_y = [data_points[idx][1] for idx in nearby_points]
                        variation = np.std(nearby_y) if len(nearby_y) > 1 else 0
                    else:
                        variation = 0

                    curvature_scores.append(variation)

                # Select points with higher curvature scores
                high_curvature_indices = np.argsort(curvature_scores)[
                    -min(20, len(data_points)) :
                ]

                for _ in range(self.max_degree):
                    if len(high_curvature_indices) > 0:
                        chosen_idx = random.choice(high_curvature_indices)
                        point_list.append(chosen_idx)
                    else:
                        point_list.append(random.randint(0, len(data_points) - 1))

            point_lists.append(point_list)

        polynomials = []
        for point_list in point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            polynomials.append(poly)

        return Individual(point_lists, polynomials)

    def _evaluate_fitness(self, individual, data_points):
        """Evaluate fitness based on coverage of all data points with adaptive tolerance."""
        covered_points = 0
        base_tolerance = 5.0

        # Calculate adaptive tolerance based on data distribution
        x_coords = [x for x, y in data_points]
        y_coords = [y for x, y in data_points]
        x_span = max(x_coords) - min(x_coords)
        y_span = max(y_coords) - min(y_coords)

        # Use tighter tolerance for smaller spans, looser for larger spans
        adaptive_tolerance = min(base_tolerance, max(2.0, min(x_span, y_span) * 0.02))

        # Multi-tier coverage evaluation
        covered_tight = 0  # Within adaptive_tolerance
        covered_loose = 0  # Within base_tolerance
        total_error = 0.0

        for x, y in data_points:
            best_pred = None
            min_error = float("inf")

            for poly in individual.polynomials:
                try:
                    pred = poly.evaluate(x)
                    error = abs(pred - y)
                    if error < min_error:
                        min_error = error
                        best_pred = pred
                except Exception:
                    continue

            if best_pred is not None:
                total_error += min_error
                if min_error <= adaptive_tolerance:
                    covered_tight += 1
                    covered_points += 1
                elif min_error <= base_tolerance:
                    covered_loose += 1
                    covered_points += 1

        coverage_ratio = (
            covered_points / len(data_points) if len(data_points) > 0 else 0
        )
        tight_ratio = covered_tight / len(data_points) if len(data_points) > 0 else 0

        # Enhanced fitness calculation prioritizing tight coverage
        fitness = coverage_ratio * 100 + tight_ratio * 20

        # Bonus system for high coverage
        if coverage_ratio >= 0.99:
            fitness += 100  # Major bonus for 99%+
        elif coverage_ratio >= 0.98:
            fitness += 75
        elif coverage_ratio >= 0.95:
            fitness += 50
        elif coverage_ratio >= 0.90:
            fitness += 25

        # Penalty for high average error
        if len(data_points) > 0:
            avg_error = total_error / len(data_points)
            if avg_error > adaptive_tolerance:
                fitness -= (avg_error - adaptive_tolerance) * 2

        return fitness

    def _tournament_selection(self, population):
        """Select an individual using tournament selection."""
        tournament_size = min(self.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2, data_points):
        """Create offspring by crossing over point selections."""
        offspring_point_lists = []

        for i in range(len(parent1.point_lists)):
            if random.random() < 0.5:
                point_list = parent1.point_lists[i].copy()
                for j in range(len(point_list)):
                    if random.random() < 0.3 and i < len(parent2.point_lists):
                        point_list[j] = parent2.point_lists[i][j]
            else:
                point_list = parent2.point_lists[i].copy()
                for j in range(len(point_list)):
                    if random.random() < 0.3:
                        point_list[j] = parent1.point_lists[i][j]

            offspring_point_lists.append(point_list)

        polynomials = []
        for point_list in offspring_point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            polynomials.append(poly)

        return Individual(offspring_point_lists, polynomials)

    def _mutate(self, individual, data_points):
        """Enhanced mutation with local search and strategic improvements."""
        # Calculate current coverage for targeted improvements
        uncovered_points = []
        tolerance = 5.0

        for i, (x, y) in enumerate(data_points):
            best_error = float("inf")
            for poly in individual.polynomials:
                try:
                    pred = poly.evaluate(x)
                    error = abs(pred - y)
                    best_error = min(best_error, error)
                except Exception:
                    continue

            if best_error > tolerance:
                uncovered_points.append(i)

        for poly_idx, point_list in enumerate(individual.point_lists):
            # Standard random mutation
            for i in range(len(point_list)):
                if random.random() < self.mutation_rate:
                    point_list[i] = random.randint(0, len(data_points) - 1)

            # Targeted mutation: bias toward uncovered points
            if uncovered_points and random.random() < self.mutation_rate * 2:
                pos_to_change = random.randint(0, len(point_list) - 1)
                # Prefer uncovered points
                if random.random() < 0.7:
                    point_list[pos_to_change] = random.choice(uncovered_points)
                else:
                    point_list[pos_to_change] = random.randint(0, len(data_points) - 1)

            # Diversity-promoting mutation
            if random.random() < self.mutation_rate * 0.5:
                unique_points = len(set(point_list))

                # If degree is too low, force some diversity
                if unique_points < min(self.max_degree, 4):
                    pos_to_change = random.randint(0, len(point_list) - 1)
                    current_points = set(point_list)

                    # Find a point not currently in the list
                    attempts = 0
                    while attempts < 20:
                        if uncovered_points and random.random() < 0.6:
                            # Prefer uncovered points for diversity
                            new_point = random.choice(uncovered_points)
                        else:
                            new_point = random.randint(0, len(data_points) - 1)

                        if new_point not in current_points:
                            point_list[pos_to_change] = new_point
                            break
                        attempts += 1

            # Local search improvement
            if random.random() < self.mutation_rate * 0.3:
                self._local_search_improvement(
                    point_list, data_points, poly_idx, individual
                )

        # Refit polynomials after mutation
        individual.polynomials = []
        for point_list in individual.point_lists:
            poly = self._fit_polynomial_to_points(point_list, data_points)
            individual.polynomials.append(poly)

        return individual

    def _local_search_improvement(self, point_list, data_points, poly_idx, individual):
        """Local search to improve coverage of current polynomial."""
        # Try replacing each point with nearby alternatives
        for i in range(len(point_list)):
            current_point_idx = point_list[i]
            current_x, current_y = data_points[current_point_idx]

            # Find nearby points
            distances = []
            for j, (x, y) in enumerate(data_points):
                dist = abs(x - current_x) + abs(y - current_y)
                distances.append((dist, j))

            distances.sort()
            nearby_candidates = [
                idx for dist, idx in distances[: min(10, len(distances))]
            ]

            # Test each nearby candidate
            best_improvement = 0
            best_candidate = current_point_idx

            for candidate_idx in nearby_candidates:
                if candidate_idx == current_point_idx:
                    continue

                # Temporarily replace point
                original_idx = point_list[i]
                point_list[i] = candidate_idx

                # Test if this improves coverage
                test_poly = self._fit_polynomial_to_points(point_list, data_points)
                if test_poly:
                    # Count coverage improvement
                    improvement = self._count_coverage_improvement(
                        test_poly, individual.polynomials, data_points, poly_idx
                    )
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_candidate = candidate_idx

                # Restore original
                point_list[i] = original_idx

            # Apply best improvement if found
            if best_candidate != current_point_idx:
                point_list[i] = best_candidate
                break  # Only improve one point per call to avoid overfitting

    def _count_coverage_improvement(self, new_poly, all_polys, data_points, poly_idx):
        """Count how many additional points would be covered with the new polynomial."""
        tolerance = 5.0
        improvement = 0

        for x, y in data_points:
            # Check if currently uncovered
            currently_covered = False
            for i, poly in enumerate(all_polys):
                if i == poly_idx:
                    continue  # Skip the polynomial we're replacing
                try:
                    pred = poly.evaluate(x)
                    if abs(pred - y) <= tolerance:
                        currently_covered = True
                        break
                except Exception:
                    continue

            if not currently_covered:
                # Check if new polynomial would cover it
                try:
                    new_pred = new_poly.evaluate(x)
                    if abs(new_pred - y) <= tolerance:
                        improvement += 1
                except Exception:
                    pass

        return improvement

    def fit(self, data_points):
        """Fit polynomials to data points using point-selection genetic algorithm."""
        data_points = [(float(p[0]), float(p[1])) for p in data_points]

        print(
            f"Starting point-selection genetic algorithm with {len(data_points)} data points"
        )
        print(
            f"Parameters: pop={self.population_size}, gen={self.max_generations}, "
            f"poly={self.max_polynomials}, degree={self.max_degree}"
        )

        population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual(data_points)
            individual.fitness = self._evaluate_fitness(individual, data_points)
            population.append(individual)

        for generation in range(self.max_generations):
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            current_best = population[0].fitness

            if generation % 50 == 0 or generation == self.max_generations - 1:
                coverage_pct = (
                    current_best / 100.0
                    if current_best <= 100
                    else (current_best - 50) / 100.0
                )
                print(
                    f"Generation {generation}: Best fitness = {current_best:.1f} "
                    f"(~{coverage_pct:.1%} coverage)"
                )

            new_population = []

            elite_size = max(1, self.population_size // 10)
            for i in range(elite_size):
                new_population.append(population[i])

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                if random.random() < self.crossover_rate:
                    offspring = self._crossover(parent1, parent2, data_points)
                else:
                    offspring = parent1

                offspring = self._mutate(offspring, data_points)
                offspring.fitness = self._evaluate_fitness(offspring, data_points)
                new_population.append(offspring)

            population = new_population

        population.sort(key=lambda ind: ind.fitness, reverse=True)
        best_individual = population[0]

        final_coverage = (
            best_individual.fitness / 100.0
            if best_individual.fitness <= 100
            else (best_individual.fitness - 50) / 100.0
        )
        print("Final results:")
        print(f"  Best fitness: {best_individual.fitness:.1f}")
        print(f"  Estimated coverage: {final_coverage:.1%}")
        print(f"  Number of polynomials: {len(best_individual.polynomials)}")

        for i, poly in enumerate(best_individual.polynomials):
            unique_points = len(set(best_individual.point_lists[i]))
            print(
                f"  Polynomial {i}: degree {poly.degree}, fitted to {unique_points} unique points"
            )

        return best_individual.polynomials
