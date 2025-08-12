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
        
        for i in range(self.max_polynomials):
            start_idx = i * genes_per_poly
            end_idx = start_idx + genes_per_poly
            point_list = solution[start_idx:end_idx].tolist()
            point_lists.append(point_list)
        
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
        
        print(f"Starting PyGAD point-selection genetic algorithm with {len(data_points)} data points")
        print(f"Parameters: pop={self.population_size}, gen={self.max_generations}, "
              f"poly={self.max_polynomials}, degree={self.max_degree}")
        
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
            mutation_type="adaptive",  # Adaptive mutation
            mutation_percent_genes=max(10, int(self.mutation_rate * 100)),
            random_seed=None,
            suppress_warnings=True,
            on_generation=self._on_generation_callback,
            mutation_by_replacement=True,
            allow_duplicate_genes=True  # Allow same point to be used multiple times
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
        final_coverage = best_fitness / 100.0 if best_fitness <= 100 else (best_fitness - 50) / 100.0
        print("Final results:")
        print(f"  Best fitness: {best_fitness:.1f}")
        print(f"  Estimated coverage: {final_coverage:.1%}")
        print(f"  Number of polynomials: {len(best_polynomials)}")
        
        for i, poly in enumerate(best_polynomials):
            unique_points = len(set(best_point_lists[i]))
            print(f"  Polynomial {i}: degree {poly.degree}, fitted to {unique_points} unique points")
        
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
                    distances.append((dx*dx + dy*dy, j))
            
            distances.sort()
            nearby_indices = [idx for _, idx in distances[:min(10, len(distances))]]
            
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
                used_points = set()
                
                for _ in range(self.max_degree):
                    # Try to select diverse points
                    attempts = 0
                    while attempts < 20:
                        if point_candidates:
                            candidate = np.random.choice(point_candidates)
                        else:
                            candidate = np.random.randint(0, len(data_points))
                        
                        # Encourage diversity by avoiding recently used points
                        if len(used_points) < self.max_degree // 2 or candidate not in used_points:
                            poly_points.append(candidate)
                            used_points.add(candidate)
                            break
                        attempts += 1
                    
                    if len(poly_points) < len(poly_points) + 1:
                        # Fallback to random if diversity fails
                        poly_points.append(np.random.randint(0, len(data_points)))
                
                solution.extend(poly_points)
            
            population.append(solution)
        
        return np.array(population, dtype=int)
    
    def _on_generation_callback(self, ga_instance):
        """Callback function called after each generation."""
        generation = ga_instance.generations_completed
        if generation % 50 == 0 or generation == self.max_generations:
            best_fitness = ga_instance.best_solution()[1]
            coverage_pct = best_fitness / 100.0 if best_fitness <= 100 else (best_fitness - 50) / 100.0
            print(f"Generation {generation}: Best fitness = {best_fitness:.1f} "
                  f"(~{coverage_pct:.1%} coverage)")

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
        """Enhanced fitness function with multi-objective optimization."""
        if len(data_points) == 0:
            return 0.0
        
        # Step 1: Count coverage with multiple tolerance levels
        covered_points_tight = 0  # Within 3 units (strict)
        covered_points_medium = 0  # Within 5 units (medium)
        covered_points_loose = 0  # Within 8 units (loose)
        total_error = 0.0
        
        for x, y in data_points:
            best_error = float('inf')
            
            # Check if this point is covered by any polynomial
            for poly in individual.polynomials:
                try:
                    pred = poly.evaluate(x)
                    error = abs(pred - y)
                    best_error = min(best_error, error)
                except Exception:
                    continue
            
            if best_error != float('inf'):
                total_error += best_error
                
                if best_error <= 3.0:
                    covered_points_tight += 1
                    covered_points_medium += 1
                    covered_points_loose += 1
                elif best_error <= 5.0:
                    covered_points_medium += 1
                    covered_points_loose += 1
                elif best_error <= 8.0:
                    covered_points_loose += 1
        
        # Step 2: Calculate multi-tier accuracy scores
        tight_accuracy = covered_points_tight / len(data_points)
        medium_accuracy = covered_points_medium / len(data_points)
        loose_accuracy = covered_points_loose / len(data_points)
        
        # Step 3: Calculate complexity and diversity scores
        total_degree = sum(poly.degree for poly in individual.polynomials)
        max_possible_degree = self.max_polynomials * self.max_degree
        complexity_score = total_degree / max_possible_degree if max_possible_degree > 0 else 0
        
        # Diversity bonus: reward different degrees across polynomials
        degrees = [poly.degree for poly in individual.polynomials]
        unique_degrees = len(set(degrees))
        diversity_bonus = unique_degrees / len(degrees) if len(degrees) > 0 else 0
        
        # Step 4: Penalize very low degree solutions
        avg_degree = np.mean(degrees) if degrees else 0
        degree_penalty = max(0, 2.5 - avg_degree) * 5  # Penalty for avg degree < 2.5
        
        # Step 5: Combined fitness with multiple objectives
        # Primary: medium accuracy (5-unit tolerance as requested)
        # Secondary: tight accuracy bonus
        # Tertiary: complexity and diversity
        fitness = (
            medium_accuracy * 100.0 +           # Primary objective (0-100)
            tight_accuracy * 20.0 +             # Tight accuracy bonus (0-20)
            complexity_score * 5.0 +            # Complexity bonus (0-5)
            diversity_bonus * 3.0 -             # Diversity bonus (0-3)
            degree_penalty                      # Penalty for low degrees
        )
        
        # Extra bonus for very high coverage
        if medium_accuracy >= 0.99:
            fitness += 50  # Major bonus for 99%+ coverage
        elif medium_accuracy >= 0.95:
            fitness += 25
        elif medium_accuracy >= 0.90:
            fitness += 10
        
        return max(0, fitness)  # Ensure non-negative
