#!/usr/bin/env python3
"""
Genetic algorithm-based polynomial fitter for letter sha    def __init__(
        self,
        population_size=100,
        generations=150,
        tournament_size=5,
        crossover_rate=0.8,
        mutation_rate=0.3,
        max_polynomials=2,  # Constrain to exactly 2 polynomials for letter structure
        max_degree=6,
        fitness_weights=None,
    ):the minimum number of low-degree polynomials that fit a letter shape with high accuracy.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class Polynomial:
    """Represents a polynomial y = f(x) with coefficients and domain."""

    coefficients: List[float]  # [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
    x_min: float
    x_max: float

    def evaluate(self, x):
        """Evaluate polynomial at x."""
        if x < self.x_min or x > self.x_max:
            return None  # Outside domain

        result = 0.0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x**i)
        return result

    def degree(self):
        """Return the degree of the polynomial."""
        return len(self.coefficients) - 1

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
        domain = f"\\ \\left\\{{{self.x_min:.3f}\\le x\\le{self.x_max:.3f}\\right\\}}"
        return f"y = {result}{domain}"


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm."""

    polynomials: List[Polynomial]
    fitness: float = 0.0

    def num_polynomials(self):
        return len(self.polynomials)

    def total_degree(self):
        return sum(p.degree() for p in self.polynomials)


class GeneticPolynomialFitter:
    """Genetic algorithm for finding optimal polynomial fits."""

    def __init__(
        self,
        population_size=200,  # Increased population for better exploration
        generations=300,  # More generations for convergence
        tournament_size=5,  # Slightly larger tournaments
        crossover_rate=0.8,
        mutation_rate=0.3,  # Higher mutation for diversity
        max_polynomials=10,
        max_degree=6,
        fitness_weights=None,
    ):
        self.population_size = population_size
        self.max_generations = generations  # Fixed parameter name
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_degree = max_degree
        self.max_polynomials = max_polynomials
        self.fitness_weights = fitness_weights

    def _create_random_polynomial(self, x_points, y_points):
        """Create a random polynomial that covers some portion of the data."""
        # Choose random domain within the data range
        x_min, x_max = np.min(x_points), np.max(x_points)
        domain_size = (x_max - x_min) * random.uniform(0.1, 0.8)
        start = random.uniform(x_min, x_max - domain_size)

        poly_x_min = start
        poly_x_max = start + domain_size

        # Choose random degree
        degree = random.randint(1, self.max_degree)

        # Generate random coefficients
        coefficients = [random.uniform(-10, 10) for _ in range(degree + 1)]

        return Polynomial(coefficients, poly_x_min, poly_x_max)

    def _create_random_individual(self, x_points, y_points):
        """Create a random individual with random polynomials."""
        num_polys = random.randint(1, self.max_polynomials)
        polynomials = []

        for _ in range(num_polys):
            poly = self._create_random_polynomial(x_points, y_points)
            polynomials.append(poly)

        return Individual(polynomials)
    
    def _create_letter_structure_individual(self, x_points, y_points):
        """Create an individual with exactly 2 polynomials optimized for letter structure."""
        polynomials = []
        
        x_min, x_max = float(np.min(x_points)), float(np.max(x_points))
        x_span = x_max - x_min
        
        # Strategy: Create 2 overlapping polynomials with different domain emphasis
        if random.random() < 0.5:
            # Strategy 1: Left-heavy and right-heavy polynomials
            
            # First polynomial: emphasizes left 70% of the domain
            left_start = x_min
            left_end = x_min + 0.8 * x_span  # 80% coverage from left
            poly1 = self._create_smart_polynomial_for_region(
                x_points, y_points, left_start, left_end
            )
            
            # Second polynomial: emphasizes right 70% of the domain  
            right_start = x_min + 0.2 * x_span  # Start at 20% point
            right_end = x_max
            poly2 = self._create_smart_polynomial_for_region(
                x_points, y_points, right_start, right_end
            )
            
            polynomials = [poly1, poly2]
        
        else:
            # Strategy 2: Upper/lower region emphasis (for crossbar separation)
            
            # Analyze y-distribution to find potential crossbar region
            y_points_sorted = np.sort(y_points)
            y_median = float(np.median(y_points_sorted))
            
            # First polynomial: covers full x-range but optimized for upper region
            poly1 = self._create_smart_polynomial_for_region(
                x_points, y_points, x_min, x_max, y_bias="upper", y_center=y_median
            )
            
            # Second polynomial: covers full x-range but optimized for lower region
            poly2 = self._create_smart_polynomial_for_region(
                x_points, y_points, x_min, x_max, y_bias="lower", y_center=y_median
            )
            
            polynomials = [poly1, poly2]
        
        return Individual(polynomials)
    
    def _create_smart_polynomial_for_region(self, x_points, y_points, x_region_min, x_region_max, y_bias=None, y_center=None):
        """Create a polynomial optimized for a specific region with optional y-bias."""
        # Ensure valid region
        x_region_min = max(x_region_min, float(np.min(x_points)))
        x_region_max = min(x_region_max, float(np.max(x_points)))
        
        if x_region_max <= x_region_min:
            x_region_min = float(np.min(x_points))
            x_region_max = float(np.max(x_points))
        
        # Get points in this region for analysis
        region_mask = (x_points >= x_region_min) & (x_points <= x_region_max)
        
        # Apply y-bias filtering if specified
        if y_bias == "upper" and y_center is not None:
            upper_mask = y_points >= y_center
            region_mask = region_mask & upper_mask
        elif y_bias == "lower" and y_center is not None:
            lower_mask = y_points < y_center
            region_mask = region_mask & lower_mask
        
        if np.sum(region_mask) > 10:  # Need sufficient points
            region_x = x_points[region_mask]
            region_y = y_points[region_mask]
            
            # Fit a simple polynomial to these points to get good initial coefficients
            try:
                degree = min(3, len(region_x) - 1, self.max_degree)
                if degree >= 1:
                    coeffs = np.polyfit(region_x, region_y, degree)
                    # Convert to ascending order (constant, linear, quadratic, ...)
                    coeffs = coeffs[::-1].tolist()
                else:
                    coeffs = [float(np.mean(region_y))]
            except Exception:
                # Fallback to mean
                coeffs = [float(np.mean(region_y))]
        else:
            # Fallback: use overall data statistics
            y_mean = float(np.mean(y_points))
            coeffs = [y_mean]
        
        # Add some randomness to avoid identical polynomials
        for i in range(len(coeffs)):
            coeffs[i] += random.gauss(0, abs(coeffs[i]) * 0.1 + 0.1)
        
        # Ensure minimum degree
        while len(coeffs) < 3:  # At least quadratic
            coeffs.append(random.gauss(0, 0.1))
        
        return Polynomial(
            coefficients=coeffs,
            x_min=x_region_min,
            x_max=x_region_max
        )

    def _evaluate_fitness(self, individual, x_points, y_points):
        """Evaluate the fitness of an individual as a holistic polynomial set - optimized for letter structure."""
        if not individual.polynomials:
            return 0.0
        
        # HEAVILY penalize if not exactly 2 polynomials
        if len(individual.polynomials) != 2:
            return 0.01  # Almost zero fitness
            
        # Holistic evaluation: for each point, find the BEST prediction across ALL polynomials
        total_squared_error = 0.0
        covered_points = 0
        max_error = 0.0
        
        for x, y in zip(x_points, y_points):
            best_prediction = None
            min_squared_error = float("inf")
            
            # Find the best polynomial prediction for this point across the entire set
            for poly in individual.polynomials:
                pred = poly.evaluate(x)
                if pred is not None:  # Point is in polynomial domain
                    squared_error = (pred - y) ** 2
                    if squared_error < min_squared_error:
                        min_squared_error = squared_error
                        best_prediction = pred
            
            if best_prediction is not None:
                total_squared_error += min_squared_error
                covered_points += 1
                # Track maximum error for additional penalty
                error = abs(best_prediction - y)
                max_error = max(max_error, error)
        
        # Calculate holistic accuracy metrics
        if covered_points == 0:
            return 0.0  # No coverage = zero fitness
        
        # Coverage ratio (what percentage of points are covered)
        coverage_ratio = covered_points / len(x_points)
        
        # Domain coverage: ensure the polynomials together cover the full x-range
        x_min_data, x_max_data = float(np.min(x_points)), float(np.max(x_points))
        x_span_data = x_max_data - x_min_data
        
        # Find the union of polynomial domains
        union_x_min = min(poly.x_min for poly in individual.polynomials)
        union_x_max = max(poly.x_max for poly in individual.polynomials)
        union_span = union_x_max - union_x_min
        
        # Domain coverage ratio
        domain_coverage = min(1.0, union_span / x_span_data) if x_span_data > 0 else 1.0
        
        # RMSE calculation
        rmse = (total_squared_error / covered_points) ** 0.5
        
        # Fitness calculation - heavily focused on coverage and accuracy
        fitness = 0.0
        
        # 1. Coverage bonus (exponential reward for high coverage)
        if coverage_ratio >= 0.98:  # 98%+ coverage gets huge bonus
            coverage_bonus = 10000.0 * (coverage_ratio ** 4)
        elif coverage_ratio >= 0.90:  # 90-98% gets good bonus
            coverage_bonus = 1000.0 * (coverage_ratio ** 2)
        else:  # <90% coverage is severely penalized
            coverage_bonus = 10.0 * coverage_ratio
        
        fitness += coverage_bonus
        
        # 2. Domain coverage bonus
        if domain_coverage >= 0.95:
            domain_bonus = 1000.0 * domain_coverage
        else:
            domain_bonus = 10.0 * domain_coverage
        
        fitness += domain_bonus
        
        # 3. Accuracy bonus (inverse of RMSE)
        if rmse < 0.5:  # Very accurate fits get exponential bonus
            accuracy_bonus = 5000.0 / (1.0 + rmse ** 2)
        else:  # Poor fits get heavily penalized
            accuracy_bonus = 100.0 / (1.0 + rmse)
        
        fitness += accuracy_bonus
        
        # 4. Max error penalty
        if max_error > 2.0:  # Any point with huge error is catastrophic
            fitness *= 0.1  # Severe penalty
        elif max_error > 1.0:
            fitness *= 0.5  # Moderate penalty
        
        # 5. Tiny simplicity bonus (almost irrelevant)
        total_degree = sum(p.degree() for p in individual.polynomials)
        if total_degree <= 8:  # Reasonable complexity
            fitness += 10.0
        
        return fitness

    def _tournament_selection(self, population, tournament_size=3):
        """Select an individual using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        """Create offspring by crossing over two parents - maintain exactly 2 polynomials."""
        # Since both parents have exactly 2 polynomials, create new combinations
        
        # Strategy: Mix polynomials from both parents
        if random.random() < 0.5:
            # Take first polynomial from parent1, second from parent2
            poly1 = parent1.polynomials[0]
            poly2 = parent2.polynomials[1] if len(parent2.polynomials) > 1 else parent2.polynomials[0]
        else:
            # Take first polynomial from parent2, second from parent1
            poly1 = parent2.polynomials[0]
            poly2 = parent1.polynomials[1] if len(parent1.polynomials) > 1 else parent1.polynomials[0]
        
        return Individual([poly1, poly2])

    def _mutate(self, individual, x_points, y_points):
        """Mutate an individual - maintain exactly 2 polynomials."""
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(["modify_coeff", "modify_domain"])
            
            if mutation_type == "modify_coeff" and individual.polynomials:
                # Modify coefficients of a random polynomial
                poly = random.choice(individual.polynomials)
                coeff_idx = random.randint(0, len(poly.coefficients) - 1)
                # Smaller mutations for stability
                poly.coefficients[coeff_idx] += random.gauss(0, 0.5)
                
            elif mutation_type == "modify_domain" and individual.polynomials:
                # Modify domain of a random polynomial
                poly = random.choice(individual.polynomials)
                x_min, x_max = float(np.min(x_points)), float(np.max(x_points))
                x_span = x_max - x_min
                
                # Small domain adjustments to maintain coverage
                domain_size = poly.x_max - poly.x_min
                max_shift = min(x_span * 0.1, domain_size * 0.2)  # Limit shifts
                
                shift = random.gauss(0, max_shift)
                new_x_min = max(x_min, min(x_max - domain_size, poly.x_min + shift))
                new_x_max = new_x_min + domain_size
                
                poly.x_min = new_x_min
                poly.x_max = new_x_max

        return individual

    def fit(self, points):
        """
        Fit polynomials to the given points using genetic algorithm.

        Args:
            points: numpy array of shape (n, 2) with (x, y) coordinates

        Returns:
            List of Polynomial objects representing the best fit
        """
        x_points = points[:, 0]
        y_points = points[:, 1]

        print(f"Fitting {len(points)} points with genetic algorithm...")
        print(f"x range: [{np.min(x_points):.3f}, {np.max(x_points):.3f}]")
        print(f"y range: [{np.min(y_points):.3f}, {np.max(y_points):.3f}]")

        # Initialize population - each individual has exactly 2 polynomials for letter structure
        population = []
        for _ in range(self.population_size):
            individual = self._create_letter_structure_individual(x_points, y_points)
            individual.fitness = self._evaluate_fitness(individual, x_points, y_points)
            population.append(individual)

        # Evolution loop
        for generation in range(self.max_generations):
            # Create new generation
            new_population = []

            # Keep best individuals (elitism)
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            elite_size = max(1, self.population_size // 10)
            new_population.extend(population[:elite_size])

            # Generate offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    offspring = self._crossover(parent1, parent2)
                else:
                    # Clone
                    offspring = Individual(population[0].polynomials.copy())

                # Mutate
                offspring = self._mutate(offspring, x_points, y_points)
                offspring.fitness = self._evaluate_fitness(
                    offspring, x_points, y_points
                )
                new_population.append(offspring)

            population = new_population

            # Print progress
            if generation % 20 == 0:
                best = max(population, key=lambda ind: ind.fitness)
                print(
                    f"Generation {generation}: Best fitness = {best.fitness:.6f}, "
                    f"Polynomials = {best.num_polynomials()}, Total degree = {best.total_degree()}"
                )

        # Return best individual
        best_individual = max(population, key=lambda ind: ind.fitness)
        print(
            f"Final result: {best_individual.num_polynomials()} polynomials, "
            f"total degree = {best_individual.total_degree()}, fitness = {best_individual.fitness:.6f}"
        )

        return best_individual.polynomials
