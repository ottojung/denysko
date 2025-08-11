#!/usr/bin/env python3
"""
Genetic algorithm-based polynomial fitter for letter sha    def __init__(
        self,
        population_size=50,   # Smaller population for faster iterations
        generations=100,      # Fewer generations, focus on quality i        else:
            # Fallback: use actual data statistics for better initialization
            y_min, y_max = float(np.min(y_points)), float(np.max(y_points))
            y_median = float(np.median(y_points))
            
            # Create a polynomial that starts near actual data values
            coeffs = [
                y_median + random.gauss(0, (y_max - y_min) * 0.2),  # Near median with variation
                random.gauss(0, 3.0),  # Small linear term for normalized coordinates
                random.gauss(0, 1.0)   # Small quadratic term
            ]on
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.4,    # Higher mutation for more exploration
        max_polynomials=2,    # Constrain to exactly 2 polynomials for letter structure
        max_degree=3,         # Lower degree to prevent instability - cubic max
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
        """Evaluate polynomial at x using normalized coordinates to prevent overflow."""
        if x < self.x_min or x > self.x_max:
            return None  # Outside domain

        # Normalize x to [-1, 1] range to prevent coefficient explosion
        # This maps domain [x_min, x_max] to [-1, 1]
        x_normalized = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        
        result = 0.0
        for i, coeff in enumerate(self.coefficients):
            result += coeff * (x_normalized**i)
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
        x_min_data, x_max_data = float(np.min(x_points)), float(np.max(x_points))
        x_span = x_max_data - x_min_data

        # Random domain size (30% to 100% of data range)
        domain_size = random.uniform(0.3, 1.0) * x_span
        max_start = x_max_data - domain_size
        poly_x_min = random.uniform(x_min_data, max_start)
        poly_x_max = poly_x_min + domain_size

        # Random degree (2 to max_degree)
        degree = random.randint(2, self.max_degree)

        # Generate coefficients for NORMALIZED coordinates [-1, 1]
        # Initialize coefficients to produce values close to the actual y data
        y_min, y_max = float(np.min(y_points)), float(np.max(y_points))
        y_mean = float(np.mean(y_points))
        y_range = y_max - y_min
        
        coefficients = []
        for i in range(degree + 1):
            if i == 0:  # Constant term - MUST be close to actual data
                # Sample actual y values from the data as starting points
                sample_y_values = np.random.choice(y_points, size=min(10, len(y_points)), replace=True)
                coeff = float(np.random.choice(sample_y_values))  # Pick actual y value
                # Add small random variation
                coeff += random.gauss(0, y_range * 0.1)  # 10% variation
                coeff = max(y_min * 0.5, min(y_max * 1.5, coeff))  # Clamp to reasonable range
            elif i == 1:  # Linear term - small slope
                coeff = random.gauss(0, y_range * 0.1)  # Small slope across normalized [-1,1]
            else:  # Higher-order terms - very small
                coeff = random.gauss(0, y_range * 0.02 / i)  # Very small contributions
            
            coefficients.append(coeff)

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
                    # Use numpy polyfit but with much more conservative approach
                    coeffs = np.polyfit(region_x, region_y, degree)
                    # Convert to ascending order (constant, linear, quadratic, ...)
                    coeffs = coeffs[::-1].tolist()
                    
                    # Scale down high-degree coefficients to prevent explosive growth
                    for i in range(len(coeffs)):
                        if i > 1:  # For quadratic and higher terms
                            coeffs[i] *= 0.1 ** (i-1)  # Increasingly smaller coefficients
                            
                else:
                    coeffs = [float(np.mean(region_y))]
            except Exception:
                # Fallback to mean
                coeffs = [float(np.mean(region_y))]
        else:
            # Fallback: use overall data statistics with reasonable coefficients
            y_min, y_max = float(np.min(y_points)), float(np.max(y_points))
            
            # Create a simple polynomial with values in the right range
            # Since we use normalized coordinates, coefficients are more predictable
            coeffs = [
                random.uniform(y_min, y_max),  # Constant term - in actual data range
                random.gauss(0, 5.0),          # Linear term for normalized x in [-1,1]
                random.gauss(0, 2.0)           # Quadratic term
            ]
        
        # Add VERY small randomness to avoid identical polynomials
        for i in range(len(coeffs)):
            noise_scale = max(0.01, abs(coeffs[i]) * 0.01)  # 1% noise
            coeffs[i] += random.gauss(0, noise_scale)
        
        # Ensure minimum degree but with small coefficients
        while len(coeffs) < 3:  # At least quadratic
            coeffs.append(random.gauss(0, 0.001))  # Very small higher-order terms
        
        # Limit degree to prevent instability
        if len(coeffs) > 4:  # Max degree 3
            coeffs = coeffs[:4]
        
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
        sum_absolute_error = 0.0
        
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
                sum_absolute_error += error
        
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
        
        # RMSE and MAE calculation
        rmse = (total_squared_error / covered_points) ** 0.5
        mae = sum_absolute_error / covered_points
        
        # Debug print for the first few evaluations to see what's happening
        debug_fitness = hasattr(self, '_debug_count')
        if not debug_fitness:
            self._debug_count = 0
        
        if self._debug_count < 3:  # Only print first 3 evaluations
            print(f"\n=== Fitness Debug {self._debug_count} ===")
            print(f"Coverage: {covered_points}/{len(x_points)} = {coverage_ratio:.3f}")
            print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, Max error: {max_error:.3f}")
            print(f"Domain coverage: {domain_coverage:.3f}")
            
            # Show some example predictions
            for i, (x, y) in enumerate(zip(x_points[:3], y_points[:3])):
                for j, poly in enumerate(individual.polynomials):
                    pred = poly.evaluate(x)
                    if pred is not None:
                        error = abs(pred - y)
                        print(f"  Point {i}: x={x:.2f}, y_actual={y:.2f}, poly{j}_pred={pred:.2f}, error={error:.2f}")
            self._debug_count += 1
        
        # RANGE-AWARE FITNESS CALCULATION
        # Heavily penalize predictions outside the expected y-range
        y_data_min, y_data_max = min(y_points), max(y_points)
        y_data_center = (y_data_min + y_data_max) / 2
        
        # Count how many predictions are in reasonable range
        in_range_count = 0
        way_off_count = 0
        
        for x, y in zip(x_points, y_points):
            for poly in individual.polynomials:
                pred = poly.evaluate(x)
                if pred is not None:
                    if y_data_min <= pred <= y_data_max * 1.2:  # Allow slight overshoot
                        in_range_count += 1
                    elif pred < 0 or pred > y_data_max * 2:  # Way off
                        way_off_count += 1
        
        range_ratio = in_range_count / max(1, in_range_count + way_off_count)
        
        # Base fitness from RMSE with range bonus
        if rmse > 200:
            fitness = 0.001
        elif rmse > 100:
            fitness = 0.1 / rmse
        elif rmse > 50:
            fitness = 1.0 / rmse
        elif rmse > 20:
            fitness = 10.0 / rmse
        elif rmse > 10:
            fitness = 50.0 / rmse
        elif rmse > 5:
            fitness = 100.0 / rmse
        else:
            fitness = 1000.0 / (1.0 + rmse)
        
        # MASSIVE bonus for predictions in correct range
        if range_ratio > 0.8:
            fitness *= 10.0  # 10x bonus for mostly correct range
        elif range_ratio > 0.5:
            fitness *= 3.0   # 3x bonus for half in range
        elif range_ratio < 0.1:
            fitness *= 0.01  # Severe penalty for mostly wrong range
        
        # 2. Progressive coverage bonuses/penalties
        if coverage_ratio >= 0.95:
            fitness *= 2.0   # Good coverage bonus
        elif coverage_ratio >= 0.80:
            fitness *= 1.5   # Decent coverage bonus
        elif coverage_ratio < 0.50:
            fitness *= 0.5   # Coverage penalty, but not catastrophic
        
        # 3. Progressive max error penalties
        if max_error > 100.0:
            fitness *= 0.1   # Large error penalty
        elif max_error > 50.0:
            fitness *= 0.3   # Medium error penalty
        elif max_error > 20.0:
            fitness *= 0.7   # Small error penalty
        
        # 4. Progressive domain coverage
        if domain_coverage < 0.60:
            fitness *= 0.3   # Domain penalty, but not catastrophic
        elif domain_coverage >= 0.90:
            fitness *= 1.5   # Domain bonus
        
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
        """Mutate an individual with range-aware mutations."""
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(["modify_coeff", "modify_domain", "range_correct"])
            
            if mutation_type == "modify_coeff" and individual.polynomials:
                # Modify coefficients of a random polynomial
                poly = random.choice(individual.polynomials)
                coeff_idx = random.randint(0, len(poly.coefficients) - 1)
                
                if coeff_idx == 0:  # Constant term - keep in reasonable range
                    y_min, y_max = min(y_points), max(y_points)
                    # Small mutation but ensure it stays somewhat reasonable
                    mutation = random.gauss(0, (y_max - y_min) * 0.1)
                    new_coeff = poly.coefficients[coeff_idx] + mutation
                    # Clamp to expanded range
                    poly.coefficients[coeff_idx] = max(y_min * 0.5, min(y_max * 1.5, new_coeff))
                else:  # Higher-order terms
                    poly.coefficients[coeff_idx] += random.gauss(0, 0.5)
                    
            elif mutation_type == "range_correct" and individual.polynomials:
                # Specifically correct polynomials that are way off range
                poly = random.choice(individual.polynomials)
                y_min, y_max = min(y_points), max(y_points)
                
                # If constant term is negative or way too large, fix it
                if poly.coefficients[0] < 0 or poly.coefficients[0] > y_max * 2:
                    # Set to a random value in the correct range
                    poly.coefficients[0] = random.uniform(y_min, y_max)
                    print(f"  Range correction: fixed constant term to {poly.coefficients[0]:.2f}")
                
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
