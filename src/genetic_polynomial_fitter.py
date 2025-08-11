#!/usr/bin/env python3
"""
Genetic algorithm-based polynomial fitter for letter shapes.
Finds the minimum number of low-degree polynomials that fit a letter shape with high accuracy.
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
        domain = f"{{x: {self.x_min:.3f} ≤ x ≤ {self.x_max:.3f}}}"
        return f"y = {result} {domain}"


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
        population_size=100,
        max_generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_degree=6,
        max_polynomials=20,
        fitness_weights={"accuracy": 10.0, "simplicity": 0.1},  # Much higher accuracy weight
    ):
        self.population_size = population_size
        self.max_generations = max_generations
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

    def _evaluate_fitness(self, individual, x_points, y_points):
        """Evaluate the fitness of an individual."""
        # Accuracy component: how well the polynomials fit the data
        total_error = 0.0
        covered_points = 0

        for x, y in zip(x_points, y_points):
            best_prediction = None
            min_error = float("inf")

            # Find the best polynomial prediction for this point
            for poly in individual.polynomials:
                pred = poly.evaluate(x)
                if pred is not None:  # Point is in polynomial domain
                    error = abs(pred - y)
                    if error < min_error:
                        min_error = error
                        best_prediction = pred

            if best_prediction is not None:
                total_error += min_error
                covered_points += 1

        # Penalize uncovered points heavily
        if covered_points == 0:
            accuracy = 0.0
        else:
            mean_error = total_error / covered_points
            uncovered_penalty = (len(x_points) - covered_points) * 50.0  # Much heavier penalty
            # Scale accuracy to be much more sensitive to error
            accuracy = 1.0 / (1.0 + mean_error * 10.0 + uncovered_penalty)

        # Simplicity component: prefer fewer polynomials and lower degrees
        num_polys_penalty = individual.num_polynomials() * 0.01  # Reduced penalty
        degree_penalty = individual.total_degree() * 0.005  # Reduced penalty
        simplicity = 1.0 / (1.0 + num_polys_penalty + degree_penalty)

        # Combined fitness with MUCH higher accuracy weight
        fitness = (
            self.fitness_weights["accuracy"] * accuracy
            + self.fitness_weights["simplicity"] * simplicity
        )

        # Exponential amplification for better solutions - but more extreme for accuracy
        fitness = accuracy**3 * simplicity**0.5

        return fitness

    def _tournament_selection(self, population, tournament_size=3):
        """Select an individual using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda ind: ind.fitness)

    def _crossover(self, parent1, parent2):
        """Create offspring by crossing over two parents."""
        # Combine polynomials from both parents
        all_polys = parent1.polynomials + parent2.polynomials

        # Randomly select polynomials for offspring
        num_polys = random.randint(1, min(self.max_polynomials, len(all_polys)))
        selected_polys = random.sample(all_polys, num_polys)

        return Individual(selected_polys)

    def _mutate(self, individual, x_points, y_points):
        """Mutate an individual."""
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(
                ["modify_coeff", "modify_domain", "add_poly", "remove_poly"]
            )

            if mutation_type == "modify_coeff" and individual.polynomials:
                # Modify coefficients of a random polynomial
                poly = random.choice(individual.polynomials)
                coeff_idx = random.randint(0, len(poly.coefficients) - 1)
                poly.coefficients[coeff_idx] += random.gauss(0, 1.0)

            elif mutation_type == "modify_domain" and individual.polynomials:
                # Modify domain of a random polynomial
                poly = random.choice(individual.polynomials)
                x_min, x_max = np.min(x_points), np.max(x_points)
                domain_size = poly.x_max - poly.x_min
                new_start = random.uniform(x_min, x_max - domain_size)
                poly.x_min = new_start
                poly.x_max = new_start + domain_size

            elif (
                mutation_type == "add_poly"
                and len(individual.polynomials) < self.max_polynomials
            ):
                # Add a new random polynomial
                new_poly = self._create_random_polynomial(x_points, y_points)
                individual.polynomials.append(new_poly)

            elif mutation_type == "remove_poly" and len(individual.polynomials) > 1:
                # Remove a random polynomial
                individual.polynomials.pop(
                    random.randint(0, len(individual.polynomials) - 1)
                )

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

        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual(x_points, y_points)
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
