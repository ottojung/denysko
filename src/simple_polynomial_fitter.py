#!/usr/bin/env python3
"""
Simple search algorithm for polynomial fitting.

Starting with two points, applies random perturbations to find better polynomial fits.
"""

import numpy as np
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


class SimplePolynomialFitter:
    """Simple search algorithm for polynomial fitting."""

    def __init__(
        self,
        max_iterations=10000,
        max_points_per_poly=5,
        max_polynomials=2,
    ):
        self.max_iterations = max_iterations
        self.max_points_per_poly = max_points_per_poly
        self.max_polynomials = max_polynomials
        self.data_points = None
        self.neighbor_cache = None

    def fit(self, data_points):
        """
        Fit polynomials to data points using simple search algorithm.
        """
        self.data_points = [(float(p[0]), float(p[1])) for p in data_points]

        print(
            f"Starting simple search algorithm with {len(self.data_points)} data points"
        )
        print(
            f"Parameters: max_iter={self.max_iterations}, max_points={self.max_points_per_poly}, max_polys={self.max_polynomials}"
        )

        # Build neighbor cache
        self._build_neighbor_cache()

        # Start with one polynomial using two random points
        current_solution = [self._get_two_random_points()]
        current_error = self._calculate_error(current_solution)

        best_solution = current_solution.copy()
        best_error = current_error

        print(f"Initial solution: 1 polynomial with 2 points, error: {best_error:.2f}")

        # Search for better solutions
        for iteration in range(self.max_iterations):
            # Try a random perturbation
            new_solution = self._apply_perturbation(current_solution.copy())
            new_error = self._calculate_error(new_solution)

            # Accept if better
            if new_error < current_error:
                current_solution = new_solution
                current_error = new_error

                # Update best if this is the best so far
                if new_error < best_error:
                    best_solution = new_solution.copy()
                    best_error = new_error

            # Progress reporting
            if (iteration + 1) % 200 == 0:
                num_polys = len(best_solution)
                total_points = sum(len(poly_points) for poly_points in best_solution)
                print(
                    f"Iteration {iteration + 1}: Best error = {best_error:.2f}, {num_polys} polynomials, {total_points} total points"
                )

        # Convert best solution to polynomials
        polynomials = []
        for poly_points in best_solution:
            poly = self._fit_polynomial_to_points(poly_points)
            polynomials.append(poly)

        # Final results
        print("Final results:")
        print(f"  Best error: {best_error:.2f}")
        print(f"  Number of polynomials: {len(polynomials)}")
        for i, poly in enumerate(polynomials):
            print(
                f"  Polynomial {i}: degree {poly.degree}, fitted to {len(poly.fit_points)} points"
            )

        return polynomials

    def _get_two_random_points(self):
        """Get two random point indices."""
        indices = np.random.choice(len(self.data_points), 2, replace=False)
        return indices.tolist()

    def _build_neighbor_cache(self):
        """Build cache of immediate neighbors for each point."""
        print("Building neighbor cache...")
        self.neighbor_cache = {}

        for i, (x1, y1) in enumerate(self.data_points):
            neighbors = []
            for j, (x2, y2) in enumerate(self.data_points):
                if i != j:
                    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    neighbors.append((j, distance))

            # Sort by distance and keep closest neighbors
            neighbors.sort(key=lambda x: x[1])
            self.neighbor_cache[i] = [
                idx for idx, _ in neighbors[:5]
            ]  # Keep 5 closest neighbors

        print(f"Built neighbor cache for {len(self.data_points)} points")

    def _apply_perturbation(self, solution):
        """Apply a random perturbation to the solution."""
        rand = np.random.random()

        if rand < 0.80:  # 80% - Replace point with immediate neighbor
            return self._replace_with_neighbor(solution)
        elif rand < 0.95:  # 15% - Replace point with random point
            return self._replace_with_random(solution)
        elif rand < 0.98:  # 3% - Add random point to existing polynomial
            return self._add_point_to_polynomial(solution)
        else:  # 2% - Add new polynomial
            return self._add_new_polynomial(solution)

    def _replace_with_neighbor(self, solution):
        """Replace one point with its immediate neighbor."""
        if not solution:
            return solution

        # Pick random polynomial
        poly_idx = np.random.randint(len(solution))
        poly_points = solution[poly_idx]

        if not poly_points:
            return solution

        # Pick random point in that polynomial
        point_idx = np.random.randint(len(poly_points))
        current_point = poly_points[point_idx]

        # Get neighbors
        neighbors = self.neighbor_cache.get(current_point, [])
        if neighbors:
            new_point = np.random.choice(neighbors)
            solution[poly_idx][point_idx] = new_point

        return solution

    def _replace_with_random(self, solution):
        """Replace one point with a random point."""
        if not solution:
            return solution

        # Pick random polynomial
        poly_idx = np.random.randint(len(solution))
        poly_points = solution[poly_idx]

        if not poly_points:
            return solution

        # Pick random point in that polynomial
        point_idx = np.random.randint(len(poly_points))

        # Replace with random point
        new_point = np.random.randint(len(self.data_points))
        solution[poly_idx][point_idx] = new_point

        return solution

    def _add_point_to_polynomial(self, solution):
        """Add a random point to an existing polynomial."""
        if not solution:
            return solution

        # Pick random polynomial
        poly_idx = np.random.randint(len(solution))
        poly_points = solution[poly_idx]

        # Check if we can add more points
        if len(poly_points) >= self.max_points_per_poly:
            return solution

        # Add random point
        new_point = np.random.randint(len(self.data_points))
        solution[poly_idx].append(new_point)

        return solution

    def _add_new_polynomial(self, solution):
        """Add a new polynomial with two random points."""
        # Check if we can add more polynomials
        if len(solution) >= self.max_polynomials:
            return solution

        # Add new polynomial with two random points
        new_poly = self._get_two_random_points()
        solution.append(new_poly)

        return solution

    def _fit_polynomial_to_points(self, point_indices):
        """Fit polynomial to the specified points."""
        unique_indices = list(set(point_indices))
        if len(unique_indices) < 2:
            if len(point_indices) >= 1:
                unique_indices = [point_indices[0], point_indices[0]]
            else:
                unique_indices = [0, 0]

        fit_points = [self.data_points[i] for i in unique_indices]
        x_vals = [p[0] for p in fit_points]
        y_vals = [p[1] for p in fit_points]

        degree = len(unique_indices) - 1
        degree = max(1, min(degree, 2))

        try:
            if degree == 0:
                coefficients = [np.mean(y_vals)]
            else:
                coefficients = np.polyfit(x_vals, y_vals, degree)
                coefficients = coefficients[::-1].tolist()
            return Polynomial(coefficients, fit_points, degree)
        except Exception:
            mean_y = np.mean(y_vals) if y_vals else 0.0
            return Polynomial([mean_y], fit_points, 0)

    def _calculate_error(self, solution):
        """Calculate total error for the solution using neighbor-based acceptance threshold.

        Procedure:
        - Compute average neighbor distance and its standard deviation from the neighbor cache.
        - Acceptable distance = 2 * avg_neighbor_distance + 2 * std_neighbor_distance
        - For each data point, compute the minimum distance to any polynomial and include it
          in the error sum only if it's <= acceptable distance.
        - Return the average of included distances. If none included, return a large penalty.
        """
        if not solution:
            return float("inf")

        # Fit polynomials for current solution
        polynomials = []
        for poly_points in solution:
            poly = self._fit_polynomial_to_points(poly_points)
            polynomials.append(poly)

        # Build list of neighbor distances from neighbor_cache
        neighbor_distances = []
        if self.neighbor_cache is not None:
            for i, neighs in self.neighbor_cache.items():
                x1, y1 = self.data_points[i]
                for n in neighs:
                    x2, y2 = self.data_points[n]
                    d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    neighbor_distances.append(d)

        # Fallback if neighbor distances not available
        if not neighbor_distances:
            avg_neighbor = 15.0
            std_neighbor = 0.0
        else:
            neighbor_arr = np.array(neighbor_distances)
            avg_neighbor = float(np.mean(neighbor_arr))
            std_neighbor = float(np.std(neighbor_arr))

        acceptable_distance = 2.0 * avg_neighbor + 2.0 * std_neighbor

        score = 0.0

        # Calculate minimum distance for each data point and include only if acceptable
        for x, y in self.data_points:
            min_distance = float("inf")

            for poly in polynomials:
                pred = poly.evaluate(x)
                distance = abs(pred - y)
                if distance < min_distance:
                    min_distance = distance

            if min_distance == float("inf"):
                raise RuntimeError("Impossible to evaluate polynomial for point")

            if min_distance <= acceptable_distance:
                score += 1

        return score
