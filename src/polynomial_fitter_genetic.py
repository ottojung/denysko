#!/usr/bin/env python3
"""
Updated polynomial fitter using genetic algorithm approach.
"""

import numpy as np
from .genetic_polynomial_fitter import GeneticPolynomialFitter


class PolynomialFitter:
    """Fit polynomials to extracted centerlines using genetic algorithm."""

    def __init__(self):
        # Create adaptive genetic algorithm fitter that handles both simple and complex letters
        self.ga_fitter = GeneticPolynomialFitter(
            population_size=120,  # Good population for exploration
            generations=120,      # Sufficient generations
            max_polynomials=4,    # Allow up to 4 polynomials, adaptive penalty will choose optimal number
            max_degree=5,         # Allow high degrees, but complexity penalty will control
            mutation_rate=0.05,   # Low mutation rate to preserve good solutions
            tournament_size=5     # Good selection pressure
        )

    def fit_polynomial_to_trace(self, trace, target_degree=3):
        """
        Fit polynomials to a trace using genetic algorithm.

        Args:
            trace: numpy array of (x, y) points
            target_degree: ignored (kept for compatibility)

        Returns:
            List of polynomial strings in Desmos format
        """
        if len(trace) < 4:  # Need minimum points for fitting
            return []

        # Sample points if too many (genetic algorithm works better with fewer points)
        if len(trace) > 500:
            print(f"Sampling {len(trace)} points down to 500 for genetic algorithm")
            indices = np.linspace(0, len(trace) - 1, 500, dtype=int)
            trace = trace[indices]

        print(f"Fitting genetic polynomials to {len(trace)} points")

        # Use genetic algorithm to find optimal polynomials
        polynomials = self.ga_fitter.fit(trace)

        # Convert to Desmos format strings
        result = []
        for poly in polynomials:
            result.append(str(poly))

        print(f"Generated {len(result)} polynomial functions")
        return result

    def fit_all_traces(self, traces, target_degree=3):
        """
        Fit polynomials to all traces.

        Args:
            traces: list of numpy arrays, each containing (x, y) points
            target_degree: ignored (kept for compatibility)

        Returns:
            List of polynomial function strings
        """
        all_functions = []

        for i, trace in enumerate(traces):
            print(f"\n--- Processing trace {i + 1}/{len(traces)} ---")
            functions = self.fit_polynomial_to_trace(trace, target_degree)
            all_functions.extend(functions)

        return all_functions
