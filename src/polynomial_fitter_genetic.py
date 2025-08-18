#!/usr/bin/env python3
"""
Updated polynomial fitter using simple search algorithm approach.
"""

import numpy as np
from .simple_polynomial_fitter import SimplePolynomialFitter


class PolynomialFitter:
    """Fit polynomials to extracted centerlines using simple search algorithm."""

    def __init__(self):
        self.simple_fitter = SimplePolynomialFitter()

    def fit_polynomial_to_trace(self, trace, target_degree=3):
        """
        Fit polynomials to a trace using simple search algorithm.

        Args:
            trace: numpy array of (x, y) points
            target_degree: ignored (kept for compatibility)

        Returns:
            List of polynomial strings in Desmos format
        """
        if len(trace) < 4:  # Need minimum points for fitting
            return []

        # Sample points if too many (search algorithm works better with fewer points)
        sampled_trace = trace
        if len(trace) > 500:
            print(f"Sampling {len(trace)} points down to 500 for search algorithm")
            indices = np.linspace(0, len(trace) - 1, 500, dtype=int)
            sampled_trace = trace[indices]

        print(f"Fitting polynomials to {len(sampled_trace)} points")

        # Use simple search algorithm to find optimal polynomials
        polynomials = self.simple_fitter.fit(sampled_trace)

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
