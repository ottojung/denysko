#!/usr/bin/env python3
"""
Updated polynomial fitter using genetic algorithm approach.
"""

import numpy as np
from .genetic_polynomial_fitter import GeneticPolynomialFitter


class PolynomialFitter:
    """Fit polynomials to extracted centerlines using genetic algorithm."""

    def __init__(self):
        self.ga_fitter = GeneticPolynomialFitter()

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

        # Calculate complexity from FULL trace before sampling
        full_complexity = self._calculate_complexity_from_full_trace(trace)
        print(f"Full trace complexity analysis: max_polynomials = {full_complexity}")

        # Sample points if too many (genetic algorithm works better with fewer points)
        sampled_trace = trace
        if len(trace) > 500:
            print(f"Sampling {len(trace)} points down to 500 for genetic algorithm")
            indices = np.linspace(0, len(trace) - 1, 500, dtype=int)
            sampled_trace = trace[indices]

        print(f"Fitting genetic polynomials to {len(sampled_trace)} points")

        # Use genetic algorithm to find optimal polynomials with complexity from full trace
        polynomials = self.ga_fitter.fit(sampled_trace)

        # Convert to Desmos format strings
        result = []
        for poly in polynomials:
            result.append(str(poly))

        print(f"Generated {len(result)} polynomial functions")
        return result
    
    def _calculate_complexity_from_full_trace(self, trace):
        """Calculate complexity metrics from full trace to determine max polynomials."""
        if len(trace) == 0:
            return 2
            
        # Extract coordinates
        x_coords = [p[0] for p in trace]
        y_coords = [p[1] for p in trace]
        
        if len(x_coords) == 0:
            return 2
            
        x_span = max(x_coords) - min(x_coords)
        y_span = max(y_coords) - min(y_coords)
        
        if x_span == 0 or y_span == 0:
            return 2
            
        # Calculate metrics
        point_density = len(trace) / (x_span * y_span) if (x_span * y_span) > 0 else 0
        compactness_ratio = y_span / x_span
        
        print(f"    Full trace metrics: points={len(trace)}, density={point_density:.1f}, compactness={compactness_ratio:.2f}")
        
        # Decision based on empirical data:
        # B: density=19.3, compactness=1.68 → 5 polynomials
        # A: density=10.8, compactness=1.18 → 2 polynomials  
        # C: density=10.1, compactness=1.51 → 2 polynomials
        # I: density=114.1, compactness=27.75 → 2 polynomials (special case: very tall, thin letter)
        
        # Special case for very tall, thin letters like I
        if compactness_ratio > 20.0:
            print("    -> Tall thin letter (2 polynomials)")
            return 2
        elif point_density > 18.0 and compactness_ratio > 1.5 and compactness_ratio < 5.0:
            print("    -> Complex letter (5 polynomials)")
            return 5
        elif point_density > 15.0 or compactness_ratio > 1.6:
            print("    -> Medium complexity letter (3 polynomials)")  
            return 3
        else:
            print("    -> Simple letter (2 polynomials)")
            return 2

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
