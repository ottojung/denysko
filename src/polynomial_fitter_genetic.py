#!/usr/bin/env python3
"""
Updated polynomial fitter using genetic algorithm approach.
"""

import numpy as np
from .genetic_polynomial_fitter import GeneticPolynomialFitter


class PolynomialFitter:
    """Fit polynomials to extracted centerlines using genetic algorithm."""
    
    def __init__(self):
        self.genetic_fitter = GeneticPolynomialFitter(
            population_size=50,
            max_generations=100,
            max_degree=5,
            max_polynomials=15,
            fitness_weights={'accuracy': 2.0, 'simplicity': 1.0}
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
        
        print(f"Fitting genetic polynomials to {len(trace)} points")
        
        # Use genetic algorithm to find optimal polynomials
        polynomials = self.genetic_fitter.fit(trace)
        
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
            print(f"\n--- Processing trace {i+1}/{len(traces)} ---")
            functions = self.fit_polynomial_to_trace(trace, target_degree)
            all_functions.extend(functions)
        
        return all_functions
