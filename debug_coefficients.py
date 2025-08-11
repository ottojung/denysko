#!/usr/bin/env python3
"""
Debug script to test polynomial coefficient initialization.
"""

import random
import numpy as np
from src.genetic_polynomial_fitter import GeneticPolynomialFitter

def test_coefficient_initialization():
    """Test coefficient initialization to see what values we're getting."""
    print("Testing coefficient initialization...")
    
    # Simulate letter A data
    np.random.seed(42)  # For reproducibility
    x_points = np.random.uniform(11.0, 58.0, 100)
    y_points = np.random.uniform(8.0, 64.0, 100)
    
    print(f"Data ranges: x=[{np.min(x_points):.1f}, {np.max(x_points):.1f}], y=[{np.min(y_points):.1f}, {np.max(y_points):.1f}]")
    
    # Create fitter and test polynomial creation
    fitter = GeneticPolynomialFitter()
    
    # Test random polynomial creation
    print("\nTesting random polynomial creation (5 samples):")
    for i in range(5):
        poly = fitter._create_random_polynomial(x_points, y_points)
        print(f"  Polynomial {i}: coeffs={poly.coefficients}, domain=[{poly.x_min:.1f}, {poly.x_max:.1f}]")
        
        # Test evaluation at a few points
        test_x = [20.0, 35.0, 50.0]
        for x in test_x:
            pred = poly.evaluate(x)
            if pred is not None:
                print(f"    x={x:.1f} -> y={pred:.2f}")
            else:
                print(f"    x={x:.1f} -> outside domain")
    
    print("\nTesting smart polynomial creation (3 samples):")
    for i in range(3):
        # Create a region for smart polynomial
        x_region_min = random.uniform(11.0, 40.0)
        x_region_max = random.uniform(40.0, 58.0)
        
        # Get points in this region
        mask = (x_points >= x_region_min) & (x_points <= x_region_max)
        if np.sum(mask) > 0:
            region_x = x_points[mask]
            region_y = y_points[mask]
            
            poly = fitter._create_smart_polynomial_for_region(
                x_region_min, x_region_max, region_x, region_y, x_points, y_points
            )
            print(f"  Smart poly {i}: coeffs={poly.coefficients}, domain=[{poly.x_min:.1f}, {poly.x_max:.1f}]")
            
            # Test evaluation
            for x in [25.0, 35.0, 45.0]:
                pred = poly.evaluate(x)
                if pred is not None:
                    print(f"    x={x:.1f} -> y={pred:.2f}")

if __name__ == "__main__":
    test_coefficient_initialization()
