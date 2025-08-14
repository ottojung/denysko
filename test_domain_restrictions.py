#!/usr/bin/env python3
"""Test script to verify domain restrictions are working properly."""

from src.polynomial_fitter_genetic import PolynomialFitter
from src.function_transformer import FunctionTransformer
import numpy as np

def test_genetic_fitter_output():
    """Test that genetic fitter generates domain restrictions."""
    print("=== Testing Genetic Fitter Domain Restrictions ===")
    
    # Simple test points
    points = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (3.0, 8.0), (4.0, 13.0)]
    
    fitter = PolynomialFitter()
    result = fitter.fit_polynomial_to_trace(np.array(points))
    
    print(f"Generated {len(result)} functions:")
    for i, func in enumerate(result, 1):
        print(f"  {i}: {func}")
        # Check if domain restrictions are present
        has_domain = '{' in func and '}' in func
        print(f"     Has domain restrictions: {has_domain}")
    
    return result

def test_function_transformer():
    """Test that function transformer preserves domain restrictions."""
    print("\n=== Testing Function Transformer ===")
    
    # Test function with domain restrictions
    test_func = "y = 2.0*x^2 + 1.0*x + 3.0 {-1.00 ≤ x ≤ 5.00}"
    
    transformer = FunctionTransformer(origin=(0, 0), scale=1.0)
    
    print(f"Original:    {test_func}")
    transformed = transformer.transform_function(test_func)
    print(f"Transformed: {transformed}")
    
    simplified = transformer.simplify_function_string(transformed)
    print(f"Simplified:  {simplified}")
    
    # Check preservation
    has_domain_orig = '{' in test_func and '}' in test_func
    has_domain_final = '{' in simplified and '}' in simplified
    print(f"Domain preserved: {has_domain_orig} -> {has_domain_final}")

def main():
    """Run all tests."""
    test_genetic_fitter_output()
    test_function_transformer()
    
    print("\n=== Summary ===")
    print("If domain restrictions are missing from genetic fitter output,")
    print("the issue is in the genetic fitter polynomial string generation.")
    print("If they're present but lost after transformation,")
    print("the issue is in the function transformer pipeline.")

if __name__ == "__main__":
    main()
