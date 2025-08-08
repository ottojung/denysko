#!/usr/bin/env python3
"""
Test polynomial generation with multi-digit exponents to verify Desmos compatibility.
"""

import sys
import os
import numpy as np

# Add the src directory to path  
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from polynomial_fitter import PolynomialFitter

def test_high_degree_polynomial():
    """Test generating a polynomial with high degree (multi-digit exponent)."""
    print("Testing high-degree polynomial generation...")
    
    # Create test data that requires a high-degree polynomial
    # Generate points for a known polynomial with degree 12: y = x^12
    x_vals = np.linspace(-1, 1, 13)  # 13 points for degree 12
    y_vals = x_vals**12
    
    # Add small noise to make it more realistic
    y_vals += np.random.normal(0, 1e-6, len(y_vals))
    
    contour = np.column_stack([x_vals, y_vals])
    
    fitter = PolynomialFitter()
    functions = fitter.fit_contour_polynomials(contour)
    
    print(f"Generated {len(functions)} functions:")
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func}")
        
        # Verify that multi-digit exponents are encoded properly
        if "^1^2" in func:  # degree 12 -> 1^2
            print("✓ Found properly encoded exponent ^1^2 (degree 12)")
        
        # Check for any raw multi-digit exponents that weren't encoded
        import re
        raw_multdigit = re.findall(r'\^(\d{2,})', func)
        if raw_multdigit:
            print(f"✗ Found unencoded multi-digit exponents: {raw_multdigit}")
        else:
            print("✓ No unencoded multi-digit exponents found")

if __name__ == "__main__":
    test_high_degree_polynomial()
