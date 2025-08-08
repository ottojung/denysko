#!/usr/bin/env python3
"""
Test polynomial generation with guaranteed high degree.
"""

import sys
import os
import numpy as np

# Add the src directory to path  
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from polynomial_fitter import PolynomialFitter

def test_forced_high_degree():
    """Force creation of a high-degree polynomial by using many points in a single curve."""
    print("Testing forced high-degree polynomial...")
    
    # Create a single smooth curve with many points to force high degree
    # Use 15 points on a monotonic function to avoid overlap detection
    x_vals = np.linspace(0, 2, 15)  # monotonic x values
    y_vals = np.sin(x_vals) + 0.1 * x_vals**2  # smooth curve
    
    contour = np.column_stack([x_vals, y_vals])
    
    # Force a higher max_points_per_piece to get degree 14
    fitter = PolynomialFitter()
    fitter.max_points_per_piece = 20  # Allow up to 20 points per piece
    
    functions = fitter.fit_contour_polynomials(contour)
    
    print(f"Generated {len(functions)} functions:")
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func}")
        
        # Check for multi-digit exponents
        import re
        # Look for encoded exponents like ^1^4 (degree 14)
        encoded_exponents = re.findall(r'\^(\d+(?:\^\d+)+)', func)
        if encoded_exponents:
            print(f"✓ Found encoded multi-digit exponents: {encoded_exponents}")
            
        # Check for any unencoded multi-digit exponents
        raw_multdigit = re.findall(r'\^(\d{2,})(?!\^)', func)  # multi-digit not followed by ^
        if raw_multdigit:
            print(f"✗ Found unencoded multi-digit exponents: {raw_multdigit}")
        else:
            print("✓ No unencoded multi-digit exponents found")

if __name__ == "__main__":
    test_forced_high_degree()
