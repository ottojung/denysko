#!/usr/bin/env python3
"""
Test the exponent encoding for Desmos compatibility.
"""

import sys
import os

# Add the src directory to path  
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from polynomial_fitter import PolynomialFitter

def test_exponent_encoding():
    """Test various multi-digit exponents get encoded properly."""
    fitter = PolynomialFitter()
    
    test_cases = [
        (7, "7"),       # single digit, unchanged
        (8, "8"),       # single digit, unchanged  
        (9, "9"),       # single digit, unchanged
        (10, "1^0"),    # 10 -> "1^0" (Desmos reads as x^10)
        (12, "1^2"),    # 12 -> "1^2" (Desmos reads as x^12)
        (23, "2^3"),    # 23 -> "2^3" (Desmos reads as x^23)
        (100, "1^0^0"), # 100 -> "1^0^0" (Desmos reads as x^100)
        (256, "2^5^6"), # 256 -> "2^5^6" (Desmos reads as x^256)
        (1234, "1^2^3^4"), # 1234 -> "1^2^3^4" (Desmos reads as x^1234)
    ]
    
    print("Testing exponent encoding:")
    for power, expected in test_cases:
        result = fitter._encode_exponent(power)
        status = "✓" if result == expected else "✗"
        print(f"  {power:4d} -> {result:>10} (expected {expected:>10}) {status}")
        
        # Verify that when we join the digits back, we get the original power
        if '^' in result:
            digits = result.split('^')
            reconstructed = int(''.join(digits))
            if reconstructed != power:
                print(f"    ERROR: digits {''.join(digits)} = {reconstructed}, not {power}")
        else:
            if int(result) != power:
                print(f"    ERROR: {result} != {power}")

if __name__ == "__main__":
    test_exponent_encoding()
