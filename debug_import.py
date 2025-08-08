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

# Import directly
import polynomial_fitter
print("Available methods:", [m for m in dir(polynomial_fitter.PolynomialFitter) if not m.startswith('__')])

fitter = polynomial_fitter.PolynomialFitter()

# Test if method exists
if hasattr(fitter, '_encode_exponent'):
    print("Method exists!")
    print("Testing 23:", fitter._encode_exponent(23))
else:
    print("Method does NOT exist")
    print("Available private methods:", [m for m in dir(fitter) if m.startswith('_') and not m.startswith('__')])
