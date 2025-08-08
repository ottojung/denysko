#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import importlib
import polynomial_fitter
importlib.reload(polynomial_fitter)  # Force reload

fitter = polynomial_fitter.PolynomialFitter()

print("Testing exponent encoding:")
test_cases = [7, 8, 9, 10, 12, 23, 100, 123, 1234]

for power in test_cases:
    try:
        result = fitter._encode_exponent(power)
        print(f"  {power} -> {result}")
    except Exception as e:
        print(f"  {power} -> ERROR: {e}")
        print(f"Available methods: {[m for m in dir(fitter) if not m.startswith('__')]}")
        break
