#!/usr/bin/env python3
"""Minimal test to verify the exponent encoding works in the real pipeline"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test basic functionality 
try:
    from polynomial_fitter import PolynomialFitter
    print("✓ Successfully imported PolynomialFitter from src/")
    
    fitter = PolynomialFitter()
    
    # Test if method exists
    if hasattr(fitter, '_encode_exponent'):
        print("✓ Method _encode_exponent exists")
        
        # Test the method
        result = fitter._encode_exponent(23)
        expected = "2^3"
        status = "✓" if result == expected else "✗"
        print(f"{status} fitter._encode_exponent(23) = '{result}' (expected '{expected}')")
        
        # Test a few more cases
        for power, exp in [(7, "7"), (10, "1^0"), (123, "1^2^3")]:
            result = fitter._encode_exponent(power)
            status = "✓" if result == exp else "✗"  
            print(f"{status} fitter._encode_exponent({power}) = '{result}' (expected '{exp}')")
    else:
        print("✗ Method _encode_exponent does not exist")
        print("Available methods:", [m for m in dir(fitter) if not m.startswith('__')])
        
except Exception as e:
    print(f"✗ Error importing or testing: {e}")
    import traceback
    traceback.print_exc()
