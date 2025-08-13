#!/usr/bin/env python3
"""Test Letter I complexity classification."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_to_desmos import TextToDesmos

def test_letter_i():
    """Test polynomial count for letter I."""
    print("=== Testing Letter I Polynomial Count ===")
    
    converter = TextToDesmos()
    result = converter.text_to_desmos_functions("I")
    
    if result:
        polynomial_count = len(result)
        print(f"Letter I generated {polynomial_count} polynomials")
        print(f"Expected: ≤3, Got: {polynomial_count}")
        
        if polynomial_count <= 3:
            print("✅ PASS - Letter I generates acceptable number of polynomials")
        else:
            print("❌ FAIL - Letter I generates too many polynomials")
        
        # Show the polynomials
        for i, poly_str in enumerate(result, 1):
            print(f"  Polynomial {i}: {poly_str[:80]}...")
            
        return polynomial_count <= 3
    else:
        print("❌ FAIL - No result generated")
        return False

if __name__ == "__main__":
    test_letter_i()
