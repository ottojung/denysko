#!/usr/bin/env python3
"""Test Letter B polynomial count with debugging."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_to_desmos import TextToDesmos

def test_letter_b():
    """Test polynomial count for letter B with debug output."""
    print("=== Testing Letter B Polynomial Count ===")
    
    converter = TextToDesmos()
    result = converter.text_to_desmos_functions("B")
    
    if result:
        polynomial_count = len(result)
        print(f"Letter B generated {polynomial_count} polynomials")
        print(f"Expected: 5, Got: {polynomial_count}")
        
        if polynomial_count == 5:
            print("✅ PASS - Letter B generates correct number of polynomials")
        else:
            print("❌ FAIL - Letter B does not generate expected number of polynomials")
        
        # Show the polynomials
        for i, poly_str in enumerate(result, 1):
            print(f"  Polynomial {i}: {poly_str[:80]}...")
            
        return polynomial_count == 5
    else:
        print("❌ FAIL - No result generated")
        return False

if __name__ == "__main__":
    test_letter_b()
