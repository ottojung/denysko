#!/usr/bin/env python3
"""Test cases to ensure correct polynomial counts for different letters."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_to_desmos import TextToDesmos

def test_letter_polynomial_counts():
    """Test that letters generate the expected number of polynomials."""
    print("=== Testing Expected Polynomial Counts ===")
    
    # Expected polynomial counts for each letter
    expected_counts = {
        "A": 2,  # Simple letter - should use 2 polynomials
        "C": 2,  # Simple letter - should use 2 polynomials  
        "B": 5,  # Complex letter - should use 5 polynomials
    }
    
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    results = {}
    
    for letter, expected_count in expected_counts.items():
        print(f"\n--- Testing letter '{letter}' ---")
        print(f"Expected polynomials: {expected_count}")
        
        try:
            functions = converter.text_to_desmos_functions(letter)
            actual_count = len(functions)
            
            print(f"Actual polynomials: {actual_count}")
            
            success = actual_count == expected_count
            print(f"Test result: {'PASS' if success else 'FAIL'}")
            
            if not success:
                print(f"ERROR: Expected {expected_count} polynomials, got {actual_count}")
            
            results[letter] = {
                'expected': expected_count,
                'actual': actual_count,
                'success': success,
                'functions': functions[:2]  # Show first 2 functions as examples
            }
            
        except Exception as e:
            print(f"ERROR generating letter '{letter}': {e}")
            results[letter] = {
                'expected': expected_count,
                'actual': 0,
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n=== TEST SUMMARY ===")
    total_tests = len(expected_counts)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    for letter, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        if 'error' in result:
            print(f"  {letter}: {status} - {result['error']}")
        else:
            print(f"  {letter}: {status} - Expected {result['expected']}, got {result['actual']}")
    
    if passed_tests == total_tests:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed!")
        print("\nFailed tests indicate complexity pressure is too weak.")
        print("The genetic algorithm should strongly penalize unnecessary polynomials.")
    
    return results

if __name__ == "__main__":
    test_letter_polynomial_counts()
