#!/usr/bin/env python3
"""
Comprehensive tests to verify complexity requirements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_to_desmos import TextToDesmos
import unittest


class TestComplexityRequirements(unittest.TestCase):
    """Test complexity requirements for different letters."""
    
    def setUp(self):
        """Set up test converter."""
        self.converter = TextToDesmos()
    
    def test_letter_a_requirement(self):
        """Letter A should output exactly 2 polynomials."""
        print("\n=== Testing Letter A ===")
        result = self.converter.text_to_desmos_functions("A")
        
        self.assertIsNotNone(result, "Letter A should generate polynomials")
        polynomial_count = len(result)
        
        print(f"Letter A generated {polynomial_count} polynomials")
        for i, poly in enumerate(result, 1):
            print(f"  Polynomial {i}: {poly[:80]}...")
            
        self.assertEqual(polynomial_count, 2, 
                        f"Letter A should generate exactly 2 polynomials, got {polynomial_count}")
        
    def test_letter_c_requirement(self):
        """Letter C should output exactly 2 polynomials."""
        print("\n=== Testing Letter C ===")
        result = self.converter.text_to_desmos_functions("C")
        
        self.assertIsNotNone(result, "Letter C should generate polynomials")
        polynomial_count = len(result)
        
        print(f"Letter C generated {polynomial_count} polynomials")
        for i, poly in enumerate(result, 1):
            print(f"  Polynomial {i}: {poly[:80]}...")
            
        self.assertEqual(polynomial_count, 2, 
                        f"Letter C should generate exactly 2 polynomials, got {polynomial_count}")
    
    def test_letter_b_requirement(self):
        """Letter B should output exactly 5 polynomials."""
        print("\n=== Testing Letter B ===")
        result = self.converter.text_to_desmos_functions("B")
        
        self.assertIsNotNone(result, "Letter B should generate polynomials")
        polynomial_count = len(result)
        
        print(f"Letter B generated {polynomial_count} polynomials")
        for i, poly in enumerate(result, 1):
            print(f"  Polynomial {i}: {poly[:80]}...")
            
        self.assertEqual(polynomial_count, 5, 
                        f"Letter B should generate exactly 5 polynomials, got {polynomial_count}")
    
    def test_complexity_requirements_summary(self):
        """Run all tests and provide summary."""
        print("\n=== Complexity Requirements Summary ===")
        
        # Test all letters
        letters = [
            ("A", 2, "simple"),
            ("C", 2, "simple"), 
            ("B", 5, "complex")
        ]
        
        results = []
        for letter, expected, complexity in letters:
            result = self.converter.text_to_desmos_functions(letter)
            actual = len(result) if result else 0
            status = "✅ PASS" if actual == expected else "❌ FAIL"
            
            print(f"Letter {letter} ({complexity}): Expected {expected}, Got {actual} - {status}")
            results.append((letter, expected, actual, actual == expected))
        
        # Verify all passed
        all_passed = all(passed for _, _, _, passed in results)
        print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        self.assertTrue(all_passed, "Not all complexity requirements were met")


if __name__ == "__main__":
    unittest.main(verbosity=2)
