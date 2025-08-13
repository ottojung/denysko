#!/usr/bin/env python3
"""Specific test cases for polynomial counts as requested by user."""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.text_to_desmos import TextToDesmos

class TestPolynomialCounts(unittest.TestCase):
    """Test cases to ensure correct polynomial counts for different letters."""
    
    def setUp(self):
        """Set up converter for tests."""
        self.converter = TextToDesmos(origin=(0, 0), scale=1.0)
    
    def test_letter_a_two_polynomials(self):
        """Letter A should output exactly 2 polynomials."""
        functions = self.converter.text_to_desmos_functions("A")
        self.assertEqual(len(functions), 2, 
                        f"Letter A should generate 2 polynomials, but got {len(functions)}")
    
    def test_letter_c_two_polynomials(self):
        """Letter C should output exactly 2 polynomials."""
        functions = self.converter.text_to_desmos_functions("C")
        self.assertEqual(len(functions), 2, 
                        f"Letter C should generate 2 polynomials, but got {len(functions)}")
    
    def test_letter_b_five_polynomials(self):
        """Letter B should output exactly 5 polynomials."""
        functions = self.converter.text_to_desmos_functions("B")
        self.assertEqual(len(functions), 5, 
                        f"Letter B should generate 5 polynomials, but got {len(functions)}")
    
    def test_complexity_pressure_strength(self):
        """Verify that complexity pressure is working correctly."""
        # Test that simple letters don't generate too many polynomials
        simple_letters = ["A", "C", "O", "I"]
        for letter in simple_letters:
            with self.subTest(letter=letter):
                functions = self.converter.text_to_desmos_functions(letter)
                self.assertLessEqual(len(functions), 3, 
                                   f"Simple letter {letter} should use ≤3 polynomials, got {len(functions)}")
    
    def test_no_reward_for_unnecessary_complexity(self):
        """Verify there's no reward for unnecessary complexity."""
        # Generate same letter multiple times - should get consistent minimal results
        results = []
        for _ in range(2):  # Test twice to check consistency
            functions = self.converter.text_to_desmos_functions("C")
            results.append(len(functions))
        
        # Should consistently generate minimal number
        self.assertEqual(results[0], results[1], "Results should be consistent")
        self.assertEqual(results[0], 2, "Letter C should consistently use 2 polynomials")

if __name__ == "__main__":
    # Run with detailed output
    unittest.main(verbosity=2)
