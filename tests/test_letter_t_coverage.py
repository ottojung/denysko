#!/usr/bin/env python3
"""
Test for letter T polynomial fitting coverage with adaptive complexity.
"""

import numpy as np
import pytest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_extractor import TextExtractor
from src.genetic_polynomial_fitter import GeneticPolynomialFitter, Polynomial


class TestLetterTCoverage:
    """Test suite for letter T polynomial fitting coverage."""
    
    @pytest.fixture
    def letter_t_points(self):
        """Generate letter T points using text extractor."""
        extractor = TextExtractor()
        paths = extractor.text_to_paths("T", font_size=100)
        
        if not paths:
            pytest.skip("No paths generated for letter T")
        
        # Extract all contour points
        all_points = []
        for path in paths:
            contours = extractor.extract_contour_points(path, num_points=1000)
            for contour in contours:
                if hasattr(contour, "__len__") and len(contour) > 1:
                    all_points.extend(contour)
        
        if not all_points:
            pytest.skip("No contour points extracted for letter T")
            
        return np.array(all_points)
    
    def test_letter_t_structure_detection(self, letter_t_points):
        """Test that letter T structure is correctly identified as complex."""
        # Letter T should have high y-variation in middle x-range (vertical stem + horizontal bar)
        x_coords = letter_t_points[:, 0]
        y_coords = letter_t_points[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_span = x_max - x_min
        y_span = y_max - y_min
        
        # Check middle section for y-variation (vertical stem creates high variation)
        middle_x = x_min + 0.5 * x_span
        middle_width = 0.3 * x_span
        
        middle_mask = (x_coords >= middle_x - middle_width/2) & (x_coords <= middle_x + middle_width/2)
        middle_y = y_coords[middle_mask]
        
        if len(middle_y) > 1:
            y_variation = np.max(middle_y) - np.min(middle_y)
            complexity_ratio = y_variation / y_span
            # Letter T should have high complexity ratio (> 0.6) due to vertical stem
            assert complexity_ratio > 0.6, f"Expected high complexity ratio for letter T, got {complexity_ratio:.3f}"
    
    def test_adaptive_polynomial_count(self, letter_t_points):
        """Test that adaptive system uses appropriate number of polynomials for letter T."""
        fitter = GeneticPolynomialFitter(
            population_size=60,   # Smaller for faster testing
            generations=50,       # Fewer generations for faster testing
            max_polynomials=4,    # Allow up to 4 polynomials
            max_degree=5,         # Allow high degrees
            mutation_rate=0.05,   # Low mutation rate
            tournament_size=3     # Smaller tournament for faster testing
        )
        
        # Sample points for faster testing
        points_array = np.array(letter_t_points)
        if len(points_array) > 200:
            indices = np.linspace(0, len(points_array)-1, 200, dtype=int)
            test_points = points_array[indices]
        else:
            test_points = points_array
        
        print("Letter T adaptive fitting:")
        print(f"  - Testing {len(test_points)} points")
        
        # Fit with genetic algorithm
        genetic_polynomials = fitter.fit(test_points)
        
        # Letter T should need more polynomials (3 or 4) due to its complex structure
        assert len(genetic_polynomials) >= 3, f"Expected ≥3 polynomials for complex letter T, got {len(genetic_polynomials)}"
        
        degrees = [poly.degree for poly in genetic_polynomials]
        print(f"  - Number of polynomials: {len(genetic_polynomials)}")
        print(f"  - Polynomial degrees: {degrees}")
    
    def test_letter_t_coverage(self, letter_t_points):
        """Test that letter T achieves good coverage with adaptive approach."""
        fitter = GeneticPolynomialFitter(
            population_size=80,   # Good population for accuracy testing
            generations=80,       # Sufficient generations
            max_polynomials=4,    # Allow up to 4 polynomials
            max_degree=5,         # Allow high degrees
            mutation_rate=0.05,   # Low mutation rate
            tournament_size=5     # Good selection pressure
        )
        
        # Sample points for testing
        points_array = np.array(letter_t_points)
        
        # Sample 150 points for testing coverage
        if len(points_array) > 150:
            test_indices = np.linspace(0, len(points_array)-1, 150, dtype=int)
            test_points = points_array[test_indices]
        else:
            test_points = points_array
        
        # Sample 300 points for training
        if len(points_array) > 300:
            train_indices = np.random.choice(len(points_array), 300, replace=False)
            sampled_points = points_array[train_indices]
        else:
            sampled_points = points_array
        
        print("Letter T coverage analysis:")
        print(f"  - Tested {len(test_points)} points")
        
        # Fit with genetic algorithm
        genetic_polynomials = fitter.fit(sampled_points)
        
        # Test coverage
        covered_points = 0
        max_error = 0.0
        
        for x, y in test_points:
            # Find the best prediction among all polynomials
            min_error = float('inf')
            
            for poly in genetic_polynomials:
                pred = poly.evaluate(x)
                if pred is not None:
                    error = abs(pred - y)
                    min_error = min(min_error, error)
            
            if min_error != float('inf') and min_error <= 15.0:  # Reasonable tolerance for complex T
                covered_points += 1
                max_error = max(max_error, min_error)
        
        coverage_ratio = covered_points / len(test_points)
        degrees = [poly.degree for poly in genetic_polynomials]
        
        print(f"  - Coverage ratio: {coverage_ratio:.1%}")
        print(f"  - Covered points: {covered_points}")
        print(f"  - Max error: {max_error:.2f}")
        print(f"  - Number of polynomials: {len(genetic_polynomials)}")
        print(f"  - Polynomial degrees: {degrees}")
        
        # Letter T should achieve reasonable coverage (≥85%) with adaptive approach
        assert coverage_ratio >= 0.85, f"Expected ≥85% coverage for letter T, got {coverage_ratio:.1%} ({covered_points}/{len(test_points)} points). Degrees: {degrees}"


def run_letter_t_test():
    """Run the letter T test suite manually."""
    print("=== Letter T Coverage Test Suite ===")
    
    test_instance = TestLetterTCoverage()
    
    try:
        # Generate letter T points
        print("Generating letter T points...")
        extractor = TextExtractor()
        paths = extractor.text_to_paths("T", font_size=100)
        
        all_points = []
        for path in paths:
            contours = extractor.extract_contour_points(path, num_points=1000)
            for contour in contours:
                if hasattr(contour, "__len__") and len(contour) > 1:
                    all_points.extend(contour)
        
        if not all_points:
            print("ERROR: No points extracted for letter T")
            return False
        
        letter_t_points = np.array(all_points)
        print(f"Extracted {len(letter_t_points)} points for letter T")
        
        # Run tests
        print("\n1. Testing structure detection...")
        test_instance.test_letter_t_structure_detection(letter_t_points)
        print("✓ Structure detection passed")
        
        print("\n2. Testing adaptive polynomial count...")
        test_instance.test_adaptive_polynomial_count(letter_t_points)
        print("✓ Adaptive polynomial count test passed")
        
        print("\n3. Testing coverage with adaptive approach...")
        test_instance.test_letter_t_coverage(letter_t_points)
        print("✓ Coverage test passed")
        
        print("\n🎉 All tests passed! Letter T fitting is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_letter_t_test()
    sys.exit(0 if success else 1)
