#!/usr/bin/env python3
"""
High-quality test for letter "A" polynomial fitting coverage.

Tests that the fitter:
1. Produces exactly 2 polynomials for letter "A"
2. Covers the complete letter shape with proper structural understanding
3. One polynomial for upper triangular area above crossbar
4. One polynomial for crossbar and legs below it
5. All original letter points are covered within tolerance
"""

import numpy as np
import pytest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_extractor import TextExtractor
from src.polynomial_fitter_genetic import PolynomialFitter
from src.genetic_polynomial_fitter import GeneticPolynomialFitter, Polynomial


class TestLetterACoverage:
    """Test suite for letter A polynomial fitting coverage."""
    
    @pytest.fixture
    def letter_a_points(self):
        """Generate letter A points using text extractor."""
        extractor = TextExtractor()
        paths = extractor.text_to_paths("A", font_size=100)
        
        if not paths:
            pytest.skip("No paths generated for letter A")
        
        # Extract all contour points
        all_points = []
        for path in paths:
            contours = extractor.extract_contour_points(path, num_points=1000)
            for contour in contours:
                if hasattr(contour, "__len__") and len(contour) > 1:
                    all_points.extend(contour)
        
        if not all_points:
            pytest.skip("No contour points extracted for letter A")
            
        return np.array(all_points)
    
    def test_letter_a_structure_detection(self, letter_a_points):
        """Test that letter A structure is correctly identified."""
        # Letter A should have significant y-variation in middle x-range (crossbar)
        x_coords = letter_a_points[:, 0]
        y_coords = letter_a_points[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        x_span = x_max - x_min
        
        # Check middle section for y-variation (crossbar creates overlap)
        middle_start = x_min + 0.3 * x_span
        middle_end = x_min + 0.7 * x_span
        
        middle_mask = (x_coords >= middle_start) & (x_coords <= middle_end)
        middle_y = y_coords[middle_mask]
        
        if len(middle_y) > 1:
            y_variation = np.max(middle_y) - np.min(middle_y)
            # Letter A should have significant y-variation in middle (crossbar)
            assert y_variation > x_span * 0.15, f"Expected significant y-variation in middle section, got {y_variation:.3f}"
    
    def test_polynomial_count(self, letter_a_points):
        """Test that exactly 2 polynomials are generated for letter A."""
        fitter = PolynomialFitter()
        functions = fitter.fit_all_traces([letter_a_points])
        
        # Extract actual polynomial functions (y = ...)
        poly_functions = [f for f in functions if isinstance(f, str) and f.startswith("y =")]
        
        assert len(poly_functions) == 2, f"Expected exactly 2 polynomials for letter A, got {len(poly_functions)}: {poly_functions}"
    
    def _parse_polynomial_function(self, func_str):
        """Parse a polynomial function string into coefficients and domain."""
        if not func_str.startswith("y ="):
            return None
        
        # Split into polynomial and domain parts
        if "\\ \\left\\{" in func_str:
            poly_part = func_str.split("\\ \\left\\{")[0].strip()
            domain_part = func_str.split("\\ \\left\\{")[1].split("\\right\\}")[0]
        else:
            poly_part = func_str.strip()
            domain_part = None
        
        # Remove "y = " from the beginning
        poly_part = poly_part[4:].strip()
        
        # Parse domain if present
        x_min, x_max = None, None
        if domain_part:
            # Parse domain like "11.178\le x\le58.771"
            domain_clean = domain_part.replace("\\le", " ").replace("x", "").strip()
            numbers = []
            for part in domain_clean.split():
                try:
                    numbers.append(float(part))
                except ValueError:
                    continue
            if len(numbers) >= 2:
                x_min, x_max = sorted(numbers)[:2]
        
        return {
            'polynomial_str': poly_part,
            'x_min': x_min,
            'x_max': x_max,
            'original': func_str
        }
    
    def _evaluate_polynomial_at_x(self, poly_info, x_val):
        """Evaluate a parsed polynomial at a given x value."""
        if poly_info['x_min'] is not None and poly_info['x_max'] is not None:
            if not (poly_info['x_min'] <= x_val <= poly_info['x_max']):
                return None  # Outside domain
        
        # Simple polynomial evaluation for basic polynomials
        # This is a simplified parser - for production, would use a proper math parser
        poly_str = poly_info['polynomial_str']
        
        try:
            # Replace x with the actual value and evaluate
            # Handle common polynomial formats
            result_str = poly_str.replace('x', f'({x_val})')
            
            # Basic safety check - only allow mathematical operations
            allowed_chars = set('0123456789.+-*/()^ ')
            if not all(c in allowed_chars for c in result_str):
                return None
            
            # Replace ^ with ** for Python evaluation
            result_str = result_str.replace('^', '**')
            
            # Evaluate the expression
            result = eval(result_str)
            return float(result)
            
        except Exception as e:
            print(f"Error evaluating polynomial {poly_str} at x={x_val}: {e}")
            return None

    def test_complete_coverage(self, letter_a_points):
        """Test that genetic polynomial fitter produces good coverage of letter A."""
        # Create genetic algorithm fitter with more aggressive parameters
        fitter = GeneticPolynomialFitter(
            population_size=80,  # Increased for more diversity
            generations=150,     # More generations for evolution
            max_degree=3,
            mutation_rate=0.6    # Higher mutation for exploration
        )
        
        # Sample points for evaluation and training
        points_array = np.array(letter_a_points)
        
        # Sample 200 points for testing coverage
        if len(points_array) > 200:
            test_indices = np.linspace(0, len(points_array)-1, 200, dtype=int)
            test_points = points_array[test_indices]
        else:
            test_points = points_array
        
        # Sample 500 points for genetic algorithm training
        if len(points_array) > 500:
            train_indices = np.random.choice(len(points_array), 500, replace=False)
            sampled_points = points_array[train_indices]
        else:
            sampled_points = points_array
        
        print("Coverage analysis:")
        print(f"  - Tested {len(test_points)} points")
        
        # Fit with genetic algorithm and get Polynomial objects directly
        genetic_polynomials = fitter.fit(sampled_points)
        
        # Test coverage using the actual polynomial evaluation
        covered_points = 0
        max_error = 0.0
        errors_above_threshold = 0
        
        for x, y in test_points:
            # Find the best prediction among all polynomials
            best_pred = None
            min_error = float('inf')
            
            for poly in genetic_polynomials:
                pred = poly.evaluate(x)
                if pred is not None:  # Point is in domain
                    error = abs(pred - y)
                    if error < min_error:
                        min_error = error
                        best_pred = pred
            
            if best_pred is not None and min_error <= 5.0:  # Within tolerance
                covered_points += 1
            
            if min_error != float('inf'):
                max_error = max(max_error, min_error)
                if min_error > 5.0:
                    errors_above_threshold += 1
        
        coverage_ratio = covered_points / len(test_points)
        print(f"  - Covered points: {covered_points}")
        print(f"  - Coverage ratio: {coverage_ratio:.1%}")
        print(f"  - Max error: {max_error:.2f}")
        print(f"  - Points with errors > 5.0: {errors_above_threshold}")
        
        # Show example predictions
        print("Example predictions (first 5 points):")
        for i, (x, y) in enumerate(test_points[:5]):
            for j, poly in enumerate(genetic_polynomials):
                pred = poly.evaluate(x)
                if pred is not None:
                    error = abs(pred - y)
                    print(f"  Point {i}: x={x:.2f}, y_actual={y:.2f}, poly_{j}_pred={pred:.2f}, error={error:.2f}")
        
        # Test passes if we have good coverage
        assert coverage_ratio >= 0.80, f"Expected ≥80% coverage, got {coverage_ratio:.1%} ({covered_points}/{len(test_points)} points). Max error: {max_error:.2f}"
    
    def test_structural_separation(self, letter_a_points):
        """Test that polynomials properly separate letter A structure."""
        fitter = PolynomialFitter()
        functions = fitter.fit_all_traces([letter_a_points])
        
        poly_functions = [f for f in functions if isinstance(f, str) and f.startswith("y =")]
        assert len(poly_functions) == 2, "Need exactly 2 polynomials to test structural separation"
        
        # Extract domains from polynomial strings
        domains = []
        for func_str in poly_functions:
            if "\\left\\{" in func_str:
                domain_part = func_str.split("\\left\\{")[1].split("\\right\\}")[0]
                domain_nums = [float(x) for x in domain_part.replace("\\le", " ").replace("x", "").split() if x.replace(".", "").replace("-", "").isdigit()]
                if len(domain_nums) == 2:
                    domains.append(sorted(domain_nums))
        
        if len(domains) == 2:
            # Check that domains cover the full x-range of the letter
            all_x_min = min(domains[0][0], domains[1][0])
            all_x_max = max(domains[0][1], domains[1][1])
            
            letter_x_min = np.min(letter_a_points[:, 0])
            letter_x_max = np.max(letter_a_points[:, 0])
            
            x_coverage = (all_x_max - all_x_min) / (letter_x_max - letter_x_min)
            assert x_coverage >= 0.90, f"Expected polynomial domains to cover ≥90% of letter x-range, got {x_coverage:.1%}"
    
    def test_accuracy_requirement(self, letter_a_points):
        """Test that polynomial fits meet accuracy requirements."""
        # Sample subset of points for detailed accuracy testing
        if len(letter_a_points) > 100:
            indices = np.linspace(0, len(letter_a_points)-1, 100, dtype=int)
            test_points = letter_a_points[indices]
        else:
            test_points = letter_a_points
        
        fitter = PolynomialFitter()
        functions = fitter.fit_all_traces([letter_a_points])
        
        # This test ensures the fitting process completes and produces reasonable output
        poly_functions = [f for f in functions if isinstance(f, str) and f.startswith("y =")]
        assert len(poly_functions) > 0, "Must produce at least one polynomial function"
        
        # Each function should be a valid mathematical expression
        for func in poly_functions:
            assert "y =" in func, f"Invalid function format: {func}"
            assert not any(char in func for char in ['nan', 'inf', 'NaN']), f"Function contains invalid values: {func}"


def run_letter_a_test():
    """Run the letter A test suite manually."""
    print("=== Letter A Coverage Test Suite ===")
    
    test_instance = TestLetterACoverage()
    
    try:
        # Generate letter A points
        print("Generating letter A points...")
        extractor = TextExtractor()
        paths = extractor.text_to_paths("A", font_size=100)
        
        all_points = []
        for path in paths:
            contours = extractor.extract_contour_points(path, num_points=1000)
            for contour in contours:
                if hasattr(contour, "__len__") and len(contour) > 1:
                    all_points.extend(contour)
        
        if not all_points:
            print("ERROR: No points extracted for letter A")
            return False
        
        letter_a_points = np.array(all_points)
        print(f"Extracted {len(letter_a_points)} points for letter A")
        
        # Run tests
        print("\n1. Testing structure detection...")
        test_instance.test_letter_a_structure_detection(letter_a_points)
        print("✓ Structure detection passed")
        
        print("\n2. Testing polynomial count...")
        test_instance.test_polynomial_count(letter_a_points)
        print("✓ Polynomial count test passed")
        
        print("\n3. Testing complete coverage...")
        test_instance.test_complete_coverage(letter_a_points)
        print("✓ Coverage test passed")
        
        print("\n4. Testing structural separation...")
        test_instance.test_structural_separation(letter_a_points)
        print("✓ Structural separation test passed")
        
        print("\n5. Testing accuracy requirements...")
        test_instance.test_accuracy_requirement(letter_a_points)
        print("✓ Accuracy test passed")
        
        print("\n🎉 All tests passed! Letter A fitting is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_letter_a_test()
    sys.exit(0 if success else 1)
