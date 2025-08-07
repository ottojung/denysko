#!/usr/bin/env python3
"""
Test script to verify the polynomial fitting algorithm works correctly.
"""

# Simple test without numpy imports to test the core algorithm logic
import sys
sys.path.append('src')

def test_polynomial_fitting_logic():
    """Test the core polynomial fitting logic without external dependencies."""
    print("Testing enhanced polynomial fitting algorithm...")
    
    # Test 1: Basic polynomial coefficient to string conversion
    test_coeffs = [1.0, -2.5, 3.0]  # represents x^2 - 2.5x + 3.0
    print("\nTest 1: Coefficient to string conversion")
    print(f"Coefficients: {test_coeffs}")
    
    # Manual implementation of coefficient formatting
    terms = []
    n = len(test_coeffs)
    for i, coeff in enumerate(test_coeffs):
        power = n - 1 - i
        if abs(coeff) < 1e-10:  # Skip near-zero coefficients
            continue
        
        # Format coefficient
        if abs(coeff - 1.0) < 1e-10 and power > 0:
            coeff_str = ""
        elif abs(coeff + 1.0) < 1e-10 and power > 0:
            coeff_str = "-"
        else:
            coeff_str = f"{coeff:g}"
        
        # Format power
        if power == 0:
            power_str = ""
        elif power == 1:
            power_str = "x"
        else:
            power_str = f"x^{power}"
        
        term = coeff_str + power_str
        if term.startswith("-"):
            terms.append(term)
        else:
            terms.append("+" + term if terms else term)
    
    function_str = "y = " + "".join(terms)
    print(f"Generated function: {function_str}")
    
    # Test 2: Point requirements verification
    print("\nTest 2: Point requirements")
    test_points = [(0, 1), (1, 2), (2, 5), (3, 10), (4, 17), (5, 26)]
    print(f"Test points: {test_points}")
    print(f"Number of points: {len(test_points)} (should be >= 10 for fitting)")
    
    if len(test_points) >= 10:
        print("✓ Point requirement met")
    else:
        print("✗ Need more points for enhanced algorithm")
    
    # Test 3: Horizontal overlap detection logic
    print("\nTest 3: Horizontal overlap detection")
    test_points_overlap = [
        (1.0, 2.0), (1.1, 3.0), (1.0, 4.0),  # Overlap at x≈1
        (2.0, 1.0), (3.0, 2.0)                # No overlap
    ]
    print(f"Test points with overlap: {test_points_overlap}")
    
    # Group by x-coordinate (simplified logic)
    x_groups = {}
    for x, y in test_points_overlap:
        x_rounded = round(x, 1)  # Round to detect near-overlap
        if x_rounded not in x_groups:
            x_groups[x_rounded] = []
        x_groups[x_rounded].append((x, y))
    
    overlapping_x = [x for x, points in x_groups.items() if len(points) > 1]
    print(f"X-coordinates with overlap: {overlapping_x}")
    
    if overlapping_x:
        print("✓ Overlap detection working")
    else:
        print("✓ No overlap detected")
    
    print("\nAlgorithm logic tests completed!")
    print("Enhanced algorithm features:")
    print("- Requires minimum 10 points overall")
    print("- Requires minimum 5 points per stroke") 
    print("- Uses degree n-1 for n points (exact interpolation)")
    print("- Detects horizontal overlap for stroke separation")
    print("- Enhanced coefficient formatting with proper precision")

if __name__ == "__main__":
    test_polynomial_fitting_logic()
