#!/usr/bin/env python3
"""
Test script to validate the new principled complexity approach.
Tests specific letters to ensure convergence on expected polynomial counts.
"""

from src.text_to_desmos import TextToDesmos

def test_principled_complexity():
    """Test the principled coverage loss approach for complexity scoring."""
    converter = TextToDesmos()
    
    expected_results = {
        'A': 2,
        'C': 2, 
        'T': 2,
        'B': 5
    }
    
    print("=== Testing Principled Coverage Loss Complexity Analysis ===\n")
    
    results = []
    for letter, expected_count in expected_results.items():
        print(f"Testing Letter {letter} (expecting {expected_count} polynomials):")
        try:
            result = converter.text_to_desmos_functions(letter)
            actual_count = len(result) if result else 0
            
            print(f"  Expected: {expected_count}, Actual: {actual_count}")
            
            if actual_count == expected_count:
                status = "✅ PERFECT MATCH"
            elif abs(actual_count - expected_count) <= 1:
                status = "🟡 CLOSE (±1)"
            else:
                status = "❌ MISMATCH"
            
            print(f"  Status: {status}")
            results.append((letter, expected_count, actual_count, status))
            
            # Show first few polynomials
            if result:
                for i, poly in enumerate(result[:3], 1):
                    print(f"    Poly {i}: {poly[:60]}...")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((letter, expected_count, 0, "❌ ERROR"))
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    perfect_matches = sum(1 for _, _, _, status in results if "PERFECT" in status)
    close_matches = sum(1 for _, _, _, status in results if "CLOSE" in status)
    mismatches = sum(1 for _, _, _, status in results if "MISMATCH" in status)
    errors = sum(1 for _, _, _, status in results if "ERROR" in status)
    
    print(f"Perfect matches: {perfect_matches}/{len(expected_results)}")
    print(f"Close matches (±1): {close_matches}/{len(expected_results)}")
    print(f"Mismatches: {mismatches}/{len(expected_results)}")
    print(f"Errors: {errors}/{len(expected_results)}")
    
    if perfect_matches == len(expected_results):
        print("\n🎉 EXCELLENT: All letters converged to exactly expected polynomial counts!")
    elif perfect_matches + close_matches == len(expected_results):
        print("\n🟡 GOOD: All letters converged within ±1 of expected counts")
    else:
        print(f"\n⚠️ NEEDS WORK: {mismatches + errors} letters didn't converge properly")
        
    # Show detailed results
    print("\nDetailed Results:")
    for letter, expected, actual, status in results:
        print(f"  {letter}: {expected} → {actual} {status}")

if __name__ == "__main__":
    test_principled_complexity()
