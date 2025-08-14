#!/usr/bin/env python3
"""
Test script to validate improved complexity scoring in the genetic polynomial fitter.
Tests multiple letters to ensure the system generates appropriate polynomial counts.
"""

from src.text_to_desmos import TextToDesmos

def test_polynomial_counts():
    """Test polynomial generation for multiple letters."""
    converter = TextToDesmos()
    
    test_letters = ['A', 'B', 'C', 'H', 'I', 'O']
    
    print("=== Testing Improved Complexity Scoring ===\n")
    
    results = []
    for letter in test_letters:
        print(f"Testing Letter {letter}:")
        try:
            result = converter.text_to_desmos_functions(letter)
            count = len(result) if result else 0
            
            print(f"  Generated {count} polynomials")
            if count <= 2:
                status = "✅ GOOD"
            elif count <= 4:
                status = "⚠️  ACCEPTABLE"
            else:
                status = "❌ TOO MANY"
            
            print(f"  Status: {status}")
            results.append((letter, count, status))
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append((letter, 0, "❌ ERROR"))
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    good_count = sum(1 for _, _, status in results if "GOOD" in status)
    acceptable_count = sum(1 for _, _, status in results if "ACCEPTABLE" in status)
    too_many_count = sum(1 for _, _, status in results if "TOO MANY" in status)
    error_count = sum(1 for _, _, status in results if "ERROR" in status)
    
    print(f"Good (≤2 polynomials): {good_count}")
    print(f"Acceptable (3-4 polynomials): {acceptable_count}")
    print(f"Too many (>4 polynomials): {too_many_count}")
    print(f"Errors: {error_count}")
    
    if too_many_count == 0 and error_count == 0:
        print("\n🎉 SUCCESS: Complexity scoring is working well!")
    elif too_many_count > 0:
        print(f"\n⚠️  NEEDS IMPROVEMENT: {too_many_count} letters still generate too many polynomials")
    else:
        print(f"\n❌ ISSUES: {error_count} letters had errors")

if __name__ == "__main__":
    test_polynomial_counts()
