#!/usr/bin/env python3
"""Test exponent encoding directly"""

def encode_exponent(power):
    """
    Encode multi-digit exponents as chained single-digit exponents for Desmos.
    E.g., 23 -> "2^3" which Desmos interprets as x^23 when used as x^2^3
    """
    if power < 10:
        return str(power)
    
    # Split the number into individual digits and chain them with ^
    digits = [d for d in str(power)]
    return "^".join(digits)

def test_encoding():
    test_cases = [
        (7, "7"),        # single digit, unchanged
        (8, "8"),        # single digit, unchanged
        (9, "9"),        # single digit, unchanged
        (10, "1^0"),     # 10 -> 1^0 (concatenates to 10)
        (11, "1^1"),     # 11 -> 1^1 (concatenates to 11)
        (23, "2^3"),     # 23 -> 2^3 (concatenates to 23)
        (45, "4^5"),     # 45 -> 4^5 (concatenates to 45)
        (123, "1^2^3"),  # 123 -> 1^2^3 (concatenates to 123)
        (999, "9^9^9"),  # 999 -> 9^9^9 (concatenates to 999)
    ]
    
    print("Testing exponent encoding:")
    print("(Desmos interprets x^2^3 as x^23)")
    print()
    
    for power, expected in test_cases:
        result = encode_exponent(power)
        status = "✓" if result == expected else "✗"
        print(f"  {power:3d} -> {result:>8} (expected {expected:>8}) {status}")
        
        # Verify the encoding produces the right digit sequence
        if '^' in result:
            digits = result.split('^')
            reconstructed = ''.join(digits)
            if reconstructed != str(power):
                print(f"    ERROR: {result} reconstructs to {reconstructed}, not {power}")

if __name__ == "__main__":
    test_encoding()
