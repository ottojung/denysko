#!/usr/bin/env python3
"""Test that multi-digit exponents are handled correctly in polynomial generation"""

# Inline the necessary parts to avoid import issues
def encode_exponent(power):
    """Encode multi-digit exponents as chained single-digit exponents for Desmos."""
    if power < 10:
        return str(power)
    digits = [d for d in str(power)]
    return "^".join(digits)

def coeffs_to_string(coeffs):
    """Convert coefficients (high->low) to function string with encoded exponents."""
    terms = []
    degree = len(coeffs) - 1
    
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-15:
            continue
            
        c_str = f"{c:.6g}"
        
        if power == 0:
            terms.append(c_str)
        elif power == 1:
            if abs(c - 1.0) < 1e-15:
                terms.append("x")
            elif abs(c + 1.0) < 1e-15:
                terms.append("-x")
            else:
                terms.append(f"{c_str}*x")
        else:
            # Use encoded exponent for multi-digit powers
            power_str = encode_exponent(power)
            if abs(c - 1.0) < 1e-15:
                terms.append(f"x^{power_str}")
            elif abs(c + 1.0) < 1e-15:
                terms.append(f"-x^{power_str}")
            else:
                terms.append(f"{c_str}*x^{power_str}")
    
    if not terms:
        return "y = 0"
        
    result = "y = " + terms[0]
    for term in terms[1:]:
        if term.startswith('-'):
            result += " - " + term[1:]
        else:
            result += " + " + term
    
    return result.replace(" + -", " - ")

# Test with a high-degree polynomial
print("Testing polynomial generation with multi-digit exponents:")
print()

test_cases = [
    ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "Degree 12 polynomial"),
    ([1] + [0]*22 + [1], "Degree 23 polynomial"),
    ([2] + [0]*99 + [3], "Degree 100 polynomial"),
]

for coeffs, description in test_cases:
    print(f"{description}:")
    result = coeffs_to_string(coeffs)
    print(f"  {result}")
    print()
