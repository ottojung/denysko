#!/usr/bin/env python3
"""
Simple test to verify domain constraint logic without running the full system.
"""

# Mock the key function to show domain constraints
def create_domain_constrained_function():
    # Simulate polynomial coefficients
    coeffs = [1.5, -2.3, 0.8, 3.2]  # Example coefficients
    degree = 3
    
    # Simulate x range
    x_min = -1.5
    x_max = 2.7
    
    # Generate function string
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-16:
            continue
            
        power = degree - i
        if power == 0:
            terms.append(f"{coeff:.12f}")
        elif power == 1:
            terms.append(f"{coeff:.12f}*x")
        else:
            terms.append(f"{coeff:.12f}*x^{power}")
    
    if terms:
        func_str = " + ".join(terms).replace("+ -", "- ")
        
        # Add domain constraints to prevent curve from extending beyond its region
        # Use Desmos conditional syntax to restrict domain
        constrained_func = f"y = ({func_str}) \\{{\\{x_min:.6f} \\leq x \\leq {x_max:.6f}\\}}"
        
        print("Domain constrained function:")
        print(constrained_func)
        print(f"Active only in domain: x ∈ [{x_min:.3f}, {x_max:.3f}]")
        return constrained_func

if __name__ == "__main__":
    print("=== Domain Constraint Test ===")
    print("Testing domain-constrained polynomial generation...")
    print()
    
    result = create_domain_constrained_function()
    
    print()
    print("This function should only appear in its specific x-range in Desmos,")
    print("preventing visual clutter from curves crossing other letter parts.")
