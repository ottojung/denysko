#!/usr/bin/env python3
"""
Example usage of the updated Text to Desmos Polynomial Plotter (No Domain Constraints)

This demonstrates the new approach where multiple polynomial functions trace
the same contours without domain restrictions to highlight letter shapes.
"""

from main import TextToDesmos
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def example_no_domain_constraints():
    """Example with multiple functions per contour, no domain constraints."""
    print("=" * 60)
    print("EXAMPLE: Multiple Functions per Contour (No Domain Limits)")
    print("=" * 60)
    
    # Create converter
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    
    # Generate functions with multiple fits per contour
    text = "A"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=80,
        points_per_char=40,
        max_degree=4,
        functions_per_contour=4  # Multiple functions per contour
    )
    
    # Save to file
    converter.save_functions(functions, "no_domain_letter_A.txt")
    
    print(f"\nGenerated {len(functions)} functions for '{text}'")
    print("Each contour has multiple polynomial approximations")
    print("Functions extend beyond letter boundaries")
    print("Saved to 'no_domain_letter_A.txt'")
    
    # Show first few functions
    print("\nFirst 3 functions (no domain constraints):")
    for i, func in enumerate(functions[:3], 1):
        print(f"{i}. {func}")
    
    return functions

def example_simple_word_no_constraints():
    """Example with a simple word using the new approach."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Simple Word with Multiple Overlapping Functions")
    print("=" * 60)
    
    # Create converter with custom positioning
    converter = TextToDesmos(origin=(-5, 2), scale=1.5)
    
    # Convert word with multiple functions per contour
    text = "Hi"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=60,
        points_per_char=35,
        max_degree=3,
        functions_per_contour=3
    )
    
    # Save to file
    converter.save_functions(functions, "no_domain_word_hi.txt")
    
    print(f"\nGenerated {len(functions)} functions for '{text}'")
    print("Positioned at origin (-5, 2) with scale 1.5")
    print("Multiple polynomial fits per contour highlight letter shapes")
    print("Saved to 'no_domain_word_hi.txt'")
    
    return functions

def example_comparison_old_vs_new():
    """Compare the old approach (with constraints) vs new approach (without)."""
    print("\n" + "=" * 60)
    print("COMPARISON: Traditional vs Multi-Function Approach")
    print("=" * 60)
    
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    text = "B"
    
    # New approach: multiple functions, no constraints
    print("NEW APPROACH: Multiple overlapping functions, no domain limits")
    functions_new = converter.text_to_desmos_functions(
        text=text,
        font_size=50,
        points_per_char=25,
        max_degree=3,
        functions_per_contour=3
    )
    
    converter.save_functions(functions_new, "comparison_new_approach_B.txt")
    
    print(f"  Generated {len(functions_new)} functions")
    print("  Functions extend across entire coordinate plane")
    print("  Multiple curves highlight the same contours")
    print("  Saved to 'comparison_new_approach_B.txt'")
    
    print("\nBENEFITS of new approach:")
    print("  ✓ Letter shape highlighted by multiple overlapping curves")
    print("  ✓ No domain constraint complexity")
    print("  ✓ Curves extend naturally, showing polynomial behavior")
    print("  ✓ Easier to copy/paste into Desmos (no constraint syntax)")
    
    return functions_new

def create_usage_instructions():
    """Create updated instructions for the new approach."""
    instructions = """
# Updated Instructions for Text to Desmos (No Domain Constraints)

## What's New:
- Functions have NO domain constraints (no {x >= ... AND x <= ...} syntax)
- Multiple polynomial functions trace the SAME contours 
- Curves extend across the entire coordinate plane
- Letter shapes are highlighted by overlapping polynomial approximations

## How to Use in Desmos:

### Step 1: Generate Functions
```python
from main import TextToDesmos
converter = TextToDesmos(origin=(0, 0), scale=1.0)
functions = converter.text_to_desmos_functions("A", functions_per_contour=4)
```

### Step 2: Copy to Desmos
1. Open https://www.desmos.com/calculator
2. Copy each function line from the .txt file
3. Paste each function into a separate Desmos expression
4. All functions are simple: y = polynomial or x = polynomial

### Step 3: View Results
- Zoom out to see the full curves
- Letter shapes appear where multiple curves intersect/overlap
- Curves extend far beyond letters - this is intentional
- The overlapping regions clearly show the letter outline

## Example Functions:
```
y = -0.002341*x^3 + 0.145673*x^2 - 2.876543*x + 15.234567
y = 0.001876*x^4 - 0.123456*x^3 + 2.654321*x^2 - 8.765432*x + 23.456789  
x = 0.003214*y^2 - 0.234567*y + 8.901234
```

## Why This Approach Works:
1. **Multiple Functions per Contour**: Several polynomials approximate the same curve
2. **Natural Extension**: Polynomials extend naturally, showing their true behavior
3. **Shape Highlighting**: Where multiple curves cluster = letter outline
4. **Simplicity**: No complex domain constraint syntax
5. **Visual Impact**: Creates interesting mathematical art while showing letters

## Tips:
- Start with `functions_per_contour=3` for good balance
- Use lower `max_degree` (2-4) for smoother curves
- Adjust `scale` to make letters appropriate size
- Try different `origin` positions to place text where you want
- Experiment with `points_per_char` to control detail level

## Mathematical Beauty:
The curves show how polynomials naturally behave beyond their fitting region,
creating interesting mathematical patterns while still clearly defining the
text shapes through their intersections and clustering.
"""
    
    with open("UPDATED_DESMOS_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("\nCreated 'UPDATED_DESMOS_INSTRUCTIONS.md'")

def main():
    """Run all examples with the new no-constraints approach."""
    print("Text to Desmos Converter - Updated Examples (No Domain Constraints)")
    print("Multiple overlapping polynomials highlight letter shapes")
    
    # Run examples
    example_no_domain_constraints()
    example_simple_word_no_constraints()
    example_comparison_old_vs_new()
    
    # Create updated instructions
    create_usage_instructions()
    
    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED - NEW APPROACH")
    print("=" * 70)
    print("Files created:")
    print("- no_domain_letter_A.txt")
    print("- no_domain_word_hi.txt") 
    print("- comparison_new_approach_B.txt")
    print("- UPDATED_DESMOS_INSTRUCTIONS.md")
    print("\nKey Changes:")
    print("✓ No domain constraints - functions extend across entire plane")
    print("✓ Multiple functions per contour highlight letter shapes")
    print("✓ Simpler Desmos integration - just copy y=... or x=... functions")
    print("✓ Mathematical beauty - see polynomial behavior beyond fitting region")
    print("\nTo use: Copy functions from .txt files into Desmos expressions")

if __name__ == "__main__":
    main()
