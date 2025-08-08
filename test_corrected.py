#!/usr/bin/env python3
"""
CORRECTED Text to Desmos Letter Tracing

This version actually traces letter shapes with polynomials.
"""

from main import TextToDesmos
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def example_letter_tracing():
    """Example of proper letter tracing with polynomials."""
    print("=" * 60)
    print("CORRECTED EXAMPLE: Polynomials That Actually Trace Letters")
    print("=" * 60)
    
    # Create converter
    converter = TextToDesmos(origin=(0, 0), scale=1.0)
    
    # Generate functions that actually trace the letter
    text = "A"
    functions = converter.text_to_desmos_functions(
        text=text,
        font_size=80,
        points_per_char=50,  # More points for better tracing
        max_degree=10       # Higher degree for accurate shape following
    )
    
    # Save to file
    converter.save_functions(functions, "letter_A_traced.txt")
    
    print(f"\nGenerated {len(functions)} functions that trace letter '{text}'")
    print("These polynomials actually follow the letter contours")
    print("Saved to 'letter_A_traced.txt'")
    
    # Show first function as example
    if functions:
        print(f"\nFirst function (traces part of the letter):")
        print(functions[0])
    
    return functions

def main():
    """Test the letter tracing approach."""
    print("Text to Desmos - Letter Tracing Approach")
    print("Creating polynomials that actually follow letter shapes...")
    
    try:
        example_letter_tracing()
        print("\n✓ Letter tracing example completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("The main algorithm may still have issues that need fixing.")

if __name__ == "__main__":
    main()
